"""Reactive mode: the robot acts only on the human's commands.

The proactive TaskManager offers robot tasks on its own -- from the task database's
trigger rules, a "robot task" chain, follow-ups such as the connector after leaving the
panel, the demo opening -- and asks permission for each. Here the robot offers nothing:
the human commands it ("pull the cables", "lift the panel", "give me the connector",
"leave"), and the command is the go-ahead, so the task starts at once.

What stays is the tracking, because a command names an action, not a piece. There is no
camera and no TCP force here (no recognition, no task detectors): the task tracker
follows the robot's own tasks -- a robot lift means the cables were pulled, holding the
panel means the human screws it, and so on (TaskManager._sync_lift) -- and what the
human says ("screw done", "cables connected", "clamped", "next piece"). The tracker
says which piece a command is for (_command_piece): the first piece, from the one the
human is on, where that task is still to do and the robot is not doing it already --
  "pull the cables", "lift the panel"   the human's piece while its cables / panel are
                                        still to do, else the next: while the human
                                        finishes piece n, that is piece n + 1
  "give me the connector"               the connector for the human's piece (the next
                                        piece's once this one has had it)
  "leave"                               the panel the robot holds
The robot names the piece it took ("Okay, I will lift the middle panel (piece 2)"), so a
wrong one can be canceled at once; the operator's live view sets the tracker right.

Within a commanded task nothing changes: the lift still turns free drive on when the
panel arrives and holds it after "done", and a brought item is handed over only when the
human takes it. But the robot does not go on to anything by itself:
  - it holds the panel until told "leave": "screw done" only updates the tracker;
  - it leaves the hand-over position a short delay after handing an item over
    (config.REACTIVE_HANDOVER_LEAVE_DELAY_S); "cancel" keeps it there;
  - nothing is offered when a task ends.
A command while the robot is busy waits in the queue and starts, still without asking,
once the robot is free -- unless its task was done or started meanwhile.
"""

import sys
from pathlib import Path

# src/ holds this layer's helpers (task database, transition table, trigger policy).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config
from events import Event, EventType, RobotTaskState as S, TaskStatus
from task_database import HUMAN, ROBOT
from task_manager import LEAVE_TASKS, ROBOT_TASK_TO_TRACKED, TaskManager

LEAVE_PANEL = next(name for name, task_id in config.TRACKED_TO_ROBOT_TASK.items()
                   if task_id == config.TASK_LEAVE)
# The robot holds a brought item out to the human.
HANDOVER_STATES = {S.R_WAITING_HANDOVER, S.R_HOLDING_HANDOVER}


class ReactiveTaskManager(TaskManager):
    """TaskManager without its initiative: robot tasks start only on command."""

    mode = "reactive"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy is not None or self.demo is not None:
            raise ValueError("Reactive mode offers nothing on its own: no trigger policy, no demo opening.")
        if self.tracker is None:
            raise ValueError("Reactive mode needs the task tracker to know which piece a command is for.")

    # -- commands ------------------------------------------------------------------------

    def _handle_robot_request(self, event: Event) -> None:
        """The human commands a robot task: it starts now, for the piece the tracker
        says, or once the robot is free."""
        task_name = event.payload.get("task_name")
        if task_name not in config.TRACKED_TO_ROBOT_TASK or not self.tracker.database.can_execute(task_name, ROBOT):
            self._say(f"Sorry, I cannot do {task_name or 'that task'}.", "Sorry, I cannot do that.")
            return
        task_id = config.TRACKED_TO_ROBOT_TASK[task_name]
        if self._under_way(task_id):
            self._say("I am already on it.", "Already on it.")
            return
        if task_id in LEAVE_TASKS:
            self._command_leave(event)
            return
        piece_id = self._command_piece(task_name)
        if piece_id is None:
            self._say(self.message_manager.get_nothing_left_message(task_name),
                      self.message_manager.get_nothing_left_message(task_name, spoken=True))
            return
        location = self._location(piece_id)
        if any((entry["task_name"], entry["piece_id"]) == (task_name, piece_id) for entry in self.waiting_triggers):
            self._say(self.message_manager.get_already_queued_message(task_id, piece_id, location),
                      self.message_manager.get_already_queued_message(task_id, piece_id, location, spoken=True))
            return
        if self.active_task is not None or self._leave_pending() or self.waiting_triggers:
            self.waiting_triggers.append({"task_name": task_name, "piece_id": piece_id, "requested": True})
            self.logger.log_message("Command queued until the robot is free.",
                                    {"task_name": task_name, "piece_id": piece_id})
            self._say(self.message_manager.get_command_message(task_id, piece_id, location, queued=True),
                      self.message_manager.get_command_message(task_id, piece_id, location, queued=True,
                                                               spoken=True))
            return
        self._execute(task_name, piece_id, event)

    def _handle_handover(self, event: Event) -> None:
        """Asking for an item ("give me the connector"): taken if the robot holds it out,
        otherwise a command to bring it."""
        active = self.active_task
        task_name = event.payload.get("task_name")
        if task_name and not (active is not None and active.state in HANDOVER_STATES):
            self.logger.log_message("Asked for an item none is held out of: commanding its bring task.",
                                    {"task_name": task_name})
            self._handle_robot_request(Event(EventType.H_REQUEST_ROBOT_TASK, event.source,
                                             payload={"task_name": task_name}))
            return
        super()._handle_handover(event)

    def _command_leave(self, event: Event) -> None:
        """"leave": release the held panel, or leave the hand-over position the robot was
        kept at -- straight away."""
        active = self.active_task
        if active is not None and active.state == S.R_HOLDING:
            self._transition(active, S.R_DONE, event, "Holding ended on the human's command to leave.")
            self.active_task = None
            self._execute(LEAVE_PANEL, active.piece_id, event)
            return
        pooled = sorted((task for task in self.pending_pool.list_all() if task.task_id in LEAVE_TASKS),
                        key=lambda task: task.piece_id)
        if pooled and active is None:
            task = pooled[0]
            self.pending_pool.remove(task.task_instance_id)
            self.active_task = task
            self._transition(task, S.R_WAITING_RESPONSE,
                             Event(EventType.H_EXECUTE_PENDING_TASK, event.source, task.task_instance_id),
                             "Pending leave commanded.")
            self._start(task, event)
            return
        self._say(self.message_manager.get_cannot_leave_message(active),
                  self.message_manager.get_cannot_leave_message(active, spoken=True))

    def _execute(self, task_name: str, piece_id: int, event: Event) -> None:
        """Start a commanded task for piece_id now."""
        self._drop_queued(task_name, piece_id)
        task = self._build_task(self._offer_context(piece_id), config.TRACKED_TO_ROBOT_TASK[task_name])
        self.active_task = task
        self._start(task, event)

    def _start(self, task, event: Event) -> None:
        """A task waiting for a yes gets it from the command itself: no permission question."""
        self._transition(task, S.R_ACCEPTED, Event(EventType.H_ACCEPT, event.source, task.task_instance_id),
                         "Commanded by the human; started without a permission question.")
        self.gh_dispatcher.dispatch_task(task)
        location = self._location(task.piece_id)
        self._say(self.message_manager.get_command_message(task.task_id, task.piece_id, location),
                  self.message_manager.get_command_message(task.task_id, task.piece_id, location, spoken=True))

    def _start_next_waiting(self) -> None:
        """Queued commands start, without asking, once the robot is free. An operator's
        offer queued from the live view is still asked about."""
        while self.active_task is None and self.waiting_triggers and not self._leave_pending():
            entry = self.waiting_triggers.popleft()
            task_name, piece_id = entry["task_name"], entry["piece_id"]
            if not entry.get("requested"):
                self._start_robot_task(task_name, piece_id)
                continue
            if not self._commandable(task_name, piece_id):
                task_id, location = config.TRACKED_TO_ROBOT_TASK[task_name], self._location(piece_id)
                self.logger.log_message("Dropped queued command: its task was done or started meanwhile.",
                                        {"task_name": task_name, "piece_id": piece_id})
                self._say(self.message_manager.get_queued_skipped_message(task_id, piece_id, location),
                          self.message_manager.get_queued_skipped_message(task_id, piece_id, location,
                                                                          spoken=True))
                continue
            self._execute(task_name, piece_id, Event(EventType.H_REQUEST_ROBOT_TASK, "task_manager",
                                                     payload={"task_name": task_name, "piece_id": piece_id}))

    # -- which piece ---------------------------------------------------------------------

    def _command_piece(self, task_name: str) -> int | None:
        """The piece a command is for: the first, from the human's piece on, where the
        task is still to do and the robot is not doing it already."""
        tracker = self.tracker
        start = tracker.human_piece_id
        if start is None:
            return None
        ids = tracker.piece_ids
        return next((piece_id for piece_id in ids[ids.index(start):]
                     if self._commandable(task_name, piece_id)), None)

    def _commandable(self, task_name: str, piece_id: int) -> bool:
        """Still to do on this piece, and not the robot's already. A task the human is
        doing counts: commanding it hands it to the robot."""
        task = self.tracker.get(task_name, piece_id)
        return (task is not None and task.status != TaskStatus.DONE
                and not (task.status == TaskStatus.WORKING and task.executor == ROBOT))

    def _under_way(self, task_id: int) -> bool:
        """The active task is this action and still at it -- not a lift that is only
        holding its panel by now, when "lift the panel" means the next one."""
        active = self.active_task
        if active is None or active.task_id not in (LEAVE_TASKS if task_id in LEAVE_TASKS else {task_id}):
            return False
        name = ROBOT_TASK_TO_TRACKED.get(active.task_id)
        tracked = self.tracker.get(name, active.piece_id) if name is not None else None
        return tracked is None or tracked.status != TaskStatus.DONE

    def _location(self, piece_id: int) -> str | None:
        return next((piece.location for piece in self.tracker.database.pieces if piece.piece_id == piece_id), None)

    # -- no initiative -------------------------------------------------------------------

    def _handle_screw_done(self, event: Event) -> None:
        """"screw done" while holding the panel confirms Screw; the robot keeps holding it
        until told to leave."""
        task = self.active_task
        if task is None or task.state != S.R_HOLDING:
            super()._handle_screw_done(event)
            return
        self.tracker.confirm_done("Screw", HUMAN, task.piece_id)
        self._withdraw_finished_offers()
        self._say(self.message_manager.get_screw_done_holding_message(),
                  self.message_manager.get_screw_done_holding_message(spoken=True))

    def _release_held_panel(self, event: Event, said: str) -> None:
        """A "panel secured" signal never releases the panel -- reactive mode runs no
        detectors, but should one report, only the human's "leave" lets go."""
        self.logger.log_message("Panel looks secured; still holding it until the human says leave.",
                                {"piece_id": self._held_piece_id})

    def _propose_followup(self, task, task_id: int) -> None:
        self.logger.log_message("No follow-up offered: the robot waits for a command.",
                                {"after": task.task_instance_id, "task_id": task_id})

    def _continue_chain(self, task) -> None:
        """No chained offer: the next robot task waits for its own command."""

    def _handover_leave_delay(self, task) -> float:
        """Leaving the hand-over position is part of the commanded bring task: never asked."""
        return config.HANDOVER_LEAVE_DELAY_S.get(task.task_id, config.REACTIVE_HANDOVER_LEAVE_DELAY_S)

    def _say(self, text: str, speech: str) -> None:
        self.cli.show_message(text, speech=speech)
