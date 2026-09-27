"""Central HRC task state-management skeleton."""

import sys
import time
from collections import deque
from pathlib import Path

# src/ holds this layer's helpers (task database, transition table, trigger policy).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config
from events import Event, EventType, RobotTaskState, TaskStatus
from models import RobotTask
from task_database import DONE_SIGNAL, HUMAN, PANEL_SECURED, PROGRESS_SIGNAL, ROBOT

S = RobotTaskState

# How a robot task's state shows up on the tracked task it performs.
ROBOT_STATE_TO_TASK_STATUS = {
    S.R_WAITING_RESPONSE: TaskStatus.PENDING,
    S.R_REFUSED: TaskStatus.PENDING,
    S.R_PENDING: TaskStatus.PENDING,
    S.R_CANCELED: TaskStatus.PENDING,
    S.R_ACCEPTED: TaskStatus.WORKING,
    S.R_DEFER: TaskStatus.WORKING,
    S.R_EXECUTING: TaskStatus.WORKING,
    S.R_PAUSED: TaskStatus.WORKING,
    S.R_REDO: TaskStatus.WORKING,
    # A brought item is done once handed over.
    S.R_WAITING_HANDOVER: TaskStatus.WORKING,
    S.R_HOLDING_HANDOVER: TaskStatus.WORKING,
    S.R_DONE: TaskStatus.DONE,
}
# A robot lift, once the human agreed to it, leads that piece's tasks until the holding
# ends (see _sync_lift): recognition is not asked meanwhile.
LIFT_MOVING_STATES = {S.R_ACCEPTED, S.R_EXECUTING, S.R_PAUSED, S.R_REDO}
LIFT_LEADING_STATES = LIFT_MOVING_STATES | {S.R_DEFER, S.R_WAITING_FREE_DRIVE, S.R_FREE_DRIVE, S.R_HOLDING}
# The active task waits for a yes / no: an answer is its own.
ASKING_STATES = {S.R_WAITING_RESPONSE, S.R_WAITING_FREE_DRIVE, S.R_WAITING_HANDOVER,
                 S.R_WAITING_HOME_PERMISSION}
# The robot still holds the panel or a brought item: canceling never offers return home.
HOLDING_STATES = {S.R_HOLDING, S.R_WAITING_HANDOVER, S.R_HOLDING_HANDOVER}
# Leaving the human's side (the held panel, or the hand-over position): while one is
# pending, the robot stays there and no other task starts.
LEAVE_TASKS = {config.TASK_LEAVE, config.TASK_LEAVE_HANDOVER}
# The active task runs: the next task of its chain may be asked about already.
ADVANCE_STATES = {S.R_EXECUTING, S.R_PAUSED, S.R_REDO}
ROBOT_TASK_TO_TRACKED = {task_id: name for name, task_id in config.TRACKED_TO_ROBOT_TASK.items()}
# The source of the recognition model's events (typed ones are "manual_recognition").
RECOGNITION_SOURCE = "recognition"
RECOGNITION_TIMER = "recognition active"


class TaskManager:
    """Owns active task state, pending tasks, and all valid transitions."""

    def __init__(
        self,
        state_machine,
        pending_pool,
        timer_manager,
        message_manager,
        cli,
        gh_dispatcher,
        ros_communication,
        logger,
        task_tracker=None,
        trigger_policy=None,
        status_callback=None,
        demo=None,
        recognition_activation_s=0.0,
    ):
        self.state_machine = state_machine
        self.pending_pool = pending_pool
        self.timer = timer_manager
        self.message_manager = message_manager
        self.cli = cli
        self.gh_dispatcher = gh_dispatcher
        self.ros = ros_communication
        self.logger = logger
        # Assembly-task tracking (task database + transition table). Without them no
        # robot task is offered automatically; human requests and follow-ups still work.
        self.tracker = task_tracker
        self.policy = trigger_policy
        self.status_callback = status_callback
        # The demo's scripted opening (demo_opening.py), or None.
        self.demo = demo
        # Recognition's own task updates count only recognition_activation_s after the
        # recognition process first reports in: its model needs that long to warm up.
        # The demo opening waits for it too.
        self.recognition_activation_s = recognition_activation_s
        self.recognition_active = recognition_activation_s <= 0
        self._recognition_seen = False
        self._demo_waiting = False

        self.active_task: RobotTask | None = None
        # A robot task asked about while the active one still runs (_offer_in_advance),
        # the task it follows, and the human's yes / later until that one succeeds.
        self.advance_task: RobotTask | None = None
        self._advance_after: str | None = None
        self._advance_answer: Event | None = None
        # Robot offers waiting for the active task to finish:
        # {"task_name", "piece_id", "requested"}.
        self.waiting_triggers = deque()
        # Piece whose panel the robot is holding, from the lift's holding state until
        # the leave succeeds or recovery takes over. Only this panel can be left.
        self._held_piece_id: int | None = None
        # The bring task just handed over: what follows it (its chain) waits until the
        # robot has left the hand-over position.
        self._handed_over: RobotTask | None = None
        # Last recognized task ignored while a robot lift leads (logged once per task).
        self._ignored_recognition: str | None = None
        self._task_instance_counts = {}
        self._round_id = 0
        self._last_status_line = None

        self.ros.publish_speed(config.DEFAULT_SPEED)
        print({"Initialize the speed to:": config.DEFAULT_SPEED})

    @property
    def held_piece_id(self) -> int | None:
        """The piece whose panel the robot holds, for the task detectors."""
        return self._held_piece_id

    def handle_event(self, event: Event) -> None:
        """Route an event to the corresponding handler."""
        self.logger.log_event(event)

        handlers = {
            EventType.RECOGNITION_TRIGGER: self._handle_human_task_update,
            EventType.HUMAN_TASK_UPDATE: self._handle_human_task_update,
            EventType.HUMAN_LOCATION_UPDATE: self._handle_human_location_update,
            EventType.TASK_SIGNAL: self._handle_task_signal,
            EventType.H_TASK_DONE: self._handle_task_done,
            EventType.H_REQUEST_ROBOT_TASK: self._handle_robot_request,
            EventType.H_NEXT_PIECE: self._handle_next_piece,
            EventType.DEMO_START: self._handle_demo_start,
            EventType.RECOGNITION_ACTIVE: self._handle_recognition_active,
            EventType.SCHEDULED_OFFER: self._handle_scheduled_offer,
            EventType.H_ACCEPT: self._handle_accept,
            EventType.H_REFUSE: self._handle_refuse,
            EventType.H_DEFER: self._handle_defer,
            EventType.RESPONSE_TIMEOUT: self._handle_response_timeout,
            EventType.DEFER_TIMEOUT: self._handle_defer_timeout,
            EventType.H_EXECUTE_PENDING_TASK: self._handle_execute_pending,
            EventType.H_PAUSE: self._handle_pause,
            EventType.H_RESUME: self._handle_resume,
            EventType.H_RESTART: self._handle_restart,
            EventType.H_CANCEL: self._handle_cancel,
            EventType.H_SPEEDUP: self._handle_speedup,
            EventType.H_SLOWDOWN: self._handle_slowdown,
            EventType.H_FREE_GO: self._handle_free_go,
            EventType.H_RETURN_HOME: self._handle_return_home,
            EventType.H_MANUAL_RECOVERY: self._handle_manual_recovery,
            EventType.H_DONE: self._handle_human_done,
            EventType.H_SCREW_DONE: self._handle_screw_done,
            EventType.H_HANDOVER: self._handle_handover,
            EventType.ROBOT_RUNNING: self._handle_robot_running,
            EventType.ROBOT_SUCCESS: self._handle_robot_success,
            EventType.ROBOT_HOMED: self._handle_robot_homed,
            # EventType.HOLD_WHEN_DISASSEMBLE: self._handle_hold_when_disassemble,
        }

        handler = handlers.get(event.event_type)
        if handler is None:
            self._log_invalid(event, "No handler registered for event.")
            return

        handler(event)
        self._start_next_waiting()
        self._report_status()

    # -- assembly-task tracking ----------------------------------------------

    def _handle_human_task_update(self, event: Event) -> None:
        """Recognition reports the human's current task; offer what it unlocks."""
        self._note_recognition(event)
        step_id = event.payload.get("step_id")
        if not isinstance(step_id, int) or not 0 <= step_id < len(config.STEP_NAMES):
            self.logger.log_message("Human task update with unknown step id.", event.payload)
            return
        self._round_id = int(event.payload.get("round_id", self._round_id))
        if self.tracker is None:
            self.logger.log_message("No task tracker; human task update not tracked.", event.payload)
            return
        task_name = config.STEP_NAMES[step_id]
        lifting = self.lift_piece_id
        warming_up = event.source == RECOGNITION_SOURCE and not self.recognition_active
        if warming_up or lifting is not None or self.demo_opening:
            # Recognition's model is still warming up. The robot's lift leads
            # (_sync_lift): Place, Align and Screw follow the robot and the dialogue until
            # the holding ends. The demo's opening is scripted. Logged once per task.
            if task_name != self._ignored_recognition:
                self._ignored_recognition = task_name
                self.logger.log_message(
                    "Recognition ignored while it warms up." if warming_up else
                    "Recognition ignored while the robot's lift leads." if lifting is not None else
                    "Recognition ignored during the scripted demo opening.",
                    {"task_name": task_name, "lift_piece_id": lifting})
            return
        self._ignored_recognition = None
        self.tracker.on_task_recognized(task_name, float(event.payload.get("progress", 0.0)))
        self._offer_triggered_tasks()

    def _offer_triggered_tasks(self) -> None:
        """Offer (or queue) every robot task the trigger rules allow right now.

        Leaving a held panel goes first, ahead of anything already queued: until the
        robot lets go of it, it can do nothing else."""
        if self.policy is None:
            return
        candidates = sorted(self.policy.candidates(), key=lambda item: not self._releases_panel(item[0]))
        for task_name, piece_id in candidates:
            releases_panel = self._releases_panel(task_name)
            if releases_panel and piece_id != self._held_piece_id:
                # Not marked offered: the rule is checked again once a panel is held.
                continue
            self.policy.mark_offered(task_name, piece_id)
            if (self.active_task is not None or self._leave_pending()
                    or (self.waiting_triggers and not releases_panel)):
                entry = {"task_name": task_name, "piece_id": piece_id, "requested": False}
                if releases_panel:
                    self.waiting_triggers.appendleft(entry)
                else:
                    self.waiting_triggers.append(entry)
                self.logger.log_message("Queued robot offer until the current task is released.",
                                        {"task_name": task_name, "piece_id": piece_id})
                continue
            self._start_robot_task(task_name, piece_id)

    @staticmethod
    def _releases_panel(task_name: str) -> bool:
        return config.TRACKED_TO_ROBOT_TASK.get(task_name) == config.TASK_LEAVE

    def _start_next_waiting(self) -> None:
        while self.active_task is None and self.waiting_triggers and not self._leave_pending():
            entry = self.waiting_triggers.popleft()
            task_name, piece_id = entry["task_name"], entry["piece_id"]
            if not self._tracked_task_open(task_name, piece_id):
                self.logger.log_message("Dropped queued robot offer: task no longer open.",
                                        {"task_name": task_name, "piece_id": piece_id})
                continue
            pooled = self._pooled_task(task_name, piece_id)
            if entry["requested"] and pooled is not None:
                self._reoffer_pooled(pooled, "Requested pending task offered again.")
            else:
                self._start_robot_task(task_name, piece_id)

    def _start_robot_task(self, task_name: str, piece_id: int) -> None:
        """Create the robot task and ask the human's permission -- always, whether the
        trigger rules chose it or the human asked for it."""
        task_id = config.TRACKED_TO_ROBOT_TASK[task_name]
        if self.policy is not None:
            self.policy.mark_offered(task_name, piece_id)
        self._propose_task(self._offer_context(piece_id), task_id)

    def _offer_context(self, piece_id: int) -> dict:
        reference = self.tracker.reference_task if self.tracker is not None else None
        return {
            "step_id": config.STEP_NAMES.index(reference) if reference in config.STEP_NAMES else -1,
            "piece_id": piece_id,
            "round_id": self._round_id,
            "progress": self.tracker.reference_progress if self.tracker is not None else 0.0,
        }

    # -- offers the workflow makes (demo opening, scheduled offers) -------------

    @property
    def demo_opening(self) -> bool:
        """The demo's scripted opening runs: recognition and detectors are ignored."""
        return self.demo is not None and self.demo.active

    def _handle_demo_start(self, event: Event) -> None:
        if self.demo is None or self.tracker is None:
            self._log_invalid(event, "No demo opening configured.")
            return
        if not self.recognition_active:
            # Recognition must be ready to take over once the opening is done.
            self._demo_waiting = True
            self.logger.log_message("Demo opening waits until recognition is active.",
                                    {"activation_s": self.recognition_activation_s})
            return
        self.demo.start(self)

    # -- recognition warm-up ----------------------------------------------------

    def _note_recognition(self, event: Event) -> None:
        """The recognition process's first event starts its warm-up: RECOGNITION_ACTIVE
        follows recognition_activation_s later."""
        if event.source != RECOGNITION_SOURCE or self._recognition_seen:
            return
        self._recognition_seen = True
        if self.recognition_active:
            return
        self.timer.schedule(RECOGNITION_TIMER, self.recognition_activation_s,
                            Event(EventType.RECOGNITION_ACTIVE, "timer"))
        self.logger.log_message("Recognition reported in; its task updates count once it has warmed up.",
                                {"activation_s": self.recognition_activation_s})

    def _handle_recognition_active(self, event: Event) -> None:
        if self.recognition_active:
            return
        self.recognition_active = True
        self._ignored_recognition = None
        self.logger.log_message("Recognition active: its task updates count from now.", {})
        if self._demo_waiting:
            self._demo_waiting = False
            self.demo.start(self)

    def schedule_offer(self, task_name: str, piece_id: int, delay: float) -> None:
        """offer_robot_task(task_name, piece_id) in delay seconds (a SCHEDULED_OFFER event)."""
        self.timer.schedule(self._offer_timer(task_name, piece_id), delay,
                            Event(EventType.SCHEDULED_OFFER, "timer",
                                  payload={"task_name": task_name, "piece_id": piece_id}))

    def cancel_scheduled_offer(self, task_name: str, piece_id: int) -> None:
        self.timer.cancel(self._offer_timer(task_name, piece_id))

    @staticmethod
    def _offer_timer(task_name: str, piece_id: int) -> str:
        return f"offer {task_name} piece {piece_id}"

    def _handle_scheduled_offer(self, event: Event) -> None:
        task_name, piece_id = event.payload.get("task_name"), event.payload.get("piece_id")
        if task_name not in config.TRACKED_TO_ROBOT_TASK:
            self.logger.log_message("Scheduled offer for a task the robot cannot do.", event.payload)
            return
        self.offer_robot_task(task_name, piece_id)

    def offer_robot_task(self, task_name: str, piece_id: int) -> None:
        """Offer a robot task the workflow calls for, not a trigger rule: now if the
        robot is free; while it still runs the task this one follows in the "robot
        task" chain, in advance; otherwise once the current task is released."""
        if not self._tracked_task_open(task_name, piece_id) or self._already_offered(task_name, piece_id):
            self.logger.log_message("Skipped robot offer: already offered, in progress or done.",
                                    {"task_name": task_name, "piece_id": piece_id})
            return
        active = self.active_task
        if active is None and not self._leave_pending():
            self._start_robot_task(task_name, piece_id)
        elif self._can_ask_in_advance(task_name, piece_id):
            self._offer_in_advance(task_name, piece_id)
        else:
            if self.policy is not None:
                self.policy.mark_offered(task_name, piece_id)
            self.waiting_triggers.append({"task_name": task_name, "piece_id": piece_id, "requested": False})
            self.logger.log_message("Queued robot offer until the current task is released.",
                                    {"task_name": task_name, "piece_id": piece_id})

    def _already_offered(self, task_name: str, piece_id: int) -> bool:
        key = (config.TRACKED_TO_ROBOT_TASK[task_name], piece_id)
        return (any(task is not None and (task.task_id, task.piece_id) == key
                    for task in (self.active_task, self.advance_task))
                or self._pooled_task(task_name, piece_id) is not None
                or any((entry["task_name"], entry["piece_id"]) == (task_name, piece_id)
                       for entry in self.waiting_triggers))

    # -- asking about the next robot task while the active one runs -------------

    @property
    def advance_answer(self) -> str | None:
        """The answer so far to the task asked in advance (H_ACCEPT / H_DEFER), or None."""
        return self._advance_answer.event_type.name if self._advance_answer is not None else None

    def _can_ask_in_advance(self, task_name: str, piece_id: int) -> bool:
        active = self.active_task
        if active is None or active.state not in ADVANCE_STATES or self.advance_task is not None:
            return False
        name = ROBOT_TASK_TO_TRACKED.get(active.task_id)
        return (self.tracker is not None and name is not None and active.piece_id == piece_id
                and self.tracker.database.next_robot_task(name) == task_name)

    def _offer_in_advance(self, task_name: str, piece_id: int) -> None:
        """Ask about the next task of the active one's chain while the active one still
        runs, so that a yes starts it the moment the active one succeeds (_start_advance)
        instead of asking only then. A no hands it to the human at once; a yes or later
        is kept until then (_answer_in_advance). The active task keeps its controls."""
        active = self.active_task
        if self.policy is not None:
            self.policy.mark_offered(task_name, piece_id)
        self._drop_queued(task_name, piece_id)
        task = self._build_task(self._offer_context(piece_id), config.TRACKED_TO_ROBOT_TASK[task_name])
        self.advance_task, self._advance_after, self._advance_answer = task, active.task_instance_id, None
        self._sync_tracker(task, Event(EventType.HUMAN_TASK_UPDATE, "task_manager"))
        self.logger.log_message("Robot task asked in advance, while the current one runs.",
                                {"task_instance_id": task.task_instance_id, "after": active.task_instance_id})
        self.cli.show_permission_request(
            self.message_manager.get_permission_message(task.task_id),
            speech=self.message_manager.get_permission_message(task.task_id, spoken=True))

    def _answer_in_advance(self, event: Event) -> bool:
        """A yes, no or later while the active task asks nothing answers the task asked
        in advance. Returns whether it did."""
        task, active = self.advance_task, self.active_task
        if task is None or self._advance_answer is not None or (
                active is not None and active.state in ASKING_STATES):
            return False
        if event.event_type == EventType.H_REFUSE:
            self.advance_task = self._advance_after = None
            self._transition(task, S.R_REFUSED, event, "Human refused the task asked in advance.")
            task.pending_reason = "refused"
            self.pending_pool.add(task)
            self.cli.show_message(self.message_manager.get_pending_message(task),
                                  speech=self.message_manager.get_pending_message(task, spoken=True))
            return True
        self._advance_answer = event
        self.logger.log_message("Answer kept until the current robot task succeeds.",
                                {"task_instance_id": task.task_instance_id, "answer": event.event_type.name})
        self.cli.show_message(self.message_manager.get_advance_acknowledgement(event.event_type),
                              speech=self.message_manager.get_advance_acknowledgement(event.event_type, spoken=True))
        return True

    def _start_advance(self, finished: RobotTask) -> bool:
        """finished succeeded: the task asked in advance after it becomes active --
        started at once after a yes, deferred after a later, asked again if still
        unanswered. Returns whether there was one."""
        task, answer = self.advance_task, self._advance_answer
        if task is None or self._advance_after != finished.task_instance_id:
            return False
        self.advance_task = self._advance_after = self._advance_answer = None
        self.active_task = task
        if answer is None:
            self._ask_permission(task)
            return True
        answer = Event(answer.event_type, answer.source, task.task_instance_id, dict(answer.payload))
        if answer.event_type == EventType.H_ACCEPT:
            self._handle_accept(answer)
        else:
            self._handle_defer(answer)
        return True

    def _drop_advance(self, message: str) -> None:
        """The task the advance question follows stopped short: nothing starts after it.
        The tracked task stays pending; the human can ask for it."""
        task = self.advance_task
        self.advance_task = self._advance_after = self._advance_answer = None
        self.logger.log_message(message, {"task_instance_id": task.task_instance_id})

    def _handle_robot_request(self, event: Event) -> None:
        """Human asks the robot for a task: no trigger rule, but the robot still asks
        permission before executing -- the request selects the task, it does not start it."""
        task_name = event.payload.get("task_name")
        if task_name not in config.TRACKED_TO_ROBOT_TASK or (
                self.tracker is not None and not self.tracker.database.can_execute(task_name, ROBOT)):
            self.cli.show_message(f"Sorry, I cannot do {task_name or 'that task'}.",
                                  speech="Sorry, I cannot do that.")
            return
        if self._releases_panel(task_name):
            if self._held_piece_id is None:
                self.cli.show_message("I am not holding a panel.", speech="I am not holding a panel.")
                return
            piece_id = self._held_piece_id
        else:
            piece_id = self._request_piece(task_name)
        if piece_id is None:
            self.cli.show_message(f"{task_name} is already done or in progress.",
                                  speech=f"{task_name} is already done or in progress.")
            return
        if self.active_task is not None or self._leave_pending():
            self.waiting_triggers.appendleft({"task_name": task_name, "piece_id": piece_id, "requested": True})
            self.cli.show_message(f"Okay, I will ask about {task_name} after the current task.",
                                  speech=f"I will ask about {task_name} next.")
            return
        pooled = self._pooled_task(task_name, piece_id)
        if pooled is not None:
            self._reoffer_pooled(pooled, "Human requested the pending task; asking permission.")
            return
        self.logger.log_message("Human requested robot task.", {"task_name": task_name, "piece_id": piece_id})
        self._start_robot_task(task_name, piece_id)

    def _request_piece(self, task_name: str) -> int | None:
        if self.tracker is None:
            return self.active_task.piece_id if self.active_task is not None else 0
        piece_id = self.tracker.first_open_piece_for(task_name)
        if piece_id is None or not self._tracked_task_open(task_name, piece_id):
            return None
        return piece_id

    def _handle_task_done(self, event: Event) -> None:
        """Human confirms a task (or, with no task name, the one being worked on), on
        payload piece_id if given, else its first open piece."""
        if self.tracker is None:
            self._log_invalid(event, "No task tracker to confirm tasks with.")
            return
        task = self.tracker.confirm_done(event.payload.get("task_name"), HUMAN, event.payload.get("piece_id"))
        if task is None:
            name = event.payload.get("task_name") or "working task"
            self.cli.show_message(f"There is no open {name} to mark done.", speech="Nothing to mark done.")
            return
        self._withdraw_finished_offers()
        self.cli.show_message(f"Okay, {task.task_name} is done.", speech=f"{task.task_name} done.")
        self._offer_triggered_tasks()

    def _handle_task_signal(self, event: Event) -> None:
        """A detector reports a signal on a task (task_transition_detector.py): progress
        moves the task on, a done signal confirms it, any other value is stored for the
        trigger rules -- and "panel secured" on the held panel asks to release it and
        leave."""
        payload = event.payload
        task_name, piece_id = payload.get("task_name"), payload.get("piece_id")
        name, value = payload.get("signal"), payload.get("value", True)
        tracked = self.tracker.get(task_name, piece_id) if self.tracker is not None else None
        if tracked is None:
            self.logger.log_message("Signal for an unknown task ignored.", payload)
            return
        if self.demo_opening:
            self.logger.log_message("Signal ignored: the demo opening is scripted.", payload)
            return
        if name == PROGRESS_SIGNAL:
            self.tracker.set_progress(task_name, piece_id, float(value))
            self._offer_triggered_tasks()
            return
        if name != DONE_SIGNAL:
            self.tracker.set_signal(task_name, piece_id, name, value)
            if name == PANEL_SECURED and value and self._holds_panel(piece_id):
                self._release_secured_panel(event)
            self._offer_triggered_tasks()
            return
        if not value or tracked.status == TaskStatus.DONE:
            self.logger.log_message("Done signal ignored: the task is already done.", payload)
            return

        active = self.active_task
        if task_name == "Screw" and self._holds_panel(piece_id):
            # Exactly what the human saying "screw done" does: it ends the holding and
            # asks about leaving. The robot says why it asks.
            self.cli.show_message("Screwing looks finished.", speech="Screwing looks finished.")
            self._handle_screw_done(Event(EventType.H_SCREW_DONE, event.source,
                                          active.task_instance_id, payload=dict(payload)))
            return
        executor = HUMAN if self.tracker.database.can_execute(task_name, HUMAN) else ROBOT
        if self.tracker.confirm_done(task_name, executor, piece_id) is None:
            return
        self._withdraw_finished_offers()
        self.cli.show_message(f"{task_name} looks done.", speech=f"{task_name} done.")
        self._offer_triggered_tasks()

    def _handle_next_piece(self, event: Event) -> None:
        if self.tracker is None:
            self._log_invalid(event, "No task tracker.")
            return
        piece_id = self.tracker.advance_piece()
        text = "All pieces are done." if piece_id is None else f"Moving on to piece {piece_id}."
        self.cli.show_message(text, speech=text)

    def _withdraw_finished_offers(self) -> None:
        """Drop robot offers for tasks that are done now (by the human, or inferred)."""
        if self.tracker is None:
            return
        for task in self.pending_pool.list_all():
            name = ROBOT_TASK_TO_TRACKED.get(task.task_id)
            if name is not None and not self._tracked_task_open(name, task.piece_id, allow_working=True):
                self.pending_pool.remove(task.task_instance_id)
                self.logger.log_message("Removed pending robot task: task already done.",
                                        {"task_instance_id": task.task_instance_id})
        task = self.active_task
        name = ROBOT_TASK_TO_TRACKED.get(task.task_id) if task is not None else None
        if (name is not None and task.state == S.R_WAITING_RESPONSE
                and not self._tracked_task_open(name, task.piece_id, allow_working=True)):
            self.timer.cancel_response_timer()
            self._transition(task, S.R_CANCELED,
                             Event(EventType.H_TASK_DONE, "task_manager", task.task_instance_id),
                             "Offer withdrawn: the human already did the task.")
            self.active_task = None

    def _tracked_task_open(self, task_name: str, piece_id: int, allow_working=False) -> bool:
        if self.tracker is None:
            return True
        if not self.tracker.database.in_task_lists(task_name):
            # Not tracked per piece (Leave from the panel): the pending pool and the
            # trigger policy's once-per-piece offer are its only state.
            return True
        task = self.tracker.get(task_name, piece_id)
        if task is None:
            return False
        return task.status != TaskStatus.DONE and (allow_working or task.status != TaskStatus.WORKING)

    def _pooled_task(self, task_name: str, piece_id: int) -> RobotTask | None:
        task_id = config.TRACKED_TO_ROBOT_TASK.get(task_name)
        for task in self.pending_pool.list_all():
            if task.task_id == task_id and task.piece_id == piece_id:
                return task
        return None

    def _leave_pending(self) -> bool:
        # A pending leave task keeps the robot at the held panel or the hand-over position.
        return any(task.task_id in LEAVE_TASKS for task in self.pending_pool.list_all())

    def _sync_tracker(self, task: RobotTask, event: Event) -> None:
        """Mirror a robot task's state onto the tracked task it performs."""
        name = ROBOT_TASK_TO_TRACKED.get(task.task_id)
        if self.tracker is None or name is None:
            return
        if task.task_id == config.TASK_LIFT_PANEL:
            self._sync_lift(task, event)
            return
        status = ROBOT_STATE_TO_TASK_STATUS.get(task.state)
        if status is not None:
            self.tracker.set_robot_status(name, task.piece_id, status)

    def _sync_lift(self, task: RobotTask, event: Event) -> None:
        """Once the human agrees to a robot lift, the lift's states -- not recognition
        -- say where that piece's tasks are:
          accepted, executing        the robot lifts the panel and carries it to the
                                     assembly location: Lift and Place working (robot),
                                     and the cables are pulled (Pull Cables done)
          waiting for free drive,    it has arrived: Lift and Place done; in free drive
          free drive                 the human aligns the panel: Align working
          holding                    Align done -- after "adjustment done", or declining
                                     free drive when no adjustment is needed (inferred) --
                                     and the human screws: Screw working until screw done
        Offered, refused or deferred, the lift is only pending or working itself; a lift
        stopped before it arrived hands Lift and Place back (pending)."""
        tracker, piece = self.tracker, task.piece_id
        if task.state in LIFT_MOVING_STATES:
            tracker.infer_done_before("Lift", piece)
            tracker.set_robot_status("Lift", piece, TaskStatus.WORKING)
            tracker.set_robot_status("Place", piece, TaskStatus.WORKING)
        elif task.state in (S.R_WAITING_FREE_DRIVE, S.R_FREE_DRIVE):
            # Free drive may follow the arrival directly (config.LIFT_ASKS_FREE_DRIVE).
            tracker.set_robot_status("Lift", piece, TaskStatus.DONE)
            tracker.set_robot_status("Place", piece, TaskStatus.DONE)
            if task.state == S.R_FREE_DRIVE:
                tracker.start_task("Align", piece)
        elif task.state == S.R_HOLDING:
            adjusted = event.event_type == EventType.H_DONE
            tracker.confirm_done("Align", HUMAN, piece, inferred=not adjusted)
            tracker.start_task("Screw", piece)
        elif (status := ROBOT_STATE_TO_TASK_STATUS.get(task.state)) is not None and status != TaskStatus.DONE:
            tracker.set_robot_status("Lift", piece, status)
            place = tracker.get("Place", piece)
            if (status == TaskStatus.PENDING and place is not None
                    and place.status == TaskStatus.WORKING and place.executor == ROBOT):
                tracker.set_robot_status("Place", piece, TaskStatus.PENDING)

    @property
    def lift_piece_id(self) -> int | None:
        """The piece whose lift the robot leads now -- from the human's yes until the
        holding ends -- or None."""
        task = self.active_task
        if task is not None and task.task_id == config.TASK_LIFT_PANEL and task.state in LIFT_LEADING_STATES:
            return task.piece_id
        return None

    def _report_status(self) -> None:
        if self.tracker is None:
            return
        line = self.tracker.status_line()
        if line != self._last_status_line:
            self._last_status_line = line
            self.logger.log_message("Task status.", {"line": line})
            if self.status_callback is not None:
                self.status_callback(line)

    # -- robot tasks ------------------------------------------------------------

    def _propose_task(self, context: dict, task_id: int) -> None:
        """Ask permission for one robot action, retaining its human context."""
        task = self._build_task(context, task_id)
        self.active_task = task
        self._sync_tracker(task, Event(EventType.HUMAN_TASK_UPDATE, "task_manager"))
        self.logger.log_message("Task entered R_WAITING_RESPONSE.", {"task_instance_id": task.task_instance_id})
        self._ask_permission(task)

    def _build_task(self, context: dict, task_id: int) -> RobotTask:
        """A new robot task waiting for permission, retaining its human context."""
        now = time.time()
        return RobotTask(
            task_instance_id=self._build_task_instance_id(context["round_id"], task_id, context["piece_id"]),
            step_id=context["step_id"],
            task_id=task_id,
            piece_id=context["piece_id"],
            round_id=context["round_id"],
            state=RobotTaskState.R_WAITING_RESPONSE,
            speed=config.DEFAULT_SPEED,
            progress=context.get("progress", 0.0),
            created_at=now,
            updated_at=now,
        )

    def _ask_permission(self, task: RobotTask) -> None:
        """The only way into execution: every robot task waits here for H_ACCEPT."""
        message = self.message_manager.get_permission_message(task.task_id)
        self.cli.show_permission_request(
            message, speech=self.message_manager.get_permission_message(task.task_id, spoken=True),
        )
        duration = self._task_duration(task, "response_timeout_seconds", config.RESPONSE_TIMEOUT_SECONDS)
        self.timer.start_response_timer(task.task_instance_id, duration)

    def _task_duration(self, task: RobotTask, key: str, default: float) -> float:
        return config.TASK_TIMINGS.get(task.task_id, {}).get(key, default)

    def _propose_followup(self, task: RobotTask, task_id: int) -> None:
        name = ROBOT_TASK_TO_TRACKED.get(task_id)
        if name is not None:
            if not self._tracked_task_open(name, task.piece_id):
                self.logger.log_message("Skipped follow-up: task no longer open.",
                                        {"task_name": name, "piece_id": task.piece_id})
                return
            if self.policy is not None:
                self.policy.mark_offered(name, task.piece_id)
            self._drop_queued(name, task.piece_id)
        self._propose_task({
            "step_id": config.HUMAN_SCREW_DONE,
            "piece_id": task.piece_id,
            "round_id": task.round_id,
            "progress": 1.0,
        }, task_id)

    def _continue_chain(self, task: RobotTask) -> None:
        """Offer the robot task after this one in its "robot task" chain (task
        database), for the same piece: straight away, without waiting for recognition,
        but asking permission like any other offer."""
        name = ROBOT_TASK_TO_TRACKED.get(task.task_id)
        if self.tracker is None or name is None:
            return
        next_name = self.tracker.database.next_robot_task(name)
        if next_name is None or next_name not in config.TRACKED_TO_ROBOT_TASK:
            return
        if not self._tracked_task_open(next_name, task.piece_id):
            self.logger.log_message("Skipped chained robot task: no longer open.",
                                    {"task_name": next_name, "piece_id": task.piece_id})
            return
        if self._pooled_task(next_name, task.piece_id) is not None:
            self.logger.log_message("Skipped chained robot task: already refused or unanswered.",
                                    {"task_name": next_name, "piece_id": task.piece_id})
            return
        self._drop_queued(next_name, task.piece_id)
        self.logger.log_message("Chained robot task offered.",
                                {"after": name, "task_name": next_name, "piece_id": task.piece_id})
        self._start_robot_task(next_name, task.piece_id)

    def _drop_queued(self, task_name: str, piece_id: int) -> None:
        self.waiting_triggers = deque(
            entry for entry in self.waiting_triggers
            if (entry["task_name"], entry["piece_id"]) != (task_name, piece_id))

    def _handle_human_location_update(self, event: Event) -> None:
        """Forward the human's world-frame position (plus, if this frame
        had one, their pelvis-relative posture keypoints) to both
        consumers -- Grasshopper (visualization) and ROS (path planning).
        Stateless: doesn't touch active_task/the state machine, just
        relays. The first one also starts recognition's warm-up."""
        self._note_recognition(event)
        xyz = (event.payload["x"], event.payload["y"], event.payload["z"])
        timestamp = event.payload.get("timestamp")
        keypoints = event.payload.get("keypoints")
        velocity = event.payload.get("velocity")
        # UDPEventReceiver is message transfer
        # self.gh_dispatcher.dispatch_human_location(xyz, timestamp)
        # ROS message transfer
        self.ros.publish_human_location(xyz, timestamp, keypoints, velocity)

    def _handle_accept(self, event: Event) -> None:
        if self._answer_in_advance(event):
            return
        if self.active_task is not None:
            if self.active_task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
                self._handle_free_go(event)
                return
            if self.active_task.state == RobotTaskState.R_WAITING_HOME_PERMISSION:
                self._handle_return_home(event)
                return
            if self.active_task.state == RobotTaskState.R_WAITING_HANDOVER:
                self._handle_handover(event)
                return
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Human accepted task.")
        self.gh_dispatcher.dispatch_task(task)
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_refuse(self, event: Event) -> None:
        if self._answer_in_advance(event):
            return
        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_WAITING_RESPONSE,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                RobotTaskState.R_WAITING_HANDOVER,
                RobotTaskState.R_WAITING_HOME_PERMISSION,
            },
        )
        if task is None:
            return

        if task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._enter_holding(task, event)
            return

        if task.state == RobotTaskState.R_WAITING_HANDOVER:
            self._transition(task, RobotTaskState.R_HOLDING_HANDOVER, event,
                             "Human not ready; holding the item until asked for it.")
            self.cli.show_message(
                self.message_manager.get_handover_wait_message(task),
                speech=self.message_manager.get_handover_wait_message(task, spoken=True),
            )
            return

        if task.state == RobotTaskState.R_WAITING_HOME_PERMISSION:
            self._enter_manual_recovery(task, event)
            return

        self.timer.cancel_response_timer()
        self._move_to_pending(task, RobotTaskState.R_REFUSED, event, "refused", "Human refused task.")

    def _move_to_pending(self, task: RobotTask, state: RobotTaskState, event: Event,
                         reason: str, message: str) -> None:
        self._transition(task, state, event, message)
        task.pending_reason = reason
        self.pending_pool.add(task)
        self.active_task = None
        self.cli.show_message(
            self.message_manager.get_pending_message(task),
            speech=self.message_manager.get_pending_message(task, spoken=True),
        )

    def _handle_defer(self, event: Event) -> None:
        if self._answer_in_advance(event):
            return
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_DEFER, event, "Human deferred task.")
        duration = self._task_duration(task, "defer_seconds", config.DEFER_SECONDS)
        task.defer_seconds = duration
        self.cli.show_message(
            self.message_manager.get_defer_message(task, duration),
            speech=self.message_manager.get_defer_message(task, duration, spoken=True),
        )
        self.timer.start_defer_timer(task.task_instance_id, duration)

    def _handle_response_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self._move_to_pending(task, RobotTaskState.R_PENDING, event, "timeout", "Response timeout.")

    def _handle_defer_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_DEFER)
        if task is None:
            return

        self.gh_dispatcher.dispatch_task(task)
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Deferred task dispatched; waiting for robot running status.")

    def _handle_execute_pending(self, event: Event) -> None:
        if self.active_task is not None:
            self._log_invalid(event, "Cannot execute pending task while active task exists.")
            return
        if event.task_instance_id is None or not self.pending_pool.contains(event.task_instance_id):
            self._log_invalid(event, "Pending task id not found.")
            return

        if any(
            task.task_id in LEAVE_TASKS and task.task_instance_id != event.task_instance_id
            for task in self.pending_pool.list_all()
        ):
            self._log_invalid(event, "Execute the pending leave task before starting another task.")
            return

        self._reoffer_pooled(self.pending_pool.get(event.task_instance_id),
                             "Pending task offered again; waiting for permission.", event)

    def _reoffer_pooled(self, task: RobotTask, message: str, event: Event | None = None) -> None:
        """Bring a refused/timed-out robot task back from the pending pool and ask
        permission again; it is dispatched only on H_ACCEPT, like any other offer."""
        if event is None:
            event = Event(EventType.H_EXECUTE_PENDING_TASK, "task_manager", task.task_instance_id)
        self.pending_pool.remove(task.task_instance_id)
        self.active_task = task
        self._transition(task, RobotTaskState.R_WAITING_RESPONSE, event, message)
        self._ask_permission(task)

    def _handle_pause(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        self.ros.publish_pause()
        self._transition(task, RobotTaskState.R_PAUSED, event, "ROS pause published.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_resume(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_PAUSED)
        if task is None:
            return

        self.ros.publish_resume(task.speed)
        self._transition(
            task,
            RobotTaskState.R_EXECUTING,
            event,
            f"Robot resumed at saved speed {task.speed}.",
        )
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_restart(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED},
        )
        if task is None:
            return

        self.ros.publish_restart()
        task.robot_running_received = False
        task.robot_success_received = False
        self._transition(task, RobotTaskState.R_REDO, event, "ROS restart published; waiting for robot running status.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_cancel(self, event: Event) -> None:

        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_ACCEPTED,
                RobotTaskState.R_EXECUTING,
                RobotTaskState.R_PAUSED,
                RobotTaskState.R_DEFER,
                RobotTaskState.R_REDO,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                RobotTaskState.R_FREE_DRIVE,
                RobotTaskState.R_HOLDING,
                RobotTaskState.R_WAITING_HANDOVER,
                RobotTaskState.R_HOLDING_HANDOVER,
                RobotTaskState.R_MANUAL_RECOVERY,
            },
        )
        if task is None:
            return

        if task.state == RobotTaskState.R_DEFER:
            self.timer.cancel_defer_timer()
            if task.task_id == config.TASK_LEAVE:
                self._enter_holding(task, event)
            elif task.task_id == config.TASK_LEAVE_HANDOVER:
                self._move_to_pending(task, RobotTaskState.R_REFUSED, event, "refused",
                                      "Delayed leave canceled; staying at the hand-over position.")
            else:
                self._finish_canceled_task(task, event, "Deferred task canceled.")
            return

        was_holding = task.state in HOLDING_STATES
        self.ros.publish_cancel()
        if task.free_drive_active:
            self.ros.publish_free_drive(False)
            task.free_drive_active = False
            self._finish_canceled_task(task, event, "Free-drive disabled; task canceled.")
            return

        if task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._finish_canceled_task(task, event, "Task canceled before free-drive started.")
            return

        self._transition(
            task,
            RobotTaskState.R_RECOVERY_EVALUATING,
            event,
            "Stop published; evaluating recovery strategy.",
        )
        time.sleep(config.RECOVERY_STOP_DELAY_SECONDS)
        joint_positions = self.ros.get_latest_joint_positions()
        gripper_has_object = self.ros.get_latest_gripper_has_object()
        if was_holding:
            # The workflow still owns a held panel or item; an old open-gripper sample
            # must not authorize returning home from this state.
            gripper_has_object = True

        return_check = self._can_return_home(joint_positions, gripper_has_object)

        #can return home directly with permission
        if return_check:
            decision_event = Event(
                event_type=EventType.RECOVERY_HOME_AVAILABLE,
                source="task_manager",
                task_instance_id=task.task_instance_id,
            )
            self._transition(
                task,
                RobotTaskState.R_WAITING_HOME_PERMISSION,
                decision_event,
                "Recovery conditions allow return home; waiting for permission.",
            )
            self.cli.show_message(
                self.message_manager.get_return_home_permission_message(),
                speech=self.message_manager.get_return_home_permission_message(spoken=True),
            )
            return

        decision_event = Event(
            event_type=EventType.RECOVERY_MANUAL_REQUIRED,
            source="task_manager",
            task_instance_id=task.task_instance_id,
        )
        self._enter_manual_recovery(task, decision_event)

    def _handle_speedup(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        # Speed Control Formula: increase speed by SPEED_STEP, but do not exceed MAX_SPEED
        task.speed = min(task.speed + config.SPEED_STEP, config.MAX_SPEED)

        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed increased.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_slowdown(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        # Speed Control Formula: decrease speed by SPEED_STEP, but do not go below MIN_SPEED
        task.speed = max(task.speed - config.SPEED_STEP, config.MIN_SPEED)

        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed decreased.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_free_go(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_FREE_DRIVE)
        if task is None:
            return

        self._enable_free_drive(task, event, "Human approved free-drive mode; free-drive enabled.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(EventType.H_FREE_GO),
            speech=self.message_manager.get_acknowledgement(EventType.H_FREE_GO, spoken=True),
        )

    def _enable_free_drive(self, task: RobotTask, event: Event, message: str) -> None:
        self.ros.publish_free_drive(True)
        task.free_drive_active = True
        self._transition(task, RobotTaskState.R_FREE_DRIVE, event, message)

    def _handle_return_home(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_HOME_PERMISSION)
        if task is None:
            return

        self.ros.publish_return_home()
        self._transition(
            task,
            RobotTaskState.R_RETURNING_HOME,
            event,
            "Return-home command published.",
        )
        self.cli.show_message(
            self.message_manager.get_acknowledgement(EventType.H_RETURN_HOME),
            speech=self.message_manager.get_acknowledgement(EventType.H_RETURN_HOME, spoken=True),
        )

    def _handle_manual_recovery(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_HOME_PERMISSION)
        if task is None:
            return

        self._enter_manual_recovery(task, event)

    def _enter_manual_recovery(self, task: RobotTask, event: Event) -> None:
        self.ros.publish_free_drive(True)
        task.free_drive_active = True
        self._transition(
            task,
            RobotTaskState.R_MANUAL_RECOVERY,
            event,
            "Manual recovery required; free-drive enabled.",
        )
        self.cli.show_message(
            self.message_manager.get_manual_recovery_message(),
            speech=self.message_manager.get_manual_recovery_message(spoken=True),
        )

    def _can_return_home(
        self,
        joint_positions: list[float] | None,
        gripper_has_object: bool | None,
    ) -> bool:
        return (
            config.RETURN_HOME_RECOVERY_ENABLED
            and gripper_has_object is False
            and self._is_in_safe_return_zone(joint_positions)
        )

    def _is_in_safe_return_zone(
        self,
        joint_positions: list[float] | None,
    ) -> bool:
        ranges = config.SAFE_RETURN_JOINT_RANGES
        if joint_positions is None or ranges is None or len(joint_positions) != len(ranges):
            return False

        return all(
            lower <= position <= upper
            for position, (lower, upper) in zip(joint_positions, ranges)
        )

    def _finish_canceled_task(self, task: RobotTask, event: Event, message: str) -> None:
        self._transition(task, RobotTaskState.R_CANCELED, event, message)
        self.active_task = None
        if (task.task_id in (config.TASK_LIFT_PANEL, config.TASK_LEAVE)
                and task.piece_id == self._held_piece_id):
            # Recovery (homing or manual) has taken the panel out of the robot's hands.
            self._held_piece_id = None
        self.cli.show_message(
            self.message_manager.get_acknowledgement(EventType.H_CANCEL),
            speech=self.message_manager.get_acknowledgement(EventType.H_CANCEL, spoken=True),
        )

    def _handle_human_done(self, event: Event) -> None:
        if self.active_task is None and self.tracker is not None:
            # No robot task to report to: "done" confirms the human's own task.
            self._handle_task_done(Event(EventType.H_TASK_DONE, event.source))
            return
        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_EXECUTING,
                RobotTaskState.R_FREE_DRIVE,
                RobotTaskState.R_MANUAL_RECOVERY,
            },
        )
        if task is None:
            return

        if task.state in {RobotTaskState.R_FREE_DRIVE, RobotTaskState.R_MANUAL_RECOVERY}:
            is_manual_recovery = task.state == RobotTaskState.R_MANUAL_RECOVERY
            self.ros.publish_free_drive(False)
            task.free_drive_active = False
            if not is_manual_recovery:
                self._enter_holding(task, event)
                return
            self._finish_canceled_task(task, event, "Manual recovery completed; free-drive disabled.")
            return

        self.ros.publish_human_done()
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Human-done published.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _enter_holding(self, task: RobotTask, event: Event) -> None:
        self._transition(task, RobotTaskState.R_HOLDING, event, "Panel held; waiting for screw done.")
        self._held_piece_id = task.piece_id
        message = self.message_manager.get_holding_message()
        speech = self.message_manager.get_holding_message(spoken=True)
        if event.event_type == EventType.H_CANCEL:
            message = "The delayed leave action is canceled. " + message
            speech = "Leave canceled. " + speech
        self.cli.show_message(message, speech=speech)

    def _handle_screw_done(self, event: Event) -> None:
        """ "screw done": confirms Screw. While the robot holds the panel, this is also
        what ends the holding, and the database's "Leave from the panel" rule (previous
        task Screw, "Done signal") then offers the leave -- ahead of anything else
        Screw unlocks, such as Bring Tool. The robot may also have been asked to leave
        already, because the panel looked secured (_release_secured_panel): then this
        only confirms that panel's Screw."""
        active = self.active_task
        if active is None or active.state != RobotTaskState.R_HOLDING:
            if active is not None and active.task_id == config.TASK_LIFT_PANEL:
                # Lifting or adjusting the panel: the screwing has not started.
                self._log_invalid(event, "Screw done while the robot is handling the panel.")
                return
            # Not holding: only the human's Screw is confirmed -- on the panel the robot
            # is leaving, if any.
            self._handle_task_done(Event(EventType.H_TASK_DONE, event.source,
                                         payload={"task_name": "Screw", "piece_id": self._held_piece_id}))
            return

        task = self._require_active(event, RobotTaskState.R_HOLDING)
        if task is None:
            return
        self._transition(task, RobotTaskState.R_DONE, event, "Screwing finished; the panel can be released.")
        self.active_task = None
        if self.tracker is not None:
            self.tracker.confirm_done("Screw", HUMAN, task.piece_id)
            self._withdraw_finished_offers()
        if task.task_id == config.TASK_LEAVE or self.policy is None:
            # A delayed leave canceled back to holding (its rule already fired for this
            # piece), or no trigger rules at all: ask about leaving again directly.
            self._propose_followup(task, config.TASK_LEAVE)
        self._offer_triggered_tasks()
        if not self._leave_offered(task.piece_id):
            self.logger.log_message(
                "Screw done, but no leave was offered; the robot keeps holding the panel. "
                "Check the 'Leave from the panel' rule in the task database.",
                {"piece_id": task.piece_id})

    def _holds_panel(self, piece_id) -> bool:
        active = self.active_task
        return (active is not None and active.state == RobotTaskState.R_HOLDING
                and piece_id == self._held_piece_id)

    def _release_secured_panel(self, event: Event) -> None:
        """A detector says the held panel is secured (every screw counted, or the TCP
        force): the holding ends and the robot asks to release the panel and leave.
        Screw stays open -- recognition, or the human saying "screw done", confirms it."""
        task = self.active_task
        self.cli.show_message("The panel looks secured.", speech="The panel looks secured.")
        self._transition(task, RobotTaskState.R_DONE, event, "Panel secured; asking to release it.")
        self.active_task = None
        self._propose_followup(task, config.TASK_LEAVE)

    def _leave_offered(self, piece_id: int) -> bool:
        active = self.active_task
        return ((active is not None and active.task_id == config.TASK_LEAVE)
                or any(self._releases_panel(entry["task_name"]) and entry["piece_id"] == piece_id
                       for entry in self.waiting_triggers))

    def _handle_handover(self, event: Event) -> None:
        """The human takes the item the robot brought -- a yes to "can I hand it over?",
        or asking for it ("give me the tool"): open the gripper, then leave the hand-over
        position, a robot task of its own: after a short delay for the items in
        config.HANDOVER_LEAVE_DELAY_S, otherwise asking first (yes / no / later, like the
        leave from the panel)."""
        task = self._require_active_in(event, {RobotTaskState.R_WAITING_HANDOVER,
                                               RobotTaskState.R_HOLDING_HANDOVER})
        if task is None:
            return
        self.ros.publish_gripper_open()
        self._transition(task, RobotTaskState.R_DONE, event, "Gripper opened; item handed over.")
        self.active_task = None
        self._handed_over = task
        context = {"step_id": task.step_id, "piece_id": task.piece_id,
                   "round_id": task.round_id, "progress": task.progress}
        delay = config.HANDOVER_LEAVE_DELAY_S.get(task.task_id)
        if delay is None:
            self.cli.show_message(self.message_manager.get_handed_over_message(task),
                                  speech=self.message_manager.get_handed_over_message(task, spoken=True))
            self._propose_task(context, config.TASK_LEAVE_HANDOVER)
        else:
            self._leave_handover_after(task, context, delay)
        if self._advance_after == task.task_instance_id:
            # Asked in advance to follow the bring task: it starts once the robot has left.
            self._advance_after = self.active_task.task_instance_id

    def _leave_handover_after(self, brought: RobotTask, context: dict, delay: float) -> None:
        """Leave the hand-over position without asking, delay seconds after saying so:
        a delayed start, as after "later" -- cancel keeps the robot there (pending)."""
        leave = self._build_task(context, config.TASK_LEAVE_HANDOVER)
        self.active_task = leave
        self._transition(leave, RobotTaskState.R_DEFER,
                         Event(EventType.H_DEFER, "task_manager", leave.task_instance_id),
                         "Leaving the hand-over position without asking.")
        leave.defer_seconds = delay
        self.cli.show_message(
            self.message_manager.get_handed_over_message(brought, delay),
            speech=self.message_manager.get_handed_over_message(brought, delay, spoken=True),
        )
        self.timer.start_defer_timer(leave.task_instance_id, delay)

    def _handle_robot_running(self, event: Event) -> None:
        # time.sleep(config.RECOVERY_STOP_DELAY_SECONDS)
        task = self._require_active_in(
            event,
            {RobotTaskState.R_ACCEPTED, RobotTaskState.R_REDO, RobotTaskState.R_EXECUTING},
        )
        if task is None:
            return

        if task.robot_running_received:
            #send default speed from config.py
            # self.ros.publish_speed(config.DEFAULT_SPEED)
            # print({"Initialize the speed to:": config.DEFAULT_SPEED})
            self.logger.log_message(
                "Ignored repeated robot running status.",
                {"task_instance_id": task.task_instance_id},
            )
            return

        task.robot_running_received = True
        if task.state != RobotTaskState.R_EXECUTING:
            self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot physical status running.")
        self._show_execution_dialogue(task)

    def _handle_robot_homed(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_RETURNING_HOME)
        if task is None:
            return

        self._finish_canceled_task(
            task,
            event,
            "Robot homed; cancellation recovery completed.",
        )

    def _handle_robot_success(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED},
        )
        if task is None:
            return

        if task.robot_success_received:
            self.logger.log_message(
                "Ignored repeated robot success.",
                {"task_instance_id": task.task_instance_id},
            )
            return
        task.robot_success_received = True

        next_state = self.state_machine.get_next_state(task.state, event.event_type, task.task_id)
        if next_state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._transition(
                task,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                event,
                "Robot success received; waiting for free-drive permission.",
            )
            self.logger.log_message(
                "Asked human for permission to enable free-drive mode.",
                {"task_instance_id": task.task_instance_id},
            )
            self.cli.show_message(
                self.message_manager.ask_permission_for_free_drive(),
                speech=self.message_manager.ask_permission_for_free_drive(spoken=True),
            )
            return
        if next_state == RobotTaskState.R_FREE_DRIVE:
            self._enable_free_drive(task, event, "Panel in position; free-drive enabled without asking.")
            self.cli.show_message(
                self.message_manager.get_free_drive_on_arrival_message(),
                speech=self.message_manager.get_free_drive_on_arrival_message(spoken=True),
            )
            return
        if next_state == RobotTaskState.R_WAITING_HANDOVER:
            self._transition(task, RobotTaskState.R_WAITING_HANDOVER, event,
                             "Robot brought the item; asking to hand it over.")
            self.cli.show_permission_request(
                self.message_manager.get_handover_question(task),
                speech=self.message_manager.get_handover_question(task, spoken=True),
            )
            return

        self._transition(task, RobotTaskState.R_DONE, event, "Robot success received.")
        self.active_task = None
        if task.task_id == config.TASK_LEAVE:
            self._held_piece_id = None
            self.cli.show_message(self.message_manager.get_left_panel_message(),
                                  speech=self.message_manager.get_left_panel_message(spoken=True))
            self._propose_followup(task, config.TASK_BRING_CONNECTOR)
            return
        if task.task_id == config.TASK_LEAVE_HANDOVER:
            brought, self._handed_over = self._handed_over, None
            self.cli.show_message(self.message_manager.get_left_handover_message(),
                                  speech=self.message_manager.get_left_handover_message(spoken=True))
            if not self._start_advance(task) and brought is not None:
                self._continue_chain(brought)
            return
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )
        if not self._start_advance(task):
            self._continue_chain(task)

    def _show_execution_dialogue(self, task: RobotTask) -> None:
        self.cli.show_message(
            self.message_manager.get_execution_message(task),
            speech=self.message_manager.get_execution_message(task, spoken=True),
        )

    def _transition(
        self,
        task: RobotTask,
        new_state: RobotTaskState,
        event: Event,
        message: str | None = None,
    ) -> None:
        """Apply every state change through one logging path."""
        old_state = task.state
        expected = self.state_machine.get_next_state(old_state, event.event_type, task.task_id)
        if expected != new_state:
            raise ValueError(
                f"Invalid transition for task {task.task_id}: "
                f"{old_state.name} + {event.event_type.name} -> {new_state.name}"
            )
        task.state = new_state
        task.updated_at = time.time()
        self.logger.log_transition(task, event, old_state, new_state, message)
        self._sync_tracker(task, event)
        if (self.advance_task is not None and task.task_instance_id == self._advance_after
                and new_state in (S.R_RECOVERY_EVALUATING, S.R_CANCELED)):
            self._drop_advance("Task asked in advance dropped: the task it follows was stopped.")
        if self.demo is not None:
            self.demo.on_transition(self, task, event)

    def _build_task_instance_id(self, round_id: int, task_id: int, piece_id: int) -> str:
        base = f"round_{round_id}_task_{task_id}_piece_{piece_id}"
        attempt = self._task_instance_counts.get(base, 0) + 1
        self._task_instance_counts[base] = attempt
        return base if attempt == 1 else f"{base}_attempt_{attempt}"

    def _require_active(self, event: Event, state: RobotTaskState) -> RobotTask | None:
        return self._require_active_in(event, {state})

    def _require_active_in(self, event: Event, states: set[RobotTaskState]) -> RobotTask | None:
        task = self.active_task
        if task is None:
            self._log_invalid(event, "No active task.")
            return None
        if event.task_instance_id is not None and event.task_instance_id != task.task_instance_id:
            self._log_invalid(event, "Event task id does not match active task.")
            return None
        if task.state not in states:
            self._log_invalid(event, "Invalid event for current task state.")
            return None
        return task

    def _log_invalid(self, event: Event, message: str) -> None:
        state = self.active_task.state if self.active_task is not None else None
        self.logger.log_message(
            message,
            {
                "event_type": event.event_type.name,
                "state": state.name if state is not None else None,
            },
        )
        self.cli.show_message(
            self.message_manager.get_invalid_event_message(state, event.event_type),
            speech=self.message_manager.get_invalid_event_message(state, event.event_type, spoken=True),
        )
