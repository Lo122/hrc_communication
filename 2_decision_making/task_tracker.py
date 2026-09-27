"""Per-piece task status tracking: not done / pending / working / done.

Every task of every piece in the task database has one TaskStatus. Tasks change
status only through this class:

- recognition reports what the human is doing (on_task_recognized) -> WORKING;
- the robot's task state is mirrored (set_robot_status) -> PENDING / WORKING / DONE;
- the human confirms a task (confirm_done) -> DONE, also closing every earlier
  recognized task of that piece, since a later step implies the earlier ones;
- otherwise a not-done recognized task is PENDING while P(task | reference task)
  from the transition table reaches pending_min_probability.

The reference task is the human task most recently recognized or confirmed; before
any it is the table's START row. Support tasks (Bring Tool etc.) are not in the
table: they turn PENDING through the robot trigger policy or a robot offer.

The human works on one task at a time. Recognition moves them to another task only
if the table expects it after the reference task (it is pending, or likely enough);
whatever they worked on before then goes back to pending, or not done if the table
does not expect it any more. A recognized task the table does not expect is ignored
-- until the tracker has stayed on the reference task longer than that task usually
takes (duration_limits, from the annotated durations), so a wrong reference cannot
stall tracking for good.

Pieces are worked in database order. Recognized human tasks belong to the "human
piece" -- the first open piece whose recognized tasks are not all done -- so a
repeated step (the annotations show Pull Cables twice per piece) stays on the
current piece instead of starting the next one.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# src/ holds this layer's helpers (task database, transition table, trigger policy).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from events import TaskStatus
from task_database import HUMAN, ROBOT


@dataclass
class TrackedTask:
    piece_id: int
    piece_location: str
    task_name: str
    status: TaskStatus = TaskStatus.NOT_DONE
    executor: str | None = None
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    inferred: bool = False
    robot_offered: bool = False
    # Reported by detectors (task_transition_detector.py), e.g. {"screw count": 0.5};
    # trigger rule conditions test them.
    signals: dict = field(default_factory=dict)


class TaskTracker:
    def __init__(self, database, transition_model, recognized_tasks, logger=None,
                 pending_min_probability=0.05, duration_limits=None, clock=time.time):
        self.database = database
        self.transitions = transition_model
        self.recognized_tasks = tuple(recognized_tasks)
        self.logger = logger
        self.pending_min_probability = pending_min_probability
        # {task name: seconds} the tracker may stay on that reference task before
        # recognition is believed against the table (reference_overran). A task
        # without an entry never overruns.
        self.duration_limits = dict(duration_limits or {})
        self.clock = clock

        self.piece_ids = [piece.piece_id for piece in database.pieces]
        self._task_order = {piece.piece_id: piece.task_list for piece in database.pieces}
        self._tasks = {
            piece.piece_id: {name: TrackedTask(piece.piece_id, piece.location, name)
                             for name in piece.task_list}
            for piece in database.pieces
        }
        self._closed_pieces: set[int] = set()
        self.reference_task: str | None = None
        self.reference_piece_id: int | None = None
        self.reference_progress = 0.0
        # When the reference task last changed, or its status did.
        self.reference_since: float | None = None
        # The recognized task last ignored since then (logged once).
        self.ignored_task: str | None = None
        self._refresh_pending()

    # -- queries --------------------------------------------------------------

    def get(self, task_name: str, piece_id: int | None = None) -> TrackedTask | None:
        piece_id = self.human_piece_id if piece_id is None else piece_id
        return self._tasks.get(piece_id, {}).get(task_name)

    def is_recognized(self, task_name: str) -> bool:
        return task_name in self.recognized_tasks

    @property
    def open_piece_ids(self) -> list[int]:
        return [piece_id for piece_id in self.piece_ids
                if piece_id not in self._closed_pieces and not self.is_piece_complete(piece_id)]

    @property
    def current_piece_id(self) -> int | None:
        """First piece with any task left (support tasks included)."""
        open_ids = self.open_piece_ids
        return open_ids[0] if open_ids else None

    @property
    def human_piece_id(self) -> int | None:
        """First open piece whose recognized (human-step) tasks are not all done."""
        for piece_id in self.open_piece_ids:
            if any(task.status != TaskStatus.DONE for name, task in self._tasks[piece_id].items()
                   if self.is_recognized(name)):
                return piece_id
        return self.current_piece_id

    def next_piece_id(self, piece_id: int | None) -> int | None:
        if piece_id is None or piece_id not in self.piece_ids:
            return None
        index = self.piece_ids.index(piece_id) + 1
        return self.piece_ids[index] if index < len(self.piece_ids) else None

    def is_piece_complete(self, piece_id: int) -> bool:
        return all(task.status == TaskStatus.DONE for task in self._tasks[piece_id].values())

    def working_task(self) -> TrackedTask | None:
        """The recognized task the human is on now, if it is still WORKING."""
        task = self._reference()
        return task if task is not None and task.status == TaskStatus.WORKING else None

    def reference_seconds(self) -> float | None:
        """How long the tracker has been on the reference task as it is now."""
        return None if self.reference_since is None else self.clock() - self.reference_since

    def reference_limit(self) -> float | None:
        return self.duration_limits.get(self.reference_task) if self.reference_task is not None else None

    def reference_overran(self) -> bool:
        """The tracker has stayed on the reference task -- working on it, or done with
        it and not moved on -- longer than that task usually takes. Recognition is then
        believed even where the transition table does not expect its task."""
        seconds, limit = self.reference_seconds(), self.reference_limit()
        return seconds is not None and limit is not None and seconds > limit

    def pending(self) -> list[tuple[TrackedTask, float | None]]:
        """PENDING tasks of the open pieces, most likely first (support tasks: None)."""
        items = []
        for piece_id in self.open_piece_ids:
            for name, task in self._tasks[piece_id].items():
                if task.status == TaskStatus.PENDING:
                    probability = (self.transitions.probability(self.reference_task, name)
                                   if self.is_recognized(name) else None)
                    items.append((task, probability))
        return sorted(items, key=lambda item: -1.0 if item[1] is None else -item[1])

    def snapshot(self) -> dict:
        return {
            "current_piece_id": self.current_piece_id,
            "human_piece_id": self.human_piece_id,
            "reference_task": self.reference_task,
            "reference_progress": self.reference_progress,
            "pieces": {
                piece_id: {name: {"status": task.status.name, "executor": task.executor,
                                  "inferred": task.inferred, "signals": dict(task.signals)}
                           for name, task in tasks.items()}
                for piece_id, tasks in self._tasks.items()
            },
        }

    def status_line(self) -> str:
        piece_id = self.current_piece_id
        if piece_id is None:
            return "[tasks] all pieces done"
        working = [task for pid in self.open_piece_ids for task in self._tasks[pid].values()
                   if task.status == TaskStatus.WORKING]
        working_text = ", ".join(self._label(task, piece_id) + (
            f" ({task.progress:.2f})" if task.executor == HUMAN and task.task_name == self.reference_task else "")
            for task in working) or "-"
        pending_text = ", ".join(
            self._label(task, piece_id) + ("" if probability is None else f" {probability:.2f}")
            for task, probability in self.pending()) or "-"
        done = sum(task.status == TaskStatus.DONE for task in self._tasks[piece_id].values())
        return (f"[tasks] piece {piece_id} ({done}/{len(self._tasks[piece_id])} done) | "
                f"working: {working_text} | pending: {pending_text}")

    @staticmethod
    def _label(task: TrackedTask, current_piece_id: int) -> str:
        return task.task_name if task.piece_id == current_piece_id else f"{task.task_name} [piece {task.piece_id}]"

    # -- updates --------------------------------------------------------------

    def on_task_recognized(self, task_name: str, progress: float = 0.0) -> TrackedTask | None:
        """Recognition says the human is doing task_name, at progress (0-1). A task
        other than the reference counts only if _believes() it; it then becomes the
        human's one working task. Returns the task, or None if ignored."""
        piece_id = self.human_piece_id
        task = self.get(task_name, piece_id)
        if task is None:
            self._log("Recognized task is not in the current piece.", task_name=task_name, piece_id=piece_id)
            return None

        if (task_name, piece_id) != (self.reference_task, self.reference_piece_id):
            if not self._believes(task):
                return None
            if task.status == TaskStatus.DONE:
                self._log("Recognized a task already done; treating it as a repeat.",
                          task_name=task_name, piece_id=piece_id)
        self._pull_back_human_work(keep=task)
        if task.status in (TaskStatus.NOT_DONE, TaskStatus.PENDING):
            self._set_status(task, TaskStatus.WORKING, HUMAN)

        if task.status == TaskStatus.WORKING:
            task.progress = progress
        self._set_reference(task_name, piece_id, 1.0 if task.status == TaskStatus.DONE else progress)
        self._refresh_pending()
        return task

    def start_task(self, task_name: str, piece_id: int, executor: str = HUMAN) -> TrackedTask | None:
        """The workflow, not recognition, says this task has started now -- e.g. Screw
        once the robot holds the panel. It becomes the reference task at progress 0,
        and whatever recognized task comes before it is done (inferred): on this piece,
        and on earlier pieces too when a human starts it, since the human has moved on."""
        task = self.get(task_name, piece_id)
        if task is None or task.status == TaskStatus.DONE:
            return task
        if executor == HUMAN:
            self._close_earlier_pieces(piece_id)
            self._pull_back_human_work(keep=task)
        self._close_earlier_tasks(task)
        if task.status != TaskStatus.WORKING or task.executor != executor:
            self._set_status(task, TaskStatus.WORKING, executor)
        task.progress = 0.0
        if self.is_recognized(task_name):
            self._set_reference(task_name, piece_id, 0.0)
        self._refresh_pending()
        return task

    def infer_done_before(self, task_name: str, piece_id: int) -> None:
        """Every recognized task before task_name on this piece is done (inferred) --
        e.g. the cables are pulled once the robot starts lifting the panel."""
        task = self.get(task_name, piece_id)
        if task is not None:
            self._close_earlier_tasks(task)
            self._refresh_pending()

    def confirm_done(self, task_name: str | None = None, executor: str = HUMAN,
                     piece_id: int | None = None, inferred: bool = False) -> TrackedTask | None:
        """Mark a task DONE. task_name None means the task the human is working on.
        inferred: nobody saw it done, the workflow implies it (e.g. no adjustment wanted)."""
        if task_name is None:
            working = self.working_task()
            if working is None:
                self._log("Done confirmation without a working task.")
                return None
            task_name, piece_id = working.task_name, working.piece_id
        if piece_id is None:
            piece_id = self.first_open_piece_for(task_name)
        task = self.get(task_name, piece_id)
        if task is None:
            self._log("No open task to confirm.", task_name=task_name)
            return None
        if task.status == TaskStatus.DONE:
            return task
        if not self.database.can_execute(task_name, executor):
            self._log("Rejected completion: executor may not do this task.",
                      task_name=task_name, piece_id=piece_id, executor=executor)
            return None

        task.inferred = inferred
        self._set_status(task, TaskStatus.DONE, executor)
        if self.is_recognized(task_name):
            self._close_earlier_tasks(task)
            reference = self._reference()
            if reference is None or reference.status != TaskStatus.WORKING or reference is task:
                self._set_reference(task_name, piece_id, 1.0)
        self._refresh_pending()
        return task

    def set_robot_status(self, task_name: str, piece_id: int, status: TaskStatus) -> TrackedTask | None:
        """Mirror a robot task's state onto its tracked task."""
        task = self.get(task_name, piece_id)
        if task is None or task.status == TaskStatus.DONE:
            return task
        if status == TaskStatus.DONE:
            return self.confirm_done(task_name, ROBOT, piece_id)
        if not self.database.can_execute(task_name, ROBOT):
            self._log("Robot task ignored: robot may not do this task.", task_name=task_name)
            return task
        if status == TaskStatus.WORKING:
            self._set_status(task, TaskStatus.WORKING, ROBOT)
        elif status == TaskStatus.PENDING:
            task.robot_offered = True
            # A human already doing it keeps it; a robot that stopped hands it back.
            if not (task.status == TaskStatus.WORKING and task.executor == HUMAN):
                self._set_status(task, TaskStatus.PENDING, None)
        self._refresh_pending()
        return task

    def set_signal(self, task_name: str, piece_id: int, name: str, value) -> TrackedTask | None:
        """A detector reported a signal on this task (not "Done signal": that goes
        through confirm_done)."""
        task = self.get(task_name, piece_id)
        if task is None:
            self._log("Signal for a task not in the database.", task_name=task_name,
                      piece_id=piece_id, signal=name)
            return None
        if task.signals.get(name) != value:
            task.signals[name] = value
            self._log("Task signal.", piece_id=piece_id, task_name=task_name, signal=name, value=value)
        return task

    def set_progress(self, task_name: str, piece_id: int, progress: float) -> TrackedTask | None:
        """How far a working task is, from something other than recognition (a detector)."""
        task = self.get(task_name, piece_id)
        if task is None or task.status != TaskStatus.WORKING:
            return task
        task.progress = progress
        if (task_name, piece_id) == (self.reference_task, self.reference_piece_id):
            self.reference_progress = progress
        return task

    def mark_pending(self, task_name: str, piece_id: int) -> None:
        """A robot trigger condition was met for this task."""
        task = self.get(task_name, piece_id)
        if task is not None and task.status == TaskStatus.NOT_DONE:
            self._set_status(task, TaskStatus.PENDING, None)

    def advance_piece(self) -> int | None:
        """Manual override: close the current piece as-is and move to the next."""
        piece_id = self.current_piece_id
        if piece_id is None:
            return None
        self._closed_pieces.add(piece_id)
        self._log("Piece closed manually.", piece_id=piece_id)
        if self.reference_piece_id == piece_id:
            self._set_reference(None, None, 0.0)
        self._refresh_pending()
        return self.current_piece_id

    # -- internals ------------------------------------------------------------

    def _reference(self) -> TrackedTask | None:
        if self.reference_task is None:
            return None
        return self.get(self.reference_task, self.reference_piece_id)

    def _set_reference(self, task_name: str | None, piece_id: int | None, progress: float) -> None:
        if (task_name, piece_id) != (self.reference_task, self.reference_piece_id):
            self.reference_since = None if task_name is None else self.clock()
            self.ignored_task = None
        self.reference_task, self.reference_piece_id = task_name, piece_id
        self.reference_progress = progress

    def _believes(self, task: TrackedTask) -> bool:
        """Whether recognition may move the human from the reference task to this one:
        it is pending, or the table expects it after the reference (a repeated step is
        done but still expected) -- or the reference has overrun its usual duration."""
        probability = self.transitions.probability(self.reference_task, task.task_name)
        if task.status == TaskStatus.PENDING or probability >= self.pending_min_probability:
            return True
        seconds, limit = self.reference_seconds(), self.reference_limit()
        context = dict(task_name=task.task_name, piece_id=task.piece_id,
                       reference_task=self.reference_task, probability=probability,
                       seconds_on_reference=None if seconds is None else round(seconds, 1),
                       limit_s=limit)
        if self.reference_overran():
            self._log("Reference task overran its usual duration; recognized task believed "
                      "although the transition table does not expect it.", **context)
            return True
        if self.ignored_task != task.task_name:
            self.ignored_task = task.task_name
            self._log("Recognized task ignored: the transition table does not expect it "
                      "after the reference task.", **context)
        return False

    def _pull_back_human_work(self, keep: TrackedTask) -> None:
        """The human works on one task at a time: any other task they are working on
        goes back to pending (_refresh_pending then keeps it pending or not done)."""
        for piece_id in self.open_piece_ids:
            for task in self._tasks[piece_id].values():
                if task is not keep and task.status == TaskStatus.WORKING and task.executor == HUMAN:
                    self._set_status(task, TaskStatus.PENDING, None)

    def first_open_piece_for(self, task_name: str) -> int | None:
        for piece_id in self.open_piece_ids:
            task = self._tasks[piece_id].get(task_name)
            if task is not None and task.status != TaskStatus.DONE:
                return piece_id
        return None

    def _close_earlier_pieces(self, piece_id: int) -> None:
        """Recognized tasks still open on pieces before piece_id are done (inferred).
        Support tasks stay open: the robot may still have a tool to return there."""
        for earlier_id in self.piece_ids[:self.piece_ids.index(piece_id)] if piece_id in self.piece_ids else []:
            for name, earlier in self._tasks[earlier_id].items():
                if self.is_recognized(name) and earlier.status != TaskStatus.DONE:
                    earlier.inferred = True
                    self._set_status(earlier, TaskStatus.DONE, earlier.executor or HUMAN)

    def _close_earlier_tasks(self, task: TrackedTask) -> None:
        order = self._task_order[task.piece_id]
        for name in order[:order.index(task.task_name)]:
            earlier = self._tasks[task.piece_id][name]
            if self.is_recognized(name) and earlier.status != TaskStatus.DONE:
                earlier.inferred = True
                self._set_status(earlier, TaskStatus.DONE, earlier.executor or HUMAN)

    def _set_status(self, task: TrackedTask, status: TaskStatus, executor: str | None) -> None:
        old = task.status
        now = self.clock()
        task.status = status
        task.executor = executor
        if status == TaskStatus.WORKING and task.started_at is None:
            task.started_at = now
        if status == TaskStatus.DONE:
            task.finished_at = now
            task.progress = 1.0
        if old != status:
            if (task.task_name, task.piece_id) == (self.reference_task, self.reference_piece_id):
                self.reference_since = now
            self._log("Task status changed.", piece_id=task.piece_id, task_name=task.task_name,
                      old_status=old.name, new_status=status.name, executor=executor,
                      inferred=task.inferred)

    def _refresh_pending(self) -> None:
        piece_id = self.human_piece_id
        if piece_id is None:
            return
        tasks = self._tasks[piece_id]
        for name in self._task_order[piece_id]:
            task = tasks[name]
            if not self.is_recognized(name) or task.robot_offered:
                continue
            if task.status in (TaskStatus.NOT_DONE, TaskStatus.PENDING):
                likely = self.transitions.probability(self.reference_task, name) >= self.pending_min_probability
                new_status = TaskStatus.PENDING if likely else TaskStatus.NOT_DONE
                if new_status != task.status:
                    self._set_status(task, new_status, None)

        active = [task for name, task in tasks.items() if self.is_recognized(name)
                  and task.status in (TaskStatus.PENDING, TaskStatus.WORKING)]
        if not active:
            # Nothing the table expects is left: fall back to database order.
            for name in self._task_order[piece_id]:
                if self.is_recognized(name) and tasks[name].status == TaskStatus.NOT_DONE:
                    self._set_status(tasks[name], TaskStatus.PENDING, None)
                    break

    def _log(self, message: str, **context) -> None:
        if self.logger is not None:
            self.logger.log_message(message, context)


def build_task_tracking(root, logger=None):
    """TaskTracker + RobotTriggerPolicy from the paths and thresholds in config.py."""
    from pathlib import Path

    import config
    from robot_trigger_policy import RobotTriggerPolicy
    from task_database import TaskDatabase
    from task_sequence_model import TransitionModel, load_duration_stat

    database = TaskDatabase.from_json(Path(root) / config.TASK_DATABASE_PATH)
    transitions = TransitionModel.from_csv(Path(root) / config.TASK_TRANSITION_TABLE_PATH)
    duration_limits = load_duration_stat(Path(root) / config.TASK_DURATION_STATS_PATH,
                                         config.TASK_OVERRUN_STAT)
    tracker = TaskTracker(database, transitions, config.STEP_NAMES, logger=logger,
                          pending_min_probability=config.PENDING_MIN_PROBABILITY,
                          duration_limits=duration_limits)
    policy = RobotTriggerPolicy(database, tracker, config.TRACKED_TO_ROBOT_TASK, logger=logger)
    return tracker, policy
