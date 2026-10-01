"""The operator's hand on the decision layer: manual changes from the live view.

Recognition and the detectors are not perfect, so in a user study the operator must be
able to set the task tracker right, and steer the robot through the same paths the
human's answers take. The live view (decision_view.py) POSTs a command; HRCSystem
queues it as a MANUAL_CONTROL event, so TaskManager applies it here -- on its own
thread, between the other events. Each command is logged (the view's timeline shows
it) and its outcome kept for the page (TaskManager.manual_results).

Commands, as the payload's "op" plus its fields:
  event         a human answer or robot report (EVENT_TYPES) as if it had come in: it
                goes through TaskManager's usual handlers (event_type[, task_name,
                task_instance_id])
  task_status   a tracked task's status, who does it and how far it is, set exactly
                (TaskTracker.set_manually): task_name, piece_id, status[, executor --
                the human if unsaid, unless only the robot may do it -- progress]
  signal        a detector signal on a tracked task, set or (value None) cleared:
                task_name, piece_id, name, value
  offer         ask about a robot task now, like any offer: task_name, piece_id
  pool_add      put a robot task in the pending pool, to be asked for later
  pool_offer    ask about a pooled robot task again: task_instance_id
  pool_remove   drop a pooled robot task; its task is the human's again
  queue_remove  drop a queued robot offer: index[, task_name, piece_id -- refused if
                the entry there is another one by now]
  clear_active  forget the active robot task -- in the decision layer only: the robot
                is not told anything (Cancel stops it). For a state that got stuck.
  clear_held    forget the panel the robot holds
After a change to the tracker the trigger rules are checked again, as after any
confirmation, so the robot may offer what the new state allows.
"""

from __future__ import annotations

import config
from events import Event, EventType, RobotTaskState as S, TaskStatus
from task_database import HUMAN, ROBOT

# Events the operator may inject: the human's answers and commands, and the robot's
# reports (for a robot that did not report, or a test without one).
EVENT_TYPES = (
    "H_ACCEPT", "H_REFUSE", "H_DEFER", "H_CANCEL", "H_PAUSE", "H_RESUME", "H_RESTART",
    "H_DONE", "H_SCREW_DONE", "H_HANDOVER", "H_FREE_GO", "H_RETURN_HOME",
    "H_MANUAL_RECOVERY", "H_TASK_DONE", "H_NEXT_PIECE",
    "ROBOT_RUNNING", "ROBOT_SUCCESS", "ROBOT_HOMED",
)
EXECUTORS = (HUMAN, ROBOT)
OPS = ("event", "task_status", "signal", "offer", "pool_add", "pool_offer", "pool_remove",
       "queue_remove", "clear_active", "clear_held")
SOURCE = "manual"


class ManualError(ValueError):
    """A command that cannot be applied; its text says why."""


def apply(manager, payload: dict) -> str:
    """Apply one command to TaskManager; returns what was done. Raises ManualError."""
    op = payload.get("op")
    handler = _HANDLERS.get(op)
    if handler is None:
        raise ManualError(f"Unknown command {op!r}.")
    return handler(manager, payload)


# -- events -------------------------------------------------------------------------

def _event(manager, payload: dict) -> str:
    name = payload.get("event_type")
    if name not in EVENT_TYPES:
        raise ManualError(f"Not an event the operator can send: {name!r}.")
    event_payload = {"task_name": payload["task_name"]} if payload.get("task_name") else {}
    manager.handle_event(Event(EventType[name], SOURCE, payload.get("task_instance_id"),
                               event_payload))
    active = manager.active_task
    return f"sent {name}" + (f"; robot now {active.state.name}" if active is not None else "; robot idle")


# -- the task tracker ------------------------------------------------------------------

def _task_status(manager, payload: dict) -> str:
    tracker = _tracker(manager)
    name, piece_id = _task(manager, payload)
    try:
        status = TaskStatus[payload.get("status")]
    except KeyError:
        raise ManualError(f"Unknown status {payload.get('status')!r}.") from None
    executor = payload.get("executor") or None
    if executor is not None and executor not in EXECUTORS:
        raise ManualError(f"Unknown executor {executor!r}.")
    if executor is None and status in (TaskStatus.WORKING, TaskStatus.DONE):
        # Unsaid: the human, unless only the robot may do it.
        executor = HUMAN if tracker.database.can_execute(name, HUMAN) else ROBOT
    progress = payload.get("progress")
    if progress is not None:
        progress = _fraction(progress, "progress")
    try:
        task = tracker.set_manually(name, piece_id, status, executor, progress)
    except ValueError as exc:
        raise ManualError(str(exc)) from None
    manager._withdraw_finished_offers()
    manager._offer_triggered_tasks()
    return (f"{name} (piece {piece_id}) set to {task.status.name}"
            + (f" by {task.executor}" if task.executor else "")
            + (f" at {task.progress:.2f}" if task.status == TaskStatus.WORKING else ""))


def _signal(manager, payload: dict) -> str:
    tracker = _tracker(manager)
    name, piece_id = _task(manager, payload)
    signal = str(payload.get("name") or "").strip()
    if not signal:
        raise ManualError("Name the signal.")
    value = payload.get("value")
    if value is None:
        tracker.clear_signal(name, piece_id, signal)
        text = f"cleared {signal!r} on {name} (piece {piece_id})"
    else:
        if not isinstance(value, (bool, int, float)):
            raise ManualError(f"A signal is true, false or a number, not {value!r}.")
        tracker.set_signal(name, piece_id, signal, value)
        text = f"set {signal!r} = {value} on {name} (piece {piece_id})"
    manager._offer_triggered_tasks()
    return text


# -- robot offers -----------------------------------------------------------------------

def _offer(manager, payload: dict) -> str:
    name, piece_id = _robot_task(manager, payload)
    if manager._releases_panel(name):
        # Leaving has no tracked task: the robot's own request path finds the panel.
        manager.handle_event(Event(EventType.H_REQUEST_ROBOT_TASK, SOURCE, payload={"task_name": name}))
    else:
        manager.offer_robot_task(name, piece_id)
    return f"offered {name} (piece {piece_id})"


def _pool_add(manager, payload: dict) -> str:
    name, piece_id = _robot_task(manager, payload)
    if manager._pooled_task(name, piece_id) is not None:
        raise ManualError(f"{name} (piece {piece_id}) is in the pending pool already.")
    task = manager._build_task(manager._offer_context(piece_id), config.TRACKED_TO_ROBOT_TASK[name])
    task.state, task.pending_reason = S.R_PENDING, "manual"
    manager.pending_pool.add(task)
    if manager.policy is not None:
        manager.policy.mark_offered(name, piece_id)
    manager._sync_tracker(task, Event(EventType.MANUAL_CONTROL, SOURCE, task.task_instance_id))
    return f"added {name} (piece {piece_id}) to the pending pool as {task.task_instance_id}"


def _pool_offer(manager, payload: dict) -> str:
    task = _pooled(manager, payload)
    manager.handle_event(Event(EventType.H_EXECUTE_PENDING_TASK, SOURCE, task.task_instance_id))
    return f"asked about {task.task_instance_id} again"


def _pool_remove(manager, payload: dict) -> str:
    task = _pooled(manager, payload)
    manager.pending_pool.remove(task.task_instance_id)
    name = _tracked_name(task.task_id)
    tracker = manager.tracker
    if tracker is not None and name is not None and tracker.database.can_execute(name, HUMAN):
        tracker.hand_to_human(name, task.piece_id)
    return f"removed {task.task_instance_id} from the pending pool"


def _queue_remove(manager, payload: dict) -> str:
    index = payload.get("index")
    if not isinstance(index, int) or not 0 <= index < len(manager.waiting_triggers):
        raise ManualError(f"No queued offer #{index}.")
    entry = manager.waiting_triggers[index]
    named = payload.get("task_name"), payload.get("piece_id")
    if named != (None, None) and named != (entry["task_name"], entry["piece_id"]):
        raise ManualError("The queue changed meanwhile; look again and retry.")
    del manager.waiting_triggers[index]
    return f"dropped the queued offer {entry['task_name']} (piece {entry['piece_id']})"


# -- stuck state -----------------------------------------------------------------------

def _clear_active(manager, payload: dict) -> str:
    task = manager.active_task
    if task is None:
        raise ManualError("There is no active robot task.")
    manager.timer.cancel_response_timer()
    manager.timer.cancel_defer_timer()
    if manager.advance_task is not None and manager._advance_after == task.task_instance_id:
        manager.advance_task = manager._advance_after = manager._advance_answer = None
    manager.active_task = None
    return (f"forgot the active robot task {task.task_instance_id} ({task.state.name}); "
            "the robot was not told")


def _clear_held(manager, payload: dict) -> str:
    if manager._held_piece_id is None:
        raise ManualError("The robot holds no panel.")
    piece_id, manager._held_piece_id = manager._held_piece_id, None
    return f"forgot the held panel of piece {piece_id}"


# -- helpers -------------------------------------------------------------------------

def _tracked_name(task_id: int) -> str | None:
    """The tracked task a robot task performs."""
    return next((name for name, tid in config.TRACKED_TO_ROBOT_TASK.items() if tid == task_id), None)


def _tracker(manager):
    if manager.tracker is None:
        raise ManualError("No task tracker is loaded.")
    return manager.tracker


def _piece(payload: dict) -> int:
    try:
        return int(payload.get("piece_id"))
    except (TypeError, ValueError):
        raise ManualError(f"Unknown piece {payload.get('piece_id')!r}.") from None


def _task(manager, payload: dict) -> tuple[str, int]:
    name, piece_id = payload.get("task_name"), _piece(payload)
    if manager.tracker.get(name, piece_id) is None:
        raise ManualError(f"{name!r} is not a task of piece {piece_id}.")
    return name, piece_id


def _robot_task(manager, payload: dict) -> tuple[str, int]:
    name = payload.get("task_name")
    if name not in config.TRACKED_TO_ROBOT_TASK:
        raise ManualError(f"The robot has no task {name!r}.")
    if manager.tracker is not None and not manager.tracker.database.can_execute(name, ROBOT):
        raise ManualError(f"The robot may not do {name}.")
    return name, _piece(payload)


def _pooled(manager, payload: dict):
    task = manager.pending_pool.get(payload.get("task_instance_id"))
    if task is None:
        raise ManualError(f"No pooled robot task {payload.get('task_instance_id')!r}.")
    return task


def _fraction(value, what: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ManualError(f"{what} is a number from 0 to 1.") from None
    if not 0.0 <= value <= 1.0:
        raise ManualError(f"{what} is a number from 0 to 1.")
    return value


_HANDLERS = {
    "event": _event, "task_status": _task_status, "signal": _signal, "offer": _offer,
    "pool_add": _pool_add, "pool_offer": _pool_offer, "pool_remove": _pool_remove,
    "queue_remove": _queue_remove, "clear_active": _clear_active, "clear_held": _clear_held,
}
