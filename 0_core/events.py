"""Event and state definitions for the HRC communication system.

This module only defines event vocabulary and shared event containers.
Events do not modify task state directly; state transitions belong to
TaskManager.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import time
from typing import Any


class RobotTaskState(Enum):
    """Persistent robot task states."""

    R_WAITING_RESPONSE = auto()

    R_ACCEPTED = auto()
    R_REFUSED = auto()
    R_DEFER = auto()
    R_PENDING = auto()

    R_EXECUTING = auto()
    R_PAUSED = auto()
    R_REDO = auto()
    R_WAITING_FREE_DRIVE = auto()
    R_FREE_DRIVE = auto()
    R_HOLDING = auto()
    # A brought item (config.HANDOVER_ITEMS): asking to hand it over, then holding it
    # until the human asks for it.
    R_WAITING_HANDOVER = auto()
    R_HOLDING_HANDOVER = auto()

    R_RECOVERY_EVALUATING = auto()
    R_WAITING_HOME_PERMISSION = auto()
    R_RETURNING_HOME = auto()
    R_MANUAL_RECOVERY = auto()

    R_CANCELED = auto()
    R_DONE = auto()


class TaskStatus(Enum):
    """Status of one assembly task (piece x task) in the task database."""

    NOT_DONE = auto()
    PENDING = auto()
    WORKING = auto()
    DONE = auto()


class EventType(Enum):
    """Instantaneous human, system, recognition, and robot feedback events."""

    # Legacy recognition event; handled exactly like HUMAN_TASK_UPDATE.
    RECOGNITION_TRIGGER = auto()
    # What the human is doing now: step_id (model head index) and progress (0-1).
    HUMAN_TASK_UPDATE = auto()
    HUMAN_LOCATION_UPDATE = auto()
    # A sensor-based detector reports a signal on a task (payload task_name, piece_id,
    # signal, value) -- see 2_decision_making/task_transition_detector.py.
    TASK_SIGNAL = auto()

    # Human confirms a task finished (payload task_name; none = current working task).
    H_TASK_DONE = auto()
    # Human asks the robot for a task (payload task_name), bypassing trigger rules.
    H_REQUEST_ROBOT_TASK = auto()
    H_NEXT_PIECE = auto()
    # The demo's scripted opening starts (2_decision_making/demo_opening.py).
    DEMO_START = auto()
    # The operator changed the decision layer's state from the live view (payload "op"
    # plus its fields): 2_decision_making/manual_control.py.
    MANUAL_CONTROL = auto()
    # Recognition has run for config.RECOGNITION_ACTIVATION_S: its task updates count.
    RECOGNITION_ACTIVE = auto()

    H_ACCEPT = auto()
    H_REFUSE = auto()
    H_DEFER = auto()
    H_EXECUTE_PENDING_TASK = auto()

    H_FREE_GO = auto()
    H_RETURN_HOME = auto()
    H_MANUAL_RECOVERY = auto()

    RECOVERY_HOME_AVAILABLE = auto()
    RECOVERY_MANUAL_REQUIRED = auto()

    H_CANCEL = auto()
    H_PAUSE = auto()
    H_RESUME = auto()
    H_RESTART = auto()

    H_SPEEDUP = auto()
    H_SLOWDOWN = auto()

    H_DONE = auto()
    H_SCREW_DONE = auto()
    # Human is ready to take the item the robot brought: the gripper opens.
    H_HANDOVER = auto()

    RESPONSE_TIMEOUT = auto()
    # The workflow starts a task a few seconds from now without asking (leaving the
    # hand-over position): R_DEFER until DEFER_TIMEOUT. A human "later" is H_DEFER, which
    # makes the task pending instead.
    DELAYED_START = auto()
    DEFER_TIMEOUT = auto()
    # A robot offer scheduled for now (payload task_name, piece_id): TaskManager.schedule_offer.
    SCHEDULED_OFFER = auto()

    ROBOT_RUNNING = auto()
    ROBOT_SUCCESS = auto()
    ROBOT_HOMED = auto()


@dataclass
class Event:
    """Common event object passed through the shared event queue."""

    event_type: EventType
    source: str
    task_instance_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

