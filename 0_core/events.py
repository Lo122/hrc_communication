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

    R_RECOVERY_EVALUATING = auto()
    R_WAITING_HOME_PERMISSION = auto()
    R_RETURNING_HOME = auto()
    R_MANUAL_RECOVERY = auto()

    R_CANCELED = auto()
    R_DONE = auto()


class EventType(Enum):
    """Instantaneous human, system, recognition, and robot feedback events."""

    RECOGNITION_TRIGGER = auto()

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

    RESPONSE_TIMEOUT = auto()
    DEFER_TIMEOUT = auto()

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

