"""Shared data models for the HRC communication system."""

from dataclasses import dataclass

from events import RobotTaskState


@dataclass
class RecognitionResult:
    """Standardized output of the recognition layer."""

    round_id: int
    step_id: int
    progress: float
    piece_id: int
    confidence: float
    timestamp: float


@dataclass
class RobotTask:
    """One concrete occurrence of a robot-assistance task."""

    task_instance_id: str
    step_id: int
    piece_id: int
    round_id: int
    state: RobotTaskState
    speed: float
    progress: float = 0.0
    #: Key into task_catalog.TASKS -- says what this task IS, independent
    #: of which human step happened to trigger it. Chained tasks have no
    #: meaningful trigger step, so step_id alone can no longer identify them.
    task_key: str = ""
    pending_reason: str | None = None
    #: When the human first said cancel on a load-bearing task. Cleared
    #: once the window passes, so an old request never counts as the
    #: first half of a new confirmation.
    cancel_requested_at: float | None = None
    free_drive_active: bool = False
    robot_running_received: bool = False
    robot_success_received: bool = False
    created_at: float | None = None
    updated_at: float | None = None
