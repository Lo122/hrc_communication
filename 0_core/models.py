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
    task_id: int
    piece_id: int
    round_id: int
    state: RobotTaskState
    speed: float
    pending_reason: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
