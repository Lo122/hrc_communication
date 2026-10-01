"""Shared data models for the HRC communication system."""

from dataclasses import dataclass

from events import RobotTaskState


@dataclass
class RecognitionResult:
    """Standardized output of the recognition layer.

    step_probabilities: every step's (smoothed) score over config.STEP_NAMES, from
    which the decision layer picks the step itself (sequence_step_selector.py);
    step_progress: every step's own progress, for a model with a lane per step.
    None when the source has no such output (typed updates, a mismatched model)."""

    round_id: int
    step_id: int
    progress: float
    piece_id: int
    confidence: float
    timestamp: float
    step_probabilities: list[float] | None = None
    step_progress: list[float] | None = None


@dataclass
class RobotTask:
    """One concrete occurrence of a robot-assistance task."""

    task_instance_id: str
    step_id: int
    piece_id: int
    round_id: int
    task_id: int
    state: RobotTaskState
    speed: float
    progress: float = 0.0
    pending_reason: str | None = None
    # The delay of the current delayed start (R_DEFER), for displays.
    defer_seconds: float | None = None
    free_drive_active: bool = False
    robot_running_received: bool = False
    robot_success_received: bool = False
    created_at: float | None = None
    updated_at: float | None = None
