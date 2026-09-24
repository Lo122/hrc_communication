"""Recognition -> decision-making task updates.

Recognition no longer decides which robot task to offer (that is the decision
layer's RobotTriggerPolicy, driven by the task database). It only reports what the
human is doing: one HUMAN_TASK_UPDATE when the stable step changes, and again
whenever its progress has moved by at least publish_delta.
"""

import config
from events import Event, EventType
from models import RecognitionResult


def normalize_progress(raw_progress: float, scale: float = config.RECOGNITION_PROGRESS_SCALE) -> float:
    """Progress-head output -> 0-1, the scale the task database thresholds use."""
    return min(max(float(raw_progress) / scale, 0.0), 1.0)


class TaskUpdatePublisher:
    """Turns the per-frame RecognitionResult stream into sparse task updates."""

    def __init__(self, publish_delta: float = config.RECOGNITION_PROGRESS_PUBLISH_DELTA,
                 progress_scale: float = config.RECOGNITION_PROGRESS_SCALE):
        self.publish_delta = publish_delta
        self.progress_scale = progress_scale
        self._last_step_id = None
        self._last_progress = None

    def update(self, recognition_result: RecognitionResult) -> list[Event]:
        step_id = int(recognition_result.step_id)
        progress = normalize_progress(recognition_result.progress, self.progress_scale)
        if (step_id == self._last_step_id
                and abs(progress - self._last_progress) < self.publish_delta):
            return []
        self._last_step_id, self._last_progress = step_id, progress
        return [task_update_event(step_id, progress, recognition_result.confidence,
                                  recognition_result.round_id, recognition_result.timestamp)]


def task_update_event(step_id: int, progress: float, confidence: float = 1.0,
                      round_id: int = 0, timestamp: float | None = None,
                      source: str = "recognition") -> Event:
    payload = {
        "step_id": int(step_id),
        "task_name": config.STEP_NAMES[step_id] if 0 <= step_id < len(config.STEP_NAMES) else None,
        "progress": float(progress),
        "confidence": float(confidence),
        "round_id": int(round_id),
    }
    if timestamp is not None:
        payload["timestamp"] = timestamp
    return Event(event_type=EventType.HUMAN_TASK_UPDATE, source=source, payload=payload)
