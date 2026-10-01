"""Recognition -> decision-making task updates.

Recognition no longer decides which robot task to offer (that is the decision
layer's RobotTriggerPolicy, driven by the task database). It only reports what the
human is doing: one HUMAN_TASK_UPDATE when the stable step changes, and again
whenever its progress has moved by at least publish_delta -- or, for a model that
reports every step's score and progress, whenever one of those has moved enough,
since the decision layer picks the step from them against the task sequence.
"""

import config
from events import Event, EventType
from models import RecognitionResult

# Decimals the per-step lists are sent with: plenty for 0-1 thresholds, and a short
# datagram.
STEP_LIST_DECIMALS = 4


def normalize_progress(raw_progress: float, scale: float = config.RECOGNITION_PROGRESS_SCALE) -> float:
    """Progress-head output -> 0-1, the scale the task database thresholds use."""
    return min(max(float(raw_progress) / scale, 0.0), 1.0)


def _moved(values, last, delta: float) -> bool:
    """Whether a per-step list changed by at least delta anywhere (or appeared, went, or
    changed length)."""
    if values is None or last is None:
        return (values is None) != (last is None)
    return len(values) != len(last) or any(abs(a - b) >= delta for a, b in zip(values, last))


class TaskUpdatePublisher:
    """Turns the per-frame RecognitionResult stream into sparse task updates."""

    def __init__(self, publish_delta: float = config.RECOGNITION_PROGRESS_PUBLISH_DELTA,
                 progress_scale: float = config.RECOGNITION_PROGRESS_SCALE,
                 probability_delta: float = config.RECOGNITION_PROBABILITY_PUBLISH_DELTA):
        self.publish_delta = publish_delta
        self.progress_scale = progress_scale
        self.probability_delta = probability_delta
        self._last_step_id = None
        self._last_progress = None
        self._last_probabilities = None
        self._last_step_progress = None

    def update(self, recognition_result: RecognitionResult) -> list[Event]:
        step_id = int(recognition_result.step_id)
        progress = normalize_progress(recognition_result.progress, self.progress_scale)
        probabilities = recognition_result.step_probabilities
        step_progress = (None if recognition_result.step_progress is None else
                         [normalize_progress(value, self.progress_scale)
                          for value in recognition_result.step_progress])
        if (step_id == self._last_step_id
                and abs(progress - self._last_progress) < self.publish_delta
                and not _moved(probabilities, self._last_probabilities, self.probability_delta)
                and not _moved(step_progress, self._last_step_progress, self.publish_delta)):
            return []
        self._last_step_id, self._last_progress = step_id, progress
        self._last_probabilities, self._last_step_progress = probabilities, step_progress
        return [task_update_event(step_id, progress, recognition_result.confidence,
                                  recognition_result.round_id, recognition_result.timestamp,
                                  step_probabilities=probabilities, step_progress=step_progress)]


def task_update_event(step_id: int, progress: float, confidence: float = 1.0,
                      round_id: int = 0, timestamp: float | None = None,
                      source: str = "recognition", step_probabilities=None,
                      step_progress=None) -> Event:
    payload = {
        "step_id": int(step_id),
        "task_name": config.STEP_NAMES[step_id] if 0 <= step_id < len(config.STEP_NAMES) else None,
        "progress": float(progress),
        "confidence": float(confidence),
        "round_id": int(round_id),
    }
    if timestamp is not None:
        payload["timestamp"] = timestamp
    # Over config.STEP_NAMES, plain floats: the payload goes over UDP as JSON.
    for key, values in (("step_probabilities", step_probabilities),
                        ("step_progress", step_progress)):
        if values is not None:
            payload[key] = [round(float(value), STEP_LIST_DECIMALS) for value in values]
    return Event(event_type=EventType.HUMAN_TASK_UPDATE, source=source, payload=payload)
