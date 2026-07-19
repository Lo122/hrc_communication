"""Recognition trigger manager."""

from config import TRIGGER_RULES
from events import Event, EventType
from models import RecognitionResult


class TriggerManager:
    """Converts recognition progress into one-time trigger events."""

    def __init__(self):
        self.previous_progress = {}
        self.triggered_keys = set()

    def update(self, recognition_result: RecognitionResult) -> list[Event]:
        """Evaluate trigger rules against the latest recognition result."""
        events = []

        for step_id, rule in TRIGGER_RULES.items():
            event = self._check_rule(recognition_result, step_id, rule)
            if event is not None:
                events.append(event)

        self._save_previous_progress(recognition_result)
        return events

    def _check_rule(
        self,
        recognition_result: RecognitionResult,
        step_id: int,
        rule: dict,
    ) -> Event | None:
        """Return RECOGNITION_TRIGGER when a configured threshold is crossed."""
        trigger_key = (recognition_result.round_id, step_id, recognition_result.piece_id)
        if trigger_key in self.triggered_keys:
            return None

        if recognition_result.step_id != step_id:
            return None
        if recognition_result.confidence < rule["min_confidence"]:
            return None

        progress_key = (
            recognition_result.round_id,
            recognition_result.step_id,
            recognition_result.piece_id,
        )
        previous = self.previous_progress.get(progress_key, 0.0)
        current = recognition_result.progress
        threshold = rule["progress_threshold"]

        if previous <= threshold and current > threshold:
            self.triggered_keys.add(trigger_key)
            return Event(
                event_type=EventType.RECOGNITION_TRIGGER,
                source="recognition",
                payload={
                    "step_id": step_id,
                    "piece_id": recognition_result.piece_id,
                    "round_id": recognition_result.round_id,
                    "progress": recognition_result.progress,
                },
            )

        return None

    def _save_previous_progress(self, recognition_result: RecognitionResult) -> None:
        """Remember progress so triggers fire on crossing, not every frame."""
        progress_key = (
            recognition_result.round_id,
            recognition_result.step_id,
            recognition_result.piece_id,
        )
        self.previous_progress[progress_key] = recognition_result.progress
