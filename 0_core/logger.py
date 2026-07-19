"""Event and transition logging skeleton."""

import json

from events import Event
from models import RobotTask


class EventLogger:
    """Records incoming events, transitions, and integration actions."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = open(file_path, "a", encoding="utf-8", buffering=1)

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")

    def log_event(self, event: Event) -> None:
        """Record every incoming event before it is handled."""
        self._write(
            {
                "type": "event",
                "event_type": event.event_type.name,
                "source": event.source,
                "task_instance_id": event.task_instance_id,
                "payload": event.payload,
            }
        )

    def log_transition(
        self,
        task: RobotTask,
        event: Event,
        old_state,
        new_state,
        message: str | None = None,
    ) -> None:
        """Record a state transition performed by TaskManager."""
        self._write(
            {
                "type": "transition",
                "task_instance_id": task.task_instance_id,
                "old_state": old_state.name,
                "new_state": new_state.name,
                "event_type": event.event_type.name,
                "message": message,
            }
        )

    def log_message(self, message: str, context: dict | None = None) -> None:
        """Record non-transition integration messages such as ignored events."""
        self._write({"type": "log", "message": message, "context": context or {}})
