"""Event and transition logging skeleton."""

from events import Event
from models import RobotTask


class EventLogger:
    """Records incoming events, transitions, and integration actions."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        # TODO: Replace print-based logging with structured file logging.

    def log_event(self, event: Event) -> None:
        """Record every incoming event before it is handled."""
        print(
            "[event]",
            event.event_type.name,
            event.source,
            event.task_instance_id,
            event.payload,
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
        print(
            "[transition]",
            task.task_instance_id,
            old_state.name,
            "->",
            new_state.name,
            event.event_type.name,
            message or "",
        )

    def log_message(self, message: str, context: dict | None = None) -> None:
        """Record non-transition integration messages such as ignored events."""
        print("[log]", message, context or {})
