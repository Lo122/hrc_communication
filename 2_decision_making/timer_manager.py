"""Timeout timer management skeleton."""

from threading import Timer

from events import Event, EventType


class TimerManager:
    """Creates response and defer timeout events."""

    def __init__(self, event_callback):
        self.event_callback = event_callback
        self.response_timer = None
        self.defer_timer = None

    def start_response_timer(self, task_instance_id: str, duration: float) -> None:
        """Start human response timeout for a waiting task."""
        self.cancel_response_timer()
        self.response_timer = Timer(duration, self._emit_response_timeout, args=[task_instance_id])
        self.response_timer.daemon = True
        self.response_timer.start()

    def cancel_response_timer(self) -> None:
        """Cancel the active response timer, if any."""
        if self.response_timer is not None:
            self.response_timer.cancel()
            self.response_timer = None

    def start_defer_timer(self, task_instance_id: str, duration: float) -> None:
        """Start deferred execution timeout for a deferred task."""
        self.cancel_defer_timer()
        self.defer_timer = Timer(duration, self._emit_defer_timeout, args=[task_instance_id])
        self.defer_timer.daemon = True
        self.defer_timer.start()

    def cancel_defer_timer(self) -> None:
        """Cancel the active defer timer, if any."""
        if self.defer_timer is not None:
            self.defer_timer.cancel()
            self.defer_timer = None

    def _emit_response_timeout(self, task_instance_id: str) -> None:
        """Emit a RESPONSE_TIMEOUT event tied to one task instance."""
        event = Event(
            event_type=EventType.RESPONSE_TIMEOUT,
            source="timer",
            task_instance_id=task_instance_id,
        )
        self.event_callback(event)

    def _emit_defer_timeout(self, task_instance_id: str) -> None:
        """Emit a DEFER_TIMEOUT event tied to one task instance."""
        event = Event(
            event_type=EventType.DEFER_TIMEOUT,
            source="timer",
            task_instance_id=task_instance_id,
        )
        self.event_callback(event)
