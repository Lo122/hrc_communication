"""Shared in-process event queue."""

from queue import Queue

from events import Event


class EventQueue:
    """Thin wrapper around Queue used by all event producers."""

    def __init__(self):
        self._queue: Queue[Event] = Queue()

    def put(self, event: Event) -> None:
        self._queue.put(event)

    def get(self) -> Event:
        return self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()
