"""Human CLI command parser."""

from events import Event, EventType


class CommandParser:
    """Converts raw text into standardized events."""

    _ALIASES = {
        "yes": EventType.H_ACCEPT,
        "accept": EventType.H_ACCEPT,
        "okay": EventType.H_ACCEPT,
        "ok": EventType.H_ACCEPT,
        "no": EventType.H_REFUSE,
        "refuse": EventType.H_REFUSE,
        "later": EventType.H_DEFER,
        "defer": EventType.H_DEFER,
        "pause": EventType.H_PAUSE,
        "continue": EventType.H_RESUME,
        "resume": EventType.H_RESUME,
        "restart": EventType.H_RESTART,
        "redo": EventType.H_RESTART,
        "cancel": EventType.H_CANCEL,
        "faster": EventType.H_SPEEDUP,
        "speed up": EventType.H_SPEEDUP,
        "slower": EventType.H_SLOWDOWN,
        "slow down": EventType.H_SLOWDOWN,
        "free drive": EventType.H_FREE_GO,
        "free go": EventType.H_FREE_GO,
        "home": EventType.H_RETURN_HOME,
        "return home": EventType.H_RETURN_HOME,
        "manual recovery": EventType.H_MANUAL_RECOVERY,
        "done": EventType.H_DONE,
        "finished": EventType.H_DONE,
    }

    def parse(self, raw_text: str) -> Event | None:
        text = raw_text.strip().lower()
        if not text:
            return None

        if text.startswith("execute "):
            task_instance_id = text.split(maxsplit=1)[1]
            return Event(
                event_type=EventType.H_EXECUTE_PENDING_TASK,
                source="human_cli",
                task_instance_id=task_instance_id,
            )

        event_type = self._ALIASES.get(text)
        if event_type is None:
            return None

        return Event(event_type=event_type, source="human_cli")
