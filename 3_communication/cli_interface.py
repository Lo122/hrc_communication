"""Command-line human interface skeleton."""

from events import Event, EventType


class CLIInterface:
    """Displays prompts and reads human text commands."""

    def __init__(self, command_parser=None):
        self.command_parser = command_parser

    def show_message(self, message: str) -> None:
        """Display a system message to the human."""
        print(message)

    def show_permission_request(self, message: str) -> None:
        """Display a permission request for a proposed robot task."""
        print(message)

    def read_raw(self) -> str:
        """Read one line without interpreting it.

        Queries ("what are you doing", "why", "repeat") are answered locally
        and never become events, so the runtime needs the text before the
        parser turns it into one -- or fails to.
        """
        return input("> ")

    def read_input(self) -> Event | None:
        """Read one CLI command and convert it into an event."""
        return self.parse_command(self.read_raw())

    def parse_command(self, raw_text: str) -> Event | None:
        """Convert a raw command string into a standardized Event."""
        if self.command_parser is None:
            return None
        return self.command_parser.parse(raw_text)

    def _build_event(
        self,
        event_type: EventType,
        task_instance_id: str | None = None,
    ) -> Event:
        """Create a human CLI event."""
        return Event(
            event_type=event_type,
            source="human_cli",
            task_instance_id=task_instance_id,
        )
