"""Grasshopper task dispatcher skeleton."""

from models import RobotTask


class GHDispatcher:
    """Formats and sends task_id + piece_id through an injected UDP sender."""

    def __init__(self, udp_sender):
        self.udp_sender = udp_sender

    def dispatch_task(self, task: RobotTask) -> str:
        """Start a predefined robot task in Grasshopper."""
        message = self.build_message(task.task_id, task.piece_id)
        self.udp_sender.send(message)
        return message

    def build_message(self, task_id: int, piece_id: int) -> str:
        """Create the outbound Grasshopper message format."""
        # TODO: Replace with the already confirmed GH UDP message format if different.
        return f"{task_id},{piece_id}"
