"""Grasshopper task dispatcher."""

import config
from models import RobotTask


class GHDispatcher:
    """Formats and sends step execution messages through an injected UDP sender."""

    def __init__(self, udp_sender):
        self.udp_sender = udp_sender

    def dispatch_task(self, task: RobotTask) -> dict:
        """Start a predefined robot step in Grasshopper."""
        message = self.build_message(task)
        self.udp_sender.send(message)
        return message

    def build_message(self, task: RobotTask) -> dict:
        """Create the outbound Grasshopper JSON message."""
        step_message = config.GH_STEP_MESSAGES.get(task.step_id, {})
        return {
            "step_id": int(task.step_id),
            "progress": float(task.progress),
            "piece_id": int(task.piece_id),
            "round_id": int(task.round_id),
            "suggested_action": step_message.get("suggested_action", "wait"),
        }
