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

    def dispatch_human_location(self, xyz: tuple[float, float, float], timestamp: float | None = None) -> dict:
        """Send the human's world-frame position (see
        1_recognition/skeleton3d_pipeline.py's world_root_xyz) to Grasshopper."""
        message = self.build_human_location_message(xyz, timestamp)
        self.udp_sender.send(message)
        return message

    def build_human_location_message(self, xyz: tuple[float, float, float], timestamp: float | None = None) -> dict:
        return {
            "human_position": {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
            "timestamp": timestamp,
        }

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
