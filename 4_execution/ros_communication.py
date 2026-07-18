"""ROS/rosbridge communication skeleton."""

from events import Event, EventType


class ROSCommunication:
    """Publishes runtime robot commands and converts robot feedback into events."""

    def __init__(self):
        self.event_callback = None
        # TODO: Initialize rosbridge client.
        # TODO: Initialize command publishers.
        # TODO: Subscribe to robot success topic.

    def set_event_callback(self, callback) -> None:
        """Connect ROS feedback to the shared event queue."""
        self.event_callback = callback

    def publish_pause(self) -> None:
        # TODO: Publish pause command.
        print("[ros] pause")

    def publish_resume(self) -> None:
        # TODO: Publish resume command.
        print("[ros] resume")

    def publish_restart(self) -> None:
        # TODO: Publish restart command.
        print("[ros] restart")

    def publish_cancel(self) -> None:
        # TODO: Publish cancel command.
        print("[ros] cancel")

    def publish_speed(self, speed: float) -> None:
        # TODO: Publish speed value.
        print(f"[ros] speed {speed}")

    def publish_human_done(self) -> None:
        # TODO: Publish human-done signal.
        print("[ros] human_done")

    def _on_robot_success_message(self, message) -> None:
        """Create ROBOT_SUCCESS event from ROS callback output."""
        event = Event(
            event_type=EventType.ROBOT_SUCCESS,
            source="ros",
            payload={"message": message},
        )

        if self.event_callback is not None:
            self.event_callback(event)
