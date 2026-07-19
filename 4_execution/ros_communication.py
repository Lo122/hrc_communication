"""ROS/rosbridge communication."""

from __future__ import annotations

import config
from events import Event, EventType

try:
    import roslibpy
except ImportError:  # pragma: no cover - depends on deployment environment
    roslibpy = None


class ROSCommunication:
    """Publishes runtime robot commands and converts robot feedback into events."""

    def __init__(
        self,
        *,
        host: str = config.ROS_BRIDGE_HOST,
        port: int = config.ROS_BRIDGE_PORT,
        auto_connect: bool = True,
    ):
        self.event_callback = None
        self.host = host
        self.port = int(port)
        self.client = None
        self._publishers = {}
        self._subscriptions = []

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """Connect to rosbridge and initialize topic publishers/subscribers."""
        if roslibpy is None:
            print("[ros] roslibpy is not installed; ROS commands will use console fallback.")
            return False

        if self.client is not None and self.client.is_connected:
            return True

        try:
            self.client = roslibpy.Ros(host=self.host, port=self.port)
            self.client.run()
            self._init_publishers()
            self._init_subscribers()
            print(f"[ros] connected to rosbridge at {self.host}:{self.port}")
            return True
        except Exception as exc:  # pragma: no cover - requires live rosbridge
            self.client = None
            print(f"[ros] could not connect to rosbridge at {self.host}:{self.port}: {exc}")
            return False

    def close(self) -> None:
        """Clean up ROS topics and close the rosbridge connection."""
        for topic in self._subscriptions:
            try:
                topic.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()

        for topic in self._publishers.values():
            try:
                topic.unadvertise()
            except Exception:
                pass
        self._publishers.clear()

        if self.client is not None:
            try:
                self.client.terminate()
            except Exception:
                pass
            self.client = None

    def set_event_callback(self, callback) -> None:
        """Connect ROS feedback to the shared event queue."""
        self.event_callback = callback

    def publish_pause(self) -> None:
        self._publish_control("pause")

    def publish_resume(self) -> None:
        self._publish_control("resume")

    def publish_restart(self) -> None:
        print("[ros] restart not implemented yet")

    def publish_cancel(self) -> None:
        self._publish_control("stop")

    def publish_speed(self, speed: float) -> None:
        self._publish("speed", {"data": float(speed)})

    def publish_human_done(self) -> None:
        self._publish("human_done", {"data": "success"})

    def _init_publishers(self) -> None:
        self._publishers = {
            "control": self._advertise(config.ROS_TOPICS["pause"], "std_msgs/String"),
            "speed": self._advertise(config.ROS_TOPICS["speed"], "std_msgs/Float64"),
            "human_done": self._advertise(config.ROS_TOPICS["human_done"], "std_msgs/String"),
        }

    def _init_subscribers(self) -> None:
        topic = roslibpy.Topic(
            self.client,
            config.ROS_TOPICS["robot_success"],
            "std_msgs/String",
        )
        topic.subscribe(self._on_robot_status_message)
        self._subscriptions.append(topic)

    def _advertise(self, topic_name: str, message_type: str):
        topic = roslibpy.Topic(self.client, topic_name, message_type)
        topic.advertise()
        return topic

    def _publish_control(self, command: str) -> None:
        self._publish("control", {"data": command})

    def _publish(self, publisher_name: str, payload: dict) -> None:
        if not self._is_ready():
            print(f"[ros] {publisher_name} {payload['data']}")
            return

        topic = self._publishers[publisher_name]
        topic.publish(roslibpy.Message(payload))

    def _is_ready(self) -> bool:
        return self.client is not None and self.client.is_connected and bool(self._publishers)

    def _on_robot_status_message(self, message: dict) -> None:
        """Create ROBOT_SUCCESS event when /Robot/status/physical reports success."""
        status = message.get("data") if isinstance(message, dict) else message
        if status == "success":
            self._on_robot_success_message(message)

    def _on_robot_success_message(self, message) -> None:
        """Create ROBOT_SUCCESS event from ROS callback output."""
        event = Event(
            event_type=EventType.ROBOT_SUCCESS,
            source="ros",
            payload={"message": message},
        )

        if self.event_callback is not None:
            self.event_callback(event)
