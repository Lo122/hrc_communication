"""ROS/rosbridge communication."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque

import config
from events import Event, EventType

try:
    import roslibpy
except ImportError:  # pragma: no cover - depends on deployment environment
    roslibpy = None

_logger = logging.getLogger(__name__)

# Wrench samples kept until the main loop drains them (drain_wrench): ~40 s at 125 Hz,
# far more than one loop pass, so only a stalled loop ever loses any.
WRENCH_BUFFER = 5000


def _to_ros_time(timestamp: float) -> dict:
    """Convert a float Unix timestamp to a ROS1 time dict ({secs, nsecs}),
    as used in std_msgs/Header.stamp. A fraction that rounds up to a full
    second carries into secs, so nsecs stays < 1e9 (same as
    4_execution/src/ros_messages.py's ros_time, which the runtime doesn't
    import to keep src/ off its sys.path)."""
    secs = math.floor(timestamp)
    nsecs = round((timestamp - secs) * 1e9)
    if nsecs == 10**9:
        secs, nsecs = secs + 1, 0
    return {"secs": secs, "nsecs": nsecs}


class ROSCommunication:
    """Publishes runtime robot commands and converts robot feedback into events."""

    def __init__(
        self,
        *,
        host: str = config.ROS_BRIDGE_HOST,
        port: int = config.ROS_BRIDGE_PORT,
        auto_connect: bool = True,
        wrench_topic: str | None = None,
    ):
        """wrench_topic: a geometry_msgs/WrenchStamped topic to buffer for the task
        detectors (drain_wrench); None subscribes to none."""
        self.event_callback = None
        self.host = host
        self.port = int(port)
        self.client = None
        self._publishers = {}
        self._subscriptions = []
        self.latest_joint_positions = None
        self.latest_gripper_open = None
        self.wrench_topic = wrench_topic
        self._wrench: deque[tuple[float, list[float]]] = deque(maxlen=WRENCH_BUFFER)
        # The newest wrench sample, (t, [fx, fy, fz, tx, ty, tz]), kept for displays --
        # reading it does not take it from the task detectors (drain_wrench).
        self.latest_wrench: tuple[float, list[float]] | None = None

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

    # -- commands ------------------------------------------------------------

    def publish_pause(self) -> None:
        """Temporarily stop motion without overwriting the task's saved speed."""
        self.publish_global_speed(config.MIN_SPEED)  # ROS speed topics are Float64, not zero, to avoid divide-by-zero errors

    def publish_resume(self, speed: float) -> None:
        """Restore the task speed saved by TaskManager."""
        self.publish_global_speed(speed)

    def publish_restart(self) -> None:
        print("[ros] restart not implemented yet")

    def publish_cancel(self) -> None:
        self._publish_control("stop")

    def publish_real_pause(self) -> None:
        self._publish_control("pause")

    def publish_return_home(self) -> None:
        self._publish_control("home")

    def publish_gripper_open(self) -> None:
        """Release the brought item into the human's hands."""
        self._publish("gripper", {"data": False})

    def publish_speed(self, speed: float) -> None:
        self.publish_global_speed(speed)

    def publish_global_speed(self, speed: float) -> None:
        self._publish("global_speed", {"data": float(speed)})

    def publish_local_speed(self, speed: float) -> None:
        self._publish("local_speed", {"data": float(speed)})

    def publish_human_done(self) -> None:
        self._publish("human_done", {"data": "success"})

    def publish_free_drive(self, enabled: bool) -> None:
        self._publish("free_drive", {"data": bool(enabled)})

    def publish_human_location(
        self,
        xyz: tuple[float, float, float],
        timestamp: float | None = None,
        keypoints: dict[str, dict[str, float]] | None = None,
        velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Publish the human's world-frame position (see
        1_recognition/src/skeleton3d_pipeline.py's world_root_xyz -- same
        world frame as /UR10/position/live, defined by the calibrated
        extrinsics) for downstream consumers like path planning, plus
        their pelvis-relative posture as an H36M-17 keypoint dict (see
        recognition_manager.py's get_last_keypoints() -- pelvis is
        included and always exactly (0,0,0), everything else relative to
        it) for consumers that need body shape, not just root position.
        keypoints is None when no valid 3D lift has been seen yet.
        velocity is the world-frame m/s from recognition's position Kalman
        filter (dt from real timestamps); null when the filter is off.

        Published as std_msgs/String, JSON-encoded: A single JSON string keeps position+keypoints as one
        message on one topic without needing a custom .msg package."""
        body = {
            "header": {
                "seq": 0,
                "stamp": _to_ros_time(timestamp if timestamp is not None else time.time()),
                "frame_id": "world",
            },
            "point": {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
            "keypoints": keypoints if keypoints is not None else {},
            "velocity": (None if velocity is None else
                         {"x": float(velocity[0]), "y": float(velocity[1]), "z": float(velocity[2])}),
        }
        self._publish("human_position", {"data": json.dumps(body)})

    # -- robot feedback ------------------------------------------------------

    def get_latest_joint_positions(self) -> list[float] | None:
        """Return a copy of the latest joint positions received from ROS."""
        if self.latest_joint_positions is None:
            return None
        return list(self.latest_joint_positions)

    def get_latest_gripper_has_object(self) -> bool | None:
        """Return whether the gripper is holding an object, or None if unknown."""
        if self.latest_gripper_open is None:
            return None
        return not self.latest_gripper_open

    def drain_wrench(self) -> list[tuple[float, list[float]]]:
        """The wrench samples received since the last call, oldest first, as
        (receive time, [fx, fy, fz, tx, ty, tz])."""
        samples = []
        while self._wrench:
            samples.append(self._wrench.popleft())
        return samples

    # -- topics --------------------------------------------------------------

    def _init_publishers(self) -> None:
        self._publishers = {
            "control": self._advertise(config.ROS_TOPICS["control"], "std_msgs/String"),
            "global_speed": self._advertise(config.ROS_TOPICS["global_speed"], "std_msgs/Float64"),
            "local_speed": self._advertise(config.ROS_TOPICS["local_speed"], "std_msgs/Float64"),
            "human_done": self._advertise(config.ROS_TOPICS["human_done"], "std_msgs/String"),
            "free_drive": self._advertise(config.ROS_TOPICS["free_drive"], "std_msgs/Bool"),
            "gripper": self._advertise(config.ROS_TOPICS["gripper"], "std_msgs/Bool"),
            "human_position": self._advertise(config.ROS_TOPICS["human_position"], "std_msgs/String"),
        }

    def _init_subscribers(self) -> None:
        for topic_key, message_type, handler in (
            ("r_task_done", "std_msgs/String", self._on_robot_task_done_signal),
            ("robot_success", "std_msgs/String", self._on_robot_status_message),
            ("robot_position", "trajectory_msgs/JointTrajectoryPoint", self._on_robot_position_message),
            ("gripper", "std_msgs/Bool", self._on_gripper_message),
        ):
            self._subscribe(config.ROS_TOPICS[topic_key], message_type, handler)
        if self.wrench_topic:
            self._subscribe(self.wrench_topic, "geometry_msgs/WrenchStamped", self._on_wrench_message)

    def _advertise(self, topic_name: str, message_type: str):
        topic = roslibpy.Topic(self.client, topic_name, message_type)
        topic.advertise()
        return topic

    def _subscribe(self, topic_name: str, message_type: str, handler) -> None:
        topic = roslibpy.Topic(self.client, topic_name, message_type)
        topic.subscribe(handler)
        self._subscriptions.append(topic)

    def _publish_control(self, command: str) -> None:
        self._publish("control", {"data": command})

    def _publish(self, publisher_name: str, payload: dict) -> None:
        if not self._is_ready():
            # Console fallback without rosbridge: log what would have been sent.
            _logger.warning("[ros] %s %s", publisher_name, payload.get("data", payload))
            return

        topic = self._publishers[publisher_name]
        topic.publish(roslibpy.Message(payload))

    def _is_ready(self) -> bool:
        return self.client is not None and self.client.is_connected and bool(self._publishers)

    # -- subscriber callbacks ------------------------------------------------

    def _on_robot_task_done_signal(self, message: dict) -> None:
        """Create task lifecycle events from /Task/signal."""
        status = message.get("data") if isinstance(message, dict) else message
        if status == "start_job":
            self._emit_robot_status_event(EventType.ROBOT_RUNNING, message)
        elif status == "end_job":
            self._emit_robot_status_event(EventType.ROBOT_SUCCESS, message)

    def _on_robot_status_message(self, message: dict) -> None:
        """Create return-home events from /Robot/status/physical."""
        status = message.get("data") if isinstance(message, dict) else message
        if status == "homed":
            self._emit_robot_status_event(EventType.ROBOT_HOMED, message)

    def _on_robot_position_message(self, message: dict) -> None:
        """Cache the latest six robot joint positions."""
        positions = message.get("positions") if isinstance(message, dict) else None
        if isinstance(positions, (list, tuple)):
            self.latest_joint_positions = [float(position) for position in positions]

    def _on_gripper_message(self, message: dict) -> None:
        """Cache whether the gripper is open; False means closed/holding."""
        value = message.get("data") if isinstance(message, dict) else None
        if isinstance(value, bool):
            self.latest_gripper_open = value

    def _on_wrench_message(self, message: dict) -> None:
        """Buffer one TCP wrench sample, stamped on arrival: the task detectors
        compare it with this machine's clock, not the publisher's."""
        try:
            force, torque = message["wrench"]["force"], message["wrench"]["torque"]
            wrench = [float(force[axis]) for axis in "xyz"] + [float(torque[axis]) for axis in "xyz"]
        except (KeyError, TypeError, ValueError):
            return
        self.latest_wrench = (time.time(), wrench)
        self._wrench.append(self.latest_wrench)

    def _emit_robot_status_event(self, event_type: EventType, message) -> None:
        event = Event(
            event_type=event_type,
            source="ros",
            payload={"message": message},
        )

        if self.event_callback is not None:
            self.event_callback(event)
