"""Live UR10e TCP pose + flange force/torque reader, with optional payload
(end-effector weight/inertia) compensation to recover the external
(environment-contact) wrench.

Requires the RTDE python client:
    pip install ur_rtde

Robot side: RTDE must be enabled (it is, by default, on e-Series controllers)
and nothing else may be exclusively holding the RTDE control interface if you
also want to call setPayload() (see below).

------------------------------------------------------------------------
Where the numbers come from
------------------------------------------------------------------------
UR e-Series arms (UR3e/UR5e/UR10e/UR16e) have a 6-axis force/torque sensor
built into the wrist, between the last joint and the tool flange. RTDE
exposes it two ways:

  - `actual_TCP_force`  (Fx,Fy,Fz,Tx,Ty,Tz in the base frame, expressed at
    the TCP): this is the sensor reading AFTER the controller has already
    subtracted the weight of whatever payload you told it about (Installation
    -> Payload: mass, center of gravity, and, on recent PolyScope, the
    inertia tensor). If that payload entry is accurate, this signal already
    has the end-effector's static weight removed for you.

  - `ft_raw_wrench` (Fx,Fy,Fz,Tx,Ty,Tz): the raw, uncompensated sensor
    reading in the sensor/flange frame -- no gravity or payload subtraction
    at all. Use this if you want to do the compensation yourself (e.g. the
    CAD payload numbers are unreliable, you want to compensate a payload you
    haven't told the controller about, or you also want to remove inertial
    effects during motion, which `actual_TCP_force` does NOT remove).

------------------------------------------------------------------------
Excluding end-effector weight AND inertia ("true" environment force)
------------------------------------------------------------------------
The sensor measures the total wrench needed to hold/accelerate the
end-effector, which is the sum of three physical contributions:

    W_sensor = W_gravity + W_inertial + W_external

So the external (environment/contact) wrench is:

    W_external = W_sensor - W_gravity - W_inertial

1) Gravity term (static weight of the end-effector), in the sensor frame:
       F_gravity = R_sensor_base^T @ (0, 0, -m*g)
       T_gravity = r_com x F_gravity
   where R_sensor_base is the sensor's orientation in the base/world frame
   (from the TCP pose), m is the end-effector mass, and r_com is the vector
   from the sensor origin to the end-effector's center of mass (both from
   the Installation payload settings, or measured/CAD'd yourself).

2) Inertial term (only matters if the end-effector accelerates -- i.e. the
   arm is moving, not just holding a static pose):
       F_inertial = m * a_com
       T_inertial = I @ alpha + omega x (I @ omega)
   where a_com is the linear acceleration of the end-effector's CoM,
   omega/alpha are its angular velocity/acceleration, and I is its inertia
   tensor about the CoM (also settable in the e-Series payload dialog). All
   vectors expressed in the sensor frame.

This is exactly what the UR controller itself computes when you set the
Installation payload correctly -- so in practice, the simplest and most
robust way to "exclude end-effector weight and inertia" is:

    1. Measure/estimate the end-effector's mass, center of gravity, and (if
       it's not a small/symmetric tool) inertia tensor.
    2. Enter them once under Installation -> Payload on the teach pendant,
       or push them at runtime with RTDEControlInterface.setPayload(...).
    3. Read `actual_TCP_force` from RTDE -- gravity AND (per UR's dynamic
       model) inertial contributions from that payload are already removed,
       leaving essentially the external/contact wrench.

The manual GravityCompensator below exists for the case where you can't (or
don't want to) rely on the controller's payload setting -- e.g. you're
testing several tools without reprogramming the installation each time, or
you want the raw sensor stream (`ft_raw_wrench`) compensated in your own
process. It implements gravity compensation exactly, and inertial
compensation using finite-differenced TCP velocity from RTDE (good enough
for slow/moderate motions; for fast motions prefer letting the controller do
it via setPayload, which uses the true joint-level dynamics model).

Usage: uv run python 4_execution/read_ur_live_data.py --ip 127.0.0.1 --publish-ros

"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
import numpy as np

try:
    from rtde_receive import RTDEReceiveInterface
    from rtde_control import RTDEControlInterface
except ImportError as exc:  # pragma: no cover - depends on deployment environment
    raise SystemExit(
        "ur_rtde is required: pip install ur_rtde"
    ) from exc


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from src.robot_utils import GravityCompensator, Payload, rotvec_to_quaternion

try:
    import roslibpy
except ImportError:  # pragma: no cover - depends on deployment environment
    roslibpy = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROBOT_IP = "127.0.0.1"  # URSim in Docker: use localhost, since 30001-30004 are
# published to the host (see `docker ps`); a real UR10e uses its LAN IP instead.

# --- latency management ---------------------------------------------------
# Three independent rates, on purpose:
#   - RTDE_HZ paces the actual robot read loop, via rtde_r.initPeriod()/
#     waitPeriod() -- this uses ur_rtde's own clock rather than time.sleep(),
#     which drifts under Python/OS scheduling jitter (worse on Windows,
#     whose sleep resolution is ~1-15ms).
#   - ROS_PUBLISH_HZ is deliberately decoupled from RTDE_HZ and lower: the
#     rosbridge websocket write in LiveDataPublisher can stall (slow
#     network, rosbridge backpressure) and must never throttle the RTDE
#     read loop, so we only publish every Nth sample (decimation), not
#     every sample.
#   - PRINT_HZ is decimated the same way, so console I/O doesn't do it
#     either -- printing every RTDE sample at 125Hz would itself become the
#     bottleneck.
RTDE_HZ = 125.0  # RTDE read/loop rate (Hz); matches e-Series' 125-500Hz control loop
ROS_PUBLISH_HZ = 20.0  # rosbridge publish rate (Hz); decimated from RTDE_HZ
PRINT_HZ = 10.0  # console print rate (Hz); decimated from RTDE_HZ


class LiveDataPublisher:
    """Publishes UR10e joint positions, TCP pose, and TCP force/torque to
    rosbridge, on their own topics (see config.ROS_TOPICS "ur_*" entries) --
    independent of ROSCommunication's "robot_position" topic, which is fed
    by the ur_robot_driver ROS node rather than this script."""

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(
        self,
        *,
        host: str = config.ROS_BRIDGE_HOST,
        port: int = config.ROS_BRIDGE_PORT,
    ):
        if roslibpy is None:
            print("[ros] roslibpy is not installed; live data will only print to console.")
            self.client = None
            return

        self.client = roslibpy.Ros(host=host, port=int(port))
        self.client.run()
        self._joint_topic = roslibpy.Topic(
            self.client, config.ROS_TOPICS["ur_joint_position"], "sensor_msgs/JointState"
        )
        self._tcp_position_topic = roslibpy.Topic(
            self.client, config.ROS_TOPICS["ur_tcp_position"], "geometry_msgs/PoseStamped"
        )
        self._tcp_force_topic = roslibpy.Topic(
            self.client, config.ROS_TOPICS["ur_tcp_force"], "geometry_msgs/WrenchStamped"
        )
        for topic in (self._joint_topic, self._tcp_position_topic, self._tcp_force_topic):
            topic.advertise()
        print(f"[ros] connected to rosbridge at {host}:{port}")

    def close(self) -> None:
        if self.client is None:
            return
        for topic in (self._joint_topic, self._tcp_position_topic, self._tcp_force_topic):
            try:
                topic.unadvertise()
            except Exception:
                pass
        self.client.terminate()
        self.client = None

    def _is_ready(self) -> bool:
        return self.client is not None and self.client.is_connected

    def publish_joint_positions(self, q: np.ndarray, timestamp: float) -> None:
        if not self._is_ready():
            return
        self._joint_topic.publish(
            roslibpy.Message(
                {
                    "header": {"stamp": _to_ros_time(timestamp), "frame_id": ""},
                    "name": self.JOINT_NAMES,
                    "position": [float(v) for v in q],
                    "velocity": [],
                    "effort": [],
                }
            )
        )

    def publish_tcp_position(self, tcp_pose: np.ndarray, timestamp: float) -> None:
        if not self._is_ready():
            return
        qx, qy, qz, qw = rotvec_to_quaternion(*tcp_pose[3:])
        self._tcp_position_topic.publish(
            roslibpy.Message(
                {
                    "header": {"stamp": _to_ros_time(timestamp), "frame_id": "base"},
                    "pose": {
                        "position": {
                            "x": float(tcp_pose[0]),
                            "y": float(tcp_pose[1]),
                            "z": float(tcp_pose[2]),
                        },
                        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                    },
                }
            )
        )

    def publish_tcp_force(self, tcp_force: np.ndarray, timestamp: float) -> None:
        if not self._is_ready():
            return
        self._tcp_force_topic.publish(
            roslibpy.Message(
                {
                    "header": {"stamp": _to_ros_time(timestamp), "frame_id": "tool0"},
                    "wrench": {
                        "force": {
                            "x": float(tcp_force[0]),
                            "y": float(tcp_force[1]),
                            "z": float(tcp_force[2]),
                        },
                        "torque": {
                            "x": float(tcp_force[3]),
                            "y": float(tcp_force[4]),
                            "z": float(tcp_force[5]),
                        },
                    },
                }
            )
        )


def _to_ros_time(timestamp: float) -> dict:
    """Convert a float Unix timestamp to a ROS1 time dict ({secs, nsecs})."""
    secs = int(timestamp)
    nsecs = int(round((timestamp - secs) * 1e9))
    return {"secs": secs, "nsecs": nsecs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=ROBOT_IP, help="UR10e controller IP")
    parser.add_argument(
        "--rtde-hz",
        type=float,
        default=RTDE_HZ,
        help="RTDE read/loop rate (Hz), paced with initPeriod/waitPeriod",
    )
    parser.add_argument(
        "--ros-hz",
        type=float,
        default=ROS_PUBLISH_HZ,
        help="rosbridge publish rate (Hz); decimated from --rtde-hz, never "
        "higher than it",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=PRINT_HZ,
        help="console print rate (Hz); decimated from --rtde-hz, never "
        "higher than it",
    )
    parser.add_argument(
        "--set-payload",
        action="store_true",
        help="push --mass/--cog/--payload to the controller via "
        "RTDEControlInterface.setPayload() so actual_TCP_force already "
        "excludes the end-effector weight",
    )
    parser.add_argument("--mass", type=float, default=0.0, help="end-effector mass, kg")
    parser.add_argument(
        "--cog",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="end-effector center of gravity in the flange frame, m",
    )
    parser.add_argument(
        "--manual-compensation",
        action="store_true",
        help="read ft_raw_wrench and subtract gravity/inertia in this "
        "process instead of relying on the controller's payload setting",
    )
    parser.add_argument(
        "--publish-ros",
        action="store_true",
        help="publish joint positions, TCP pose, and TCP force to rosbridge "
        "on /UR10e/position/live, /UR10e/TCPPosition/live, and "
        "/UR10e/TCPForce/live (see config.ROS_TOPICS)",
    )
    args = parser.parse_args()

    if args.ros_hz > args.rtde_hz or args.print_hz > args.rtde_hz:
        raise SystemExit("--ros-hz and --print-hz cannot exceed --rtde-hz")

    rtde_r = RTDEReceiveInterface(args.ip, frequency=args.rtde_hz)
    print(f"[rtde] connected to receive interface at {args.ip} ({args.rtde_hz:.0f} Hz)")

    if args.set_payload:
        rtde_c = RTDEControlInterface(args.ip)
        ok = rtde_c.setPayload(args.mass, list(args.cog))
        print(f"[rtde] setPayload(mass={args.mass}, cog={args.cog}) -> {ok}")
        rtde_c.disconnect()

    compensator = None
    if args.manual_compensation:
        compensator = GravityCompensator(
            Payload(mass=args.mass, center_of_gravity=tuple(args.cog))
        )

    publisher = LiveDataPublisher() if args.publish_ros else None

    # Decimation: publish/print every Nth RTDE sample rather than every
    # sample, so neither the rosbridge websocket write nor console I/O can
    # ever slow down the RTDE read loop's pacing (see the latency-management
    # note above RTDE_HZ/ROS_PUBLISH_HZ/PRINT_HZ).
    publish_every_n = max(1, round(args.rtde_hz / args.ros_hz))
    print_every_n = max(1, round(args.rtde_hz / args.print_hz))

    sample_count = 0
    try:
        while True:
            t_start = rtde_r.initPeriod()

            joint_positions = np.array(rtde_r.getActualQ())  # 6 joint angles, rad
            tcp_pose = np.array(rtde_r.getActualTCPPose())  # [x,y,z,rx,ry,rz]
            tcp_speed = np.array(rtde_r.getActualTCPSpeed())  # [vx,vy,vz,wx,wy,wz]
            tcp_force = np.array(
                rtde_r.getActualTCPForce()
            )  # controller-compensated wrench at TCP

            if sample_count % print_every_n == 0:
                print(
                    f"q=[{', '.join(f'{v:+.3f}' for v in joint_positions)}] rad  "
                    f"pos=({tcp_pose[0]:+.4f},{tcp_pose[1]:+.4f},{tcp_pose[2]:+.4f}) m  "
                    f"rot=({tcp_pose[3]:+.3f},{tcp_pose[4]:+.3f},{tcp_pose[5]:+.3f}) rad  "
                    f"F=({tcp_force[0]:+.2f},{tcp_force[1]:+.2f},{tcp_force[2]:+.2f}) N  "
                    f"T=({tcp_force[3]:+.2f},{tcp_force[4]:+.2f},{tcp_force[5]:+.2f}) Nm"
                )

            if publisher is not None and sample_count % publish_every_n == 0:
                now = time.time()
                publisher.publish_joint_positions(joint_positions, now)
                publisher.publish_tcp_position(tcp_pose, now)
                publisher.publish_tcp_force(tcp_force, now)

            if compensator is not None:
                raw_wrench = np.array(rtde_r.getFtRawWrench())
                ext_wrench = compensator.compensate(
                    raw_wrench, tcp_pose, tcp_speed, time.time()
                )
                if sample_count % print_every_n == 0:
                    print(
                        f"  external (payload-removed): "
                        f"F=({ext_wrench[0]:+.2f},{ext_wrench[1]:+.2f},{ext_wrench[2]:+.2f}) N  "
                        f"T=({ext_wrench[3]:+.2f},{ext_wrench[4]:+.2f},{ext_wrench[5]:+.2f}) Nm"
                    )

            sample_count += 1
            rtde_r.waitPeriod(t_start)
    except KeyboardInterrupt:
        pass
    finally:
        rtde_r.disconnect()
        print("[rtde] disconnected")
        if publisher is not None:
            publisher.close()


if __name__ == "__main__":
    main()
