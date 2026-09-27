"""Publish the UR10e's state from RTDE to rosbridge, for other ROS consumers.

Every loop (READ_RATE_HZ) reads the joints and the TCP over RTDE and publishes
    config.ROS_TOPICS["joint_state"]     sensor_msgs/JointState
    config.ROS_TOPICS["robot_position"]  trajectory_msgs/JointTrajectoryPoint
    config.ROS_TOPICS["tcp_pose"]        geometry_msgs/PoseStamped
    config.ROS_TOPICS["ft_wrench"]       geometry_msgs/WrenchStamped (actual_TCP_force)
and listens on config.ROS_TOPICS["ft_zero"] (std_msgs/Empty) to tare the FT sensor,
which needs --enable-ft-zero (see ur_reader_node).

read_ur_live_data.py (next to this file) is the other RTDE reader: it publishes on
its own "ur_*" topics, logs to file and can compensate the payload. Both stay; see
the note in config.ROS_TOPICS.

Usage:
    uv run python 4_execution/eval/ur_state_reader.py [--ip IP] [--plot] [--enable-ft-zero]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EXECUTION = Path(__file__).resolve().parents[1]
ROOT = EXECUTION.parent
# The repo root last, so it ends up first: multi-actor-interface ships its own
# top-level `config` package (see pyproject.toml).
for _path in (EXECUTION / "src", ROOT):
    sys.path.insert(0, str(_path))

import roslibpy
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import config
import ros_messages

READ_RATE_HZ = 100.0  # paced by rtde_r.initPeriod()/waitPeriod(), not time.sleep()
JOINT_NAMES = [f"joint_{i + 1}" for i in range(6)]

PLOT_HISTORY_SECONDS = 10.0
PLOT_UPDATE_EVERY_N_LOOPS = 10  # ~10 Hz plot refresh at a 100 Hz read loop


class StatePublisher:
    """The four state topics this reader advertises on rosbridge."""

    def __init__(self, client: roslibpy.Ros):
        self._joint_state = _advertise(client, "joint_state", "sensor_msgs/JointState")
        self._position = _advertise(
            client, "robot_position", "trajectory_msgs/JointTrajectoryPoint"
        )
        self._tcp_pose = _advertise(client, "tcp_pose", "geometry_msgs/PoseStamped")
        self._wrench = _advertise(client, "ft_wrench", "geometry_msgs/WrenchStamped")

    def publish(self, *, q, qd, qdd, joint_torques, tcp_pose, wrench, timestamp: float) -> None:
        """One RTDE sample; every message carries the same stamp."""
        self._joint_state.publish(
            roslibpy.Message(
                ros_messages.joint_state(
                    timestamp, q, names=JOINT_NAMES, velocities=qd, efforts=joint_torques
                )
            )
        )
        self._position.publish(
            roslibpy.Message(ros_messages.joint_trajectory_point(q, qd, qdd))
        )
        self._tcp_pose.publish(
            roslibpy.Message(ros_messages.pose_stamped(timestamp, tcp_pose, ""))
        )
        self._wrench.publish(
            roslibpy.Message(ros_messages.wrench_stamped(timestamp, wrench, ""))
        )

    def close(self) -> None:
        for topic in (self._joint_state, self._position, self._tcp_pose, self._wrench):
            try:
                topic.unadvertise()
            except Exception:
                pass


def _advertise(client: roslibpy.Ros, topic_key: str, message_type: str) -> roslibpy.Topic:
    topic = roslibpy.Topic(client, config.ROS_TOPICS[topic_key], message_type)
    topic.advertise()
    return topic


def zero_ftsensor(rtde_c: RTDEControlInterface) -> None:
    """Tare the FT sensor. The robot must be stationary when this is called."""
    print("Zeroing FT sensor...")
    if rtde_c.zeroFtSensor():
        print("FT sensor zeroed.")
    else:
        print("Failed to zero FT sensor.")


def ur_reader_node(
    ip: str = config.ROBOT_IP,
    *,
    enable_plot: bool = False,
    enable_ft_zero: bool = False,
) -> None:
    print(f"Connecting to UR at {ip}...")
    rtde_r = RTDEReceiveInterface(ip, frequency=READ_RATE_HZ)

    # The control interface is opt-in because a controller has exactly one RTDE control
    # session. Constructing RTDEControlInterface uploads rtde_control.script, which stops
    # whatever control script is already running and claims the same RTDE input registers.
    # If anything else owns that session -- the Multi-Actor-Interface-Library UR bridge, a
    # second copy of this reader, read_ur_live_data.py --set-payload -- the loser's socket
    # is closed by the controller and its receive thread prints
    #   RTDEReceiveInterface boost system Exception: (asio.misc:2) End of file
    # Reading state needs no control session at all, so the default here is read-only.
    rtde_c = None
    if enable_ft_zero:
        print(
            "[warn] --enable-ft-zero opens an RTDE *control* session. Nothing else "
            "(the MAIL UR bridge, another reader) may hold one at the same time."
        )
        rtde_c = RTDEControlInterface(ip)

    print(f"Connecting to rosbridge at {config.ROS_BRIDGE_HOST}:{config.ROS_BRIDGE_PORT}...")
    ros_client = roslibpy.Ros(host=config.ROS_BRIDGE_HOST, port=config.ROS_BRIDGE_PORT)
    ros_client.run()
    print("Connected to rosbridge.")

    def _handle_ft_zero(_msg) -> None:
        if rtde_c is None:
            print(
                f"[skip] {config.ROS_TOPICS['ft_zero']} needs a control session; "
                "restart this reader with --enable-ft-zero to tare from here."
            )
            return
        zero_ftsensor(rtde_c)

    if rtde_c is not None:
        zero_ftsensor(rtde_c)
    ft_zero_topic = roslibpy.Topic(ros_client, config.ROS_TOPICS["ft_zero"], "std_msgs/Empty")
    ft_zero_topic.subscribe(_handle_ft_zero)

    publisher = StatePublisher(ros_client)

    plotter = None
    if enable_plot:
        # Imported only when asked for, to keep matplotlib off the default path.
        from plot_utils import LiveWrenchPlot

        plotter = LiveWrenchPlot(history_s=PLOT_HISTORY_SECONDS)

    sample_count = 0
    try:
        while ros_client.is_connected:
            t_start = rtde_r.initPeriod()

            # RTDE has no *actual* joint accelerations or torques; the controller's
            # targets are the closest it offers.
            wrench = rtde_r.getActualTCPForce()  # [Fx, Fy, Fz, Tx, Ty, Tz], N / Nm
            publisher.publish(
                q=rtde_r.getActualQ(),  # rad
                qd=rtde_r.getActualQd(),  # rad/s
                qdd=rtde_r.getTargetQdd(),  # rad/s^2, target
                joint_torques=rtde_r.getTargetMoment(),  # Nm, target
                tcp_pose=rtde_r.getActualTCPPose(),  # [x, y, z, rx, ry, rz], m / rad
                wrench=wrench,
                timestamp=time.time(),
            )

            if plotter is not None and sample_count % PLOT_UPDATE_EVERY_N_LOOPS == 0:
                plotter.update(wrench)

            sample_count += 1
            rtde_r.waitPeriod(t_start)
    finally:
        publisher.close()
        ft_zero_topic.unsubscribe()
        ros_client.terminate()
        if plotter is not None:
            plotter.close()
        if rtde_c is not None:
            rtde_c.disconnect()
        rtde_r.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR RTDE state reader / ROS publisher.")
    parser.add_argument("--ip", default=config.ROBOT_IP, help="UR controller IP")
    parser.add_argument("--plot", action="store_true", help="Show a live plot of the TCP force/torque wrench.")
    parser.add_argument(
        "--enable-ft-zero",
        action="store_true",
        help="open an RTDE control session so this reader can tare the FT sensor. Off by "
        "default: a controller has only one control session, and taking it here drops "
        "whoever else holds it (see ur_reader_node).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        ur_reader_node(args.ip, enable_plot=args.plot, enable_ft_zero=args.enable_ft_zero)
    except KeyboardInterrupt:
        pass
