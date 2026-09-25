import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

# Repo root first: multi-actor-interface installs a top-level `config` package too,
# which would otherwise shadow this repo's config.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import roslibpy
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from config import ROBOT_IP, ROS_BRIDGE_HOST, ROS_BRIDGE_PORT, ROS_TOPICS

READ_RATE_HZ = 100
JOINT_NAMES = [f"joint_{i + 1}" for i in range(6)]

PLOT_HISTORY_SECONDS = 10
PLOT_UPDATE_EVERY_N_LOOPS = 10  # ~10 Hz plot refresh at a 100 Hz read loop


def _rotvec_to_quaternion(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Convert a UR axis-angle rotation vector (rx, ry, rz) to a quaternion (x, y, z, w)."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    half = angle / 2.0
    sin_half_over_angle = math.sin(half) / angle
    return rx * sin_half_over_angle, ry * sin_half_over_angle, rz * sin_half_over_angle, math.cos(half)


def _ros_stamp() -> dict:
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)
    return {"secs": secs, "nsecs": nsecs}


def _ros_header() -> dict:
    return {"stamp": _ros_stamp(), "frame_id": ""}


class ForceTorquePlotter:
    """Rolling live plot of the TCP force/torque wrench."""

    def __init__(self, history_seconds: float = PLOT_HISTORY_SECONDS, rate_hz: float = READ_RATE_HZ):
        import matplotlib.pyplot as plt

        self._plt = plt

        max_len = max(int(history_seconds * rate_hz), 1)
        self.times: deque = deque(maxlen=max_len)
        self.force = [deque(maxlen=max_len) for _ in range(3)]
        self.torque = [deque(maxlen=max_len) for _ in range(3)]
        self._t0 = time.time()

        plt.ion()
        self.fig, (self.ax_force, self.ax_torque) = plt.subplots(2, 1, sharex=True)

        self.force_lines = [self.ax_force.plot([], [], label=label)[0] for label in ("Fx", "Fy", "Fz")]
        self.torque_lines = [self.ax_torque.plot([], [], label=label)[0] for label in ("Tx", "Ty", "Tz")]

        self.ax_force.set_ylabel("Force (N)")
        self.ax_torque.set_ylabel("Torque (Nm)")
        self.ax_torque.set_xlabel("Time (s)")
        self.ax_force.legend(loc="upper right")
        self.ax_torque.legend(loc="upper right")
        self.ax_force.grid(True)
        self.ax_torque.grid(True)
        self.fig.tight_layout()

    def update(self, wrench: list) -> None:
        self.times.append(time.time() - self._t0)
        for i in range(3):
            self.force[i].append(wrench[i])
            self.torque[i].append(wrench[3 + i])

        for i, line in enumerate(self.force_lines):
            line.set_data(self.times, self.force[i])
        for i, line in enumerate(self.torque_lines):
            line.set_data(self.times, self.torque[i])

        for axis in (self.ax_force, self.ax_torque):
            axis.relim()
            axis.autoscale_view()

        self.fig.canvas.draw_idle()
        # Pumps the GUI event loop; also yields briefly, which is why plot
        # updates are throttled to every Nth read-loop iteration.
        self._plt.pause(0.001)


def zero_ftsensor(rtde_c: RTDEControlInterface) -> None:
    """Tare the FT sensor. The robot must be stationary when this is called."""
    print("Zeroing FT sensor...")
    if rtde_c.zeroFtSensor():
        print("FT sensor zeroed.")
    else:
        print("Failed to zero FT sensor.")


def ur_reader_node(enable_plot: bool = False, enable_ft_zero: bool = False) -> None:
    print(f"Connecting to UR at {ROBOT_IP}...")
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

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
        rtde_c = RTDEControlInterface(ROBOT_IP)

    print(f"Connecting to rosbridge at {ROS_BRIDGE_HOST}:{ROS_BRIDGE_PORT}...")
    ros_client = roslibpy.Ros(host=ROS_BRIDGE_HOST, port=ROS_BRIDGE_PORT)
    ros_client.run()
    print("Connected to rosbridge.")

    def _handle_ft_zero(_msg) -> None:
        if rtde_c is None:
            print(
                f"[skip] {ROS_TOPICS['ft_zero']} needs a control session; "
                "restart this reader with --enable-ft-zero to tare from here."
            )
            return
        zero_ftsensor(rtde_c)

    if rtde_c is not None:
        zero_ftsensor(rtde_c)
    ft_zero_topic = roslibpy.Topic(ros_client, ROS_TOPICS["ft_zero"], "std_msgs/Empty")
    ft_zero_topic.subscribe(_handle_ft_zero)

    joint_state_pub = roslibpy.Topic(ros_client, ROS_TOPICS["joint_state"], "sensor_msgs/JointState")
    position_pub = roslibpy.Topic(ros_client, ROS_TOPICS["robot_position"], "trajectory_msgs/JointTrajectoryPoint")
    tcp_pose_pub = roslibpy.Topic(ros_client, ROS_TOPICS["tcp_pose"], "geometry_msgs/PoseStamped")
    wrench_pub = roslibpy.Topic(ros_client, ROS_TOPICS["ft_wrench"], "geometry_msgs/WrenchStamped")
    for publisher in (joint_state_pub, position_pub, tcp_pose_pub, wrench_pub):
        publisher.advertise()

    plotter = ForceTorquePlotter() if enable_plot else None

    loop_period = 1.0 / READ_RATE_HZ
    loop_count = 0

    try:
        while ros_client.is_connected:
            loop_start = time.time()

            # 1. Joint Data (Lists of 6 floats)
            q = rtde_r.getActualQ()               # Joint positions (rad)
            qd = rtde_r.getActualQd()              # Joint velocities (rad/s)
            qdd = rtde_r.getActualQdd()            # Joint accelerations (rad/s^2)
            joint_torques = rtde_r.getActualJointT()  # Joint force/torque (Fx, Fy, Fz, Tx, Ty, Tz) Units: Nm

            # 2. TCP Data (Lists of 6 floats: [x, y, z, rx, ry, rz])
            tcp_pose = rtde_r.getActualTCPPose()    # TCP position and orientation (x, y, z, rx, ry, rz) Units: m, rad
            tcp_wrench = rtde_r.getActualTCPForce()  # TCP force/torque (Fx, Fy, Fz, Tx, Ty, Tz) Units: N, Nm

            header = _ros_header()

            joint_state_pub.publish(roslibpy.Message({
                "header": header,
                "name": JOINT_NAMES,
                "position": q,
                "velocity": qd,
                "effort": joint_torques,
            }))

            position_pub.publish(roslibpy.Message({
                "positions": q,
                "velocities": qd,
                "accelerations": qdd,
            }))

            qx, qy, qz, qw = _rotvec_to_quaternion(*tcp_pose[3:])
            tcp_pose_pub.publish(roslibpy.Message({
                "header": header,
                "pose": {
                    "position": {"x": tcp_pose[0], "y": tcp_pose[1], "z": tcp_pose[2]},
                    "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                },
            }))

            wrench_pub.publish(roslibpy.Message({
                "header": header,
                "wrench": {
                    "force": {"x": tcp_wrench[0], "y": tcp_wrench[1], "z": tcp_wrench[2]},
                    "torque": {"x": tcp_wrench[3], "y": tcp_wrench[4], "z": tcp_wrench[5]},
                },
            }))

            if plotter is not None:
                loop_count += 1
                if loop_count % PLOT_UPDATE_EVERY_N_LOOPS == 0:
                    plotter.update(tcp_wrench)

            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)
    finally:
        for publisher in (joint_state_pub, position_pub, tcp_pose_pub, wrench_pub):
            try:
                publisher.unadvertise()
            except Exception:
                pass
        ft_zero_topic.unsubscribe()
        ros_client.terminate()
        if rtde_c is not None:
            rtde_c.disconnect()
        rtde_r.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR RTDE state reader / ROS publisher.")
    parser.add_argument("--plot", action="store_true", help="Show a live plot of the TCP force/torque wrench.")
    parser.add_argument(
        "--enable-ft-zero",
        action="store_true",
        help="open an RTDE control session so this reader can tare the FT sensor. Off by "
        "default: a controller has only one control session, and taking it here drops "
        "whoever else holds it (see ur_reader_node).",
    )
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _parse_args()
    try:
        ur_reader_node(enable_plot=args.plot, enable_ft_zero=args.enable_ft_zero)
    except KeyboardInterrupt:
        pass
