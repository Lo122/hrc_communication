"""ROS1 message bodies for the UR state the RTDE readers publish via rosbridge.

eval/read_ur_live_data.py and eval/ur_state_reader.py publish the same message
types from the same RTDE samples, on different topics (see config.ROS_TOPICS).
Both build their messages here, as plain dicts ready for roslibpy.Message --
nothing here imports roslibpy, so the message layout can be checked without a
rosbridge.

Callers pass their own joint names and frame_ids, so each topic keeps what it
has always published.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from robot_utils import rotvec_to_quaternion


def ros_time(timestamp: float) -> dict:
    """Float Unix time -> ROS1 time ({secs, nsecs}), as in std_msgs/Header.stamp.

    The fraction is scaled on its own (timestamp * 1e9 as a whole would be past
    float precision, ~256 ns steps); a fraction that rounds up to a full second
    (e.g. 5.9999999999) carries into secs, so nsecs always stays < 1e9.
    """
    secs = math.floor(timestamp)
    nsecs = round((timestamp - secs) * 1e9)
    if nsecs == 10**9:
        secs, nsecs = secs + 1, 0
    return {"secs": secs, "nsecs": nsecs}


def header(timestamp: float, frame_id: str = "") -> dict:
    return {"stamp": ros_time(timestamp), "frame_id": frame_id}


def joint_state(
    timestamp: float,
    positions: Sequence[float],
    *,
    names: Sequence[str],
    velocities: Sequence[float] = (),
    efforts: Sequence[float] = (),
    frame_id: str = "",
) -> dict:
    """sensor_msgs/JointState; velocities/efforts may be left empty."""
    return {
        "header": header(timestamp, frame_id),
        "name": list(names),
        "position": _floats(positions),
        "velocity": _floats(velocities),
        "effort": _floats(efforts),
    }


def pose_stamped(timestamp: float, tcp_pose: Sequence[float], frame_id: str) -> dict:
    """geometry_msgs/PoseStamped from a UR pose [x, y, z, rx, ry, rz] (m, rotation
    vector in rad)."""
    qx, qy, qz, qw = rotvec_to_quaternion(*tcp_pose[3:6])
    return {
        "header": header(timestamp, frame_id),
        "pose": {
            "position": _xyz(tcp_pose[0:3]),
            "orientation": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
        },
    }


def wrench_stamped(timestamp: float, wrench: Sequence[float], frame_id: str) -> dict:
    """geometry_msgs/WrenchStamped from [Fx, Fy, Fz, Tx, Ty, Tz] (N, Nm)."""
    return {
        "header": header(timestamp, frame_id),
        "wrench": {"force": _xyz(wrench[0:3]), "torque": _xyz(wrench[3:6])},
    }


def joint_trajectory_point(
    positions: Sequence[float],
    velocities: Sequence[float],
    accelerations: Sequence[float],
) -> dict:
    """trajectory_msgs/JointTrajectoryPoint (no header in this message type)."""
    return {
        "positions": _floats(positions),
        "velocities": _floats(velocities),
        "accelerations": _floats(accelerations),
    }


def _floats(values: Sequence[float]) -> list[float]:
    # numpy scalars are float subclasses, but plain floats keep the JSON obvious.
    return [float(v) for v in values]


def _xyz(values: Sequence[float]) -> dict:
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}
