"""UR10e kinematics/dynamics helpers: rotation-vector conversions and
payload (end-effector weight/inertia) gravity compensation.

Split out of read_ur_live_data.py so the robot math can be reused/tested
independently of the RTDE polling loop and ROS publishing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.80665  # m/s^2


def rotvec_to_quaternion(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Convert a UR-style rotation vector (axis-angle) to a quaternion
    (x, y, z, w), as needed for geometry_msgs/Quaternion."""
    rotvec = np.array([rx, ry, rz], dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    axis = rotvec / angle
    s = np.sin(angle / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0))


def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert a UR-style rotation vector (axis-angle, as returned in TCP
    pose) to a 3x3 rotation matrix."""
    rotvec = np.array([rx, ry, rz], dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class Payload:
    """End-effector payload description, matching UR's Installation ->
    Payload dialog."""

    mass: float  # kg
    center_of_gravity: tuple[float, float, float]  # m, in the flange/TCP frame
    inertia: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3))
    )  # kg*m^2 about the CoG, in the flange frame


class GravityCompensator:
    """Subtracts an end-effector's weight (and, optionally, its inertial
    wrench during motion) from a raw flange F/T reading to estimate the
    external (contact/environment) wrench.

    Feed it `ft_raw_wrench` (uncompensated sensor reading), not
    `actual_TCP_force` (which the controller may have already compensated).
    """

    def __init__(self, payload: Payload):
        self.payload = payload
        self._prev_omega: np.ndarray | None = None
        self._prev_time: float | None = None

    def compensate(
        self,
        raw_wrench: np.ndarray,
        tcp_pose: np.ndarray,
        tcp_speed: np.ndarray,
        timestamp: float,
    ) -> np.ndarray:
        """Return the estimated external wrench [Fx,Fy,Fz,Tx,Ty,Tz] in the
        flange/sensor frame.

        raw_wrench: 6-vector from `ft_raw_wrench`.
        tcp_pose: 6-vector [x,y,z,rx,ry,rz] from `actual_TCP_pose`.
        tcp_speed: 6-vector [vx,vy,vz,wx,wy,wz] from `actual_TCP_speed`.
        """
        m = self.payload.mass
        r_com = np.array(self.payload.center_of_gravity, dtype=float)
        I = self.payload.inertia

        R = rpy_to_matrix(*tcp_pose[3:])  # sensor/flange -> base
        omega = np.array(tcp_speed[3:], dtype=float)  # angular vel, base frame

        # --- gravity term, expressed in the sensor/flange frame ---
        g_base = np.array([0.0, 0.0, -GRAVITY])
        g_flange = R.T @ g_base
        F_gravity = m * g_flange
        T_gravity = np.cross(r_com, F_gravity)

        # --- inertial term (finite-differenced angular accel; linear accel
        # from CoM's motion due to rotation about the flange origin) ---
        F_inertial = np.zeros(3)
        T_inertial = np.zeros(3)
        if self._prev_omega is not None and self._prev_time is not None:
            dt = timestamp - self._prev_time
            if dt > 1e-6:
                omega_flange = R.T @ omega
                alpha_flange = (omega_flange - self._prev_omega) / dt
                a_com = np.cross(alpha_flange, r_com) + np.cross(
                    omega_flange, np.cross(omega_flange, r_com)
                )
                F_inertial = m * a_com
                T_inertial = I @ alpha_flange + np.cross(
                    omega_flange, I @ omega_flange
                )
            self._prev_omega = R.T @ omega
        else:
            self._prev_omega = R.T @ omega
        self._prev_time = timestamp

        F_ext = raw_wrench[:3] - F_gravity - F_inertial
        T_ext = raw_wrench[3:] - T_gravity - T_inertial
        return np.concatenate([F_ext, T_ext])
