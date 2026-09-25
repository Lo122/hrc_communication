"""Constant-velocity Kalman filter for the human's world-frame root position.

Applied to world_root_xyz (see skeleton3d_pipeline.py), after the pixel -> camera
-> world transform, so its noise parameters are in real metres on the floor
rather than in pixels or along the camera ray.

State is [x, y, z, vx, vy, vz] in the world frame. The time step between two
updates is the difference of their real timestamps (seconds) -- never a frame
count or an assumed 1/fps. Live frames arrive irregularly (YOLO/MotionBERT
latency varies, frames are dropped), and a fixed per-frame dt would report a
wrong velocity whenever the actual spacing differs from the nominal one.

Process model: white-noise acceleration (the discretised continuous model), so
the process noise grows with dt the way real motion uncertainty does. A human
standing still and a human walking are both covered by accel_std; a larger
value tracks turns/starts faster, a smaller one smooths harder.

Outliers: an innovation whose Mahalanobis distance exceeds gate_chi2 is
rejected and the prediction is returned instead. If max_consecutive_rejects
measurements in a row are rejected, the person has most likely really moved
(or a different detection took over) and the filter re-initialises on the new
measurement rather than defending a stale track forever.
"""
from __future__ import annotations

import numpy as np

# 99.7% quantile of chi-square with 3 degrees of freedom.
DEFAULT_GATE_CHI2 = 14.16


class PositionKalmanFilter:
    def __init__(self, measurement_std_m: float = 0.08, accel_std_mps2: float = 2.0,
                 initial_velocity_std_mps: float = 1.0, max_gap_s: float = 1.0,
                 gate_chi2: float | None = DEFAULT_GATE_CHI2,
                 max_consecutive_rejects: int = 3):
        """
        measurement_std_m: per-axis std of one world_root_xyz measurement.
        accel_std_mps2: std of the unmodelled acceleration (process noise).
        initial_velocity_std_mps: velocity uncertainty right after (re)initialising.
        max_gap_s: a gap longer than this between updates re-initialises the
            filter -- predicting a walking person 5 s ahead is worse than
            starting again from the new measurement.
        gate_chi2: Mahalanobis gate for outlier rejection; None disables gating.
        max_consecutive_rejects: re-initialise after this many rejections in a row.
        """
        self.measurement_std_m = float(measurement_std_m)
        self.accel_std_mps2 = float(accel_std_mps2)
        self.initial_velocity_std_mps = float(initial_velocity_std_mps)
        self.max_gap_s = float(max_gap_s)
        self.gate_chi2 = gate_chi2
        self.max_consecutive_rejects = int(max_consecutive_rejects)

        self._H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self._R = np.eye(3) * self.measurement_std_m ** 2
        self.reset()

    def reset(self) -> None:
        """Drop all history -- the next update() initialises from its measurement."""
        self.x: np.ndarray | None = None  # (6,) state
        self.P: np.ndarray | None = None  # (6, 6) covariance
        self.last_timestamp: float | None = None
        self.consecutive_rejects = 0
        self.last_rejected = False

    @property
    def initialized(self) -> bool:
        return self.x is not None

    @property
    def position(self) -> np.ndarray | None:
        return None if self.x is None else self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray | None:
        """World-frame velocity in m/s, None before the first update."""
        return None if self.x is None else self.x[3:].copy()

    def _initialize(self, z: np.ndarray, timestamp: float) -> None:
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.diag([self.measurement_std_m ** 2] * 3
                         + [self.initial_velocity_std_mps ** 2] * 3)
        self.last_timestamp = timestamp
        self.consecutive_rejects = 0
        self.last_rejected = False

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.accel_std_mps2 ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt ** 4 / 4.0) * q
        Q[:3, 3:] = np.eye(3) * (dt ** 3 / 2.0) * q
        Q[3:, :3] = Q[:3, 3:]
        Q[3:, 3:] = np.eye(3) * (dt ** 2) * q
        return F, Q

    def update(self, measurement_xyz, timestamp: float) -> np.ndarray:
        """Fuse one world-frame position measured at `timestamp` (seconds).

        Returns the filtered position (3,). A non-finite measurement is ignored
        and the current estimate returned -- None if there is none yet.
        """
        z = np.asarray(measurement_xyz, dtype=np.float64).reshape(3)
        timestamp = float(timestamp)
        if not np.all(np.isfinite(z)):
            return self.position

        if self.x is None:
            self._initialize(z, timestamp)
            return self.position

        dt = timestamp - self.last_timestamp
        if dt > self.max_gap_s:
            self._initialize(z, timestamp)
            return self.position
        # Duplicate or out-of-order stamp: no time has passed, so skip the
        # predict step rather than predicting backwards.
        dt = max(dt, 0.0)

        F, Q = self._transition(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        innovation = z - self._H @ x_pred
        S = self._H @ P_pred @ self._H.T + self._R
        S_inv = np.linalg.inv(S)

        if self.gate_chi2 is not None:
            mahalanobis2 = float(innovation @ S_inv @ innovation)
            if mahalanobis2 > self.gate_chi2:
                self.consecutive_rejects += 1
                if self.consecutive_rejects >= self.max_consecutive_rejects:
                    self._initialize(z, timestamp)
                    return self.position
                # Keep the prediction: time still advanced, and the grown
                # covariance lets the next good measurement pull harder.
                self.x, self.P = x_pred, P_pred
                self.last_timestamp = max(self.last_timestamp, timestamp)
                self.last_rejected = True
                return self.position

        K = P_pred @ self._H.T @ S_inv
        self.x = x_pred + K @ innovation
        # Joseph form: stays symmetric positive-definite under rounding.
        I_KH = np.eye(6) - K @ self._H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self._R @ K.T
        self.last_timestamp = max(self.last_timestamp, timestamp)
        self.consecutive_rejects = 0
        self.last_rejected = False
        return self.position
