"""Live per-person metric depth (Z_root) estimation from 2D keypoints alone,
using known camera intrinsics and an assumed/measured body height -- no
depth network required. Self-contained copy of
world_pose/pose/metric_depth_estimator.py (numpy/math/time only, no extra
dependency) for this app/venv, which deliberately avoids heavier monocular
depth models (Depth Pro, yolo26-depth) -- see that module's docstring for
the full method (anthropometric torso-height depth, pitch-foreshortening
correction, 1-Euro temporal smoothing) and world_pose/tests/
test_metric_depth_estimator.py for its test coverage.
"""
import math
import time

import numpy as np

# Fraction of total body height spanned by the shoulder-midpoint ->
# hip-midpoint segment. A population-average anthropometric ratio, not
# measured per-person.
TORSO_HEIGHT_RATIO = 0.29

# COCO-17 keypoint indices (matches ultralytics YOLO-pose output order).
COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER = 5, 6
COCO_LEFT_HIP, COCO_RIGHT_HIP = 11, 12


class OneEuroFilter:
    """Self-contained 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012) for a
    single scalar signal sampled at irregular timestamps.

    min_cutoff: base cutoff frequency (Hz) used when the signal appears
        stationary -- lower = smoother but laggier at rest.
    beta: how much the cutoff frequency increases with the estimated speed
        of the signal -- higher = reacts faster to real, fast changes, at
        the cost of passing through more noise while moving.
    d_cutoff: cutoff frequency (Hz) for the internal derivative estimate
        used to compute that speed. Rarely needs tuning.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        """Drop all history -- the next __call__ is treated as the first sample."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _smoothing_factor(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, timestamp: float) -> float:
        """Filter one new sample. `timestamp` must be non-decreasing across calls
        (seconds; any consistent unit works since only differences are used)."""
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return x

        dt = max(timestamp - self._t_prev, 1e-6)  # guard against dt<=0 (duplicate/out-of-order timestamps)

        # Low-pass the derivative first, to get a stable speed estimate.
        dx = (x - self._x_prev) / dt
        a_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Faster estimated speed -> higher cutoff -> less smoothing/lag.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp
        return x_hat


def _extract_torso_midpoints(keypoints_2d):
    """keypoints_2d: either a (17, 2) array-like in COCO order, or a dict
    with keys "left_shoulder", "right_shoulder", "left_hip", "right_hip"
    (pixel (x, y) pairs). Returns (shoulder_mid, hip_mid), each an (2,)
    ndarray, or None if any required joint is missing/non-finite (NaN/inf
    is this module's convention for "not visible enough to use" -- callers
    with a confidence score should substitute NaN for low-confidence
    joints before calling update())."""
    if isinstance(keypoints_2d, dict):
        try:
            ls = np.asarray(keypoints_2d["left_shoulder"], dtype=np.float64)
            rs = np.asarray(keypoints_2d["right_shoulder"], dtype=np.float64)
            lh = np.asarray(keypoints_2d["left_hip"], dtype=np.float64)
            rh = np.asarray(keypoints_2d["right_hip"], dtype=np.float64)
        except KeyError:
            return None
    else:
        keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
        required = (COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP)
        if keypoints_2d.ndim != 2 or keypoints_2d.shape[0] <= max(required):
            return None
        ls, rs = keypoints_2d[COCO_LEFT_SHOULDER], keypoints_2d[COCO_RIGHT_SHOULDER]
        lh, rh = keypoints_2d[COCO_LEFT_HIP], keypoints_2d[COCO_RIGHT_HIP]

    joints = (ls, rs, lh, rh)
    if not all(np.all(np.isfinite(j)) for j in joints):
        return None

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    return shoulder_mid, hip_mid


class _PersonDepthState:
    """Per-track_id state: its own 1-Euro filter instance plus the last
    successfully filtered depth, so a frame with unusable keypoints can
    fall back to "whatever we last knew" instead of returning garbage."""

    __slots__ = ("filter", "last_depth")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.last_depth = None


class MetricDepthEstimator:
    """Tracks one 1-Euro-filtered metric depth estimate per person (by
    track_id), computed from 2D torso keypoints + known camera intrinsics +
    an assumed body height -- see module docstring for the full method.
    """

    def __init__(self, focal_length_y: float, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0,
                 min_torso_pixels: float = 5.0):
        """
        focal_length_y: camera's fy intrinsic, in pixels (same units as the
            keypoint pixel coordinates passed to update()).
        min_cutoff, beta, d_cutoff: OneEuroFilter parameters, applied to
            every tracked person's depth stream (see OneEuroFilter's
            docstring for what each controls).
        min_torso_pixels: treat a torso projected to fewer pixels than this
            as degenerate/out-of-bounds input (e.g. keypoints collapsed
            onto each other) rather than dividing by a near-zero number.
        """
        self.focal_length_y = float(focal_length_y)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.min_torso_pixels = float(min_torso_pixels)
        self._people: dict[int, _PersonDepthState] = {}

    def _state_for(self, track_id: int) -> _PersonDepthState:
        state = self._people.get(track_id)
        if state is None:
            state = _PersonDepthState(self.min_cutoff, self.beta, self.d_cutoff)
            self._people[track_id] = state
        return state

    def update(self, track_id: int, keypoints_2d, user_height_meters: float = 1.70,
               pitch_angle_rad: float = 0.0, timestamp: float = None) -> float:
        """Feed one new frame's keypoints for one tracked person; returns
        the current filtered metric depth estimate (meters), or None if
        this person has never had a usable measurement yet.

        track_id: stable identity for this person across frames (from
            whatever tracker assigns them) -- each track_id gets its own
            independent 1-Euro filter, so switching track_ids does not
            reuse another person's smoothing state.
        keypoints_2d: (17, 2) array-like in COCO order, or a dict with
            "left_shoulder"/"right_shoulder"/"left_hip"/"right_hip" pixel
            coordinates. Use NaN (or omit the dict key) for any joint that
            isn't visible/confident enough to trust.
        user_height_meters: this person's real height, if known; the
            population-average default (1.70m) is used otherwise.
        pitch_angle_rad: forward/backward torso lean, 0 = upright facing
            the camera -- see module docstring's foreshortening-correction
            section.
        timestamp: seconds (any consistent clock); defaults to time.time().
        """
        state = self._state_for(track_id)
        timestamp = time.time() if timestamp is None else float(timestamp)

        torso = _extract_torso_midpoints(keypoints_2d)
        if torso is not None:
            shoulder_mid, hip_mid = torso
            h_torso_pixels = float(np.linalg.norm(shoulder_mid - hip_mid))

            if h_torso_pixels >= self.min_torso_pixels:
                h_torso_meters = user_height_meters * TORSO_HEIGHT_RATIO
                cos_pitch = max(math.cos(pitch_angle_rad), 0.2)
                h_torso_corrected = h_torso_pixels / cos_pitch

                z_raw = (self.focal_length_y * h_torso_meters) / h_torso_corrected
                z_filtered = state.filter(z_raw, timestamp)
                state.last_depth = z_filtered
                return z_filtered

        # Missing/low-confidence/degenerate keypoints this frame -- fall
        # back to the last good estimate (None if we've never had one).
        return state.last_depth

    def remove_track(self, track_id: int) -> None:
        """Drop a person's state entirely (e.g. once their track ends), so a
        later, unrelated person reusing that track_id doesn't inherit stale
        filter history."""
        self._people.pop(track_id, None)

    def reset(self, track_id: int = None) -> None:
        """Clear filter state -- for one track_id, or every tracked person
        if track_id is omitted."""
        if track_id is None:
            self._people.clear()
        else:
            self._people.pop(track_id, None)
