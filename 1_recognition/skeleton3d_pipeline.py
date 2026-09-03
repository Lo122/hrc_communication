"""Realtime 3D posture pipeline for RecognitionManager.

YOLO 2D pose -> KeypointOutlierHoldFilter -> MotionBERT streaming 2D->3D
lift -> (optional) world-frame fusion via MetricDepthEstimator + calibrated
extrinsics, plus a streaming version of skeleton_pipeline/features/
h36m_features.py's compute_all_features panels (one frame at a time instead
of a whole clip) -- the feature names/shapes must match that module's
panel/column naming exactly, since the trained model's norm stats (the
*_mean.npy/*_std.npy pairs in best_model/3d_skeleton/*.npz) were computed
offline with h36m_features.py against those exact names.

This mirrors src/pose_detection_live.py's pipeline -- see that module's
docstring for the full posture ("root-relative body shape") vs location
("absolute distance from the camera") reasoning and the world-frame
convention (world_skeleton = world_root_xyz + R_world_from_camera @
root_relative). RecognitionManager.update_from_frame uses the building
blocks here instead of pose_detection_live.py's standalone preview loop.

Without calibrated extrinsics (see src/pose_detection_live.py's docstring
for how to produce them via app/calibrate_camera.py), `skeleton` falls back
to MotionBERT's camera-frame root-relative shape -- posture only, tilted by
however the camera happens to be mounted, since there is no gravity
reference without extrinsics.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from vision_model.vision_config import VisionConfig

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from camera_utils.calibration_io import load_extrinsics, load_intrinsics  # noqa: E402
from camera_utils import transforms as tf  # noqa: E402
from skeleton_pipeline.coco_h36m import coco_to_h36m_xy  # noqa: E402
from skeleton_pipeline.keypoint_filter import KeypointOutlierHoldFilter  # noqa: E402
from skeleton_pipeline.metric_depth_estimator import MetricDepthEstimator  # noqa: E402
from skeleton_pipeline.features.h36m_features import (  # noqa: E402
    FEATURE_JOINTS, compute_distance_from_center_ratio, compute_joint_angles,
    compute_polar, compute_ratios,
)

CALIB_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "calib_data"

# H36M-17 indices (see skeleton_pipeline/coco_h36m.py's docstring).
H36M_ROOT = 0
H36M_THORAX = 8

# Deterministic column order for the streaming "joint_angles"/"ratios"
# feature vectors -- must match compute_joint_angles()/compute_ratios()'s
# dict keys exactly (see skeleton_pipeline/features/h36m_features.py).
JOINT_ANGLE_KEYS = [
    "left_elbow_angle_deg", "right_elbow_angle_deg",
    "left_shoulder_angle_deg", "right_shoulder_angle_deg",
    "left_hip_angle_deg", "right_hip_angle_deg",
    "left_knee_angle_deg", "right_knee_angle_deg",
    "neck_angle_deg",
]
RATIO_KEYS = ["elbow_over_shoulder_ratio", "wrist_over_shoulder_ratio"]


def estimate_pitch_from_torso_vector(root_relative: np.ndarray) -> float:
    """See src/pose_detection_live.py's function of the same name --
    angle (radians, unsigned) between the pelvis->thorax vector and
    vertical, fed to MetricDepthEstimator's foreshortening correction."""
    torso_vector = root_relative[H36M_THORAX]
    norm = float(np.linalg.norm(torso_vector))
    if norm < 1e-6:
        return 0.0
    cos_pitch = np.clip(torso_vector[2] / norm, -1.0, 1.0)
    return float(np.arccos(cos_pitch))


def run_yolo_2d(yolo, frame, imgsz=None):
    kwargs = {"imgsz": imgsz} if imgsz is not None else {}
    results = yolo(frame, verbose=False, **kwargs)
    if not results or results[0].keypoints is None or results[0].keypoints.xy.numel() == 0:
        return None, None
    kpts_xy = results[0].keypoints.xy[0].cpu().numpy()
    conf = (results[0].keypoints.conf[0].cpu().numpy()
            if results[0].keypoints.conf is not None
            else np.ones(kpts_xy.shape[0], dtype=np.float32))
    return kpts_xy, conf


class RealtimeSkeleton3DPipeline:
    """Turns raw BGR frames into a (optionally world-located) H36M-17
    root-relative skeleton, one frame at a time -- the 3D building blocks
    of src/pose_detection_live.py packaged for RecognitionManager.

    config: VisionConfig (see vision_model/vision_config.py) -- bundles
    the YOLO/calibration/MotionBERT/depth-estimator knobs that used to be
    ~15 separate keyword args here.
    """

    def __init__(self, config: VisionConfig):
        from ultralytics import YOLO
        from skeleton_pipeline.motionbert_lifter import (
            DEFAULT_CHECKPOINT, DEFAULT_CONFIG, MotionBERTStreamingLifter,
        )

        self.config = config
        self.conf_threshold = config.conf_threshold
        self.user_height_m = config.user_height_m
        self.yolo_imgsz = config.yolo_imgsz

        calib_dir = Path(config.camera.calib_dir or CALIB_DATA_DIR)
        intrinsics_path = calib_dir / config.camera.intrinsics_file
        extrinsics_path = calib_dir / config.camera.extrinsics_file
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"No intrinsics found at {intrinsics_path}. 3D posture recognition needs a "
                "calibrated camera -- run src/app/calibrate_camera.py (see "
                "src/pose_detection_live.py's module docstring) first.")
        self.K, _dist, _image_size = load_intrinsics(intrinsics_path)
        fy = float(self.K[1, 1])

        self.T_world_from_camera = None
        self.have_extrinsics = extrinsics_path.exists()
        if self.have_extrinsics:
            self.T_world_from_camera, _ground_z, _robot_base = load_extrinsics(extrinsics_path)
        else:
            print(f"NOTE: no extrinsics at {extrinsics_path} -- posture will stay camera-frame "
                  "(tilted by however the camera is mounted, no gravity/world reference). Run "
                  "src/app/calibrate_camera.py's extrinsic step to fix this.")

        device = config.device or "cpu"
        self.yolo = YOLO(str(config.yolo_model_path))
        self.lifter = MotionBERTStreamingLifter(
            config_path=config.motionbert_config or DEFAULT_CONFIG,
            checkpoint_path=config.motionbert_checkpoint or DEFAULT_CHECKPOINT,
            clip_len=config.clip_len, device=device)
        self.kp_filter = KeypointOutlierHoldFilter() if config.use_keypoint_filter else None
        self.depth_estimator = MetricDepthEstimator(
            focal_length_y=fy, min_cutoff=config.min_cutoff, beta=config.beta,
            d_cutoff=config.d_cutoff)

    def process(self, frame, timestamp: float) -> dict[str, Any]:
        """Returns a dict with keys: keypoints_2d, keypoints_conf,
        kp_status, root_relative (camera-frame MotionBERT output, or None),
        world_root_xyz ((3,), nan-filled if unavailable), skeleton (what
        feature extraction should consume -- world-frame fusion if
        calibrated extrinsics + depth were available this frame, else the
        camera-frame root_relative, else None if no detection at all)."""
        h, w = frame.shape[:2]
        keypoints_2d, keypoints_conf = run_yolo_2d(self.yolo, frame, imgsz=self.yolo_imgsz)
        kp_status = None
        if self.kp_filter is not None:
            keypoints_2d, keypoints_conf, kp_status = self.kp_filter.filter(keypoints_2d, keypoints_conf)

        out: dict[str, Any] = {
            "keypoints_2d": keypoints_2d,
            "keypoints_conf": keypoints_conf,
            "kp_status": kp_status,
            "root_relative": None,
            "world_root_xyz": np.full(3, np.nan),
            "skeleton": None,
        }
        if keypoints_2d is None or not np.any(keypoints_2d):
            return out

        root_relative = self.lifter.lift(keypoints_2d, image_size=(w, h), keypoints_conf=keypoints_conf)
        if root_relative is None:
            return out
        out["root_relative"] = root_relative
        out["skeleton"] = root_relative  # fallback if no extrinsics/depth this frame

        pitch_rad = estimate_pitch_from_torso_vector(root_relative)
        masked_kpts = keypoints_2d.astype(np.float64).copy()
        masked_kpts[keypoints_conf < self.conf_threshold] = np.nan
        z_filtered = self.depth_estimator.update(
            track_id=0, keypoints_2d=masked_kpts, user_height_meters=self.user_height_m,
            pitch_angle_rad=pitch_rad, timestamp=timestamp)

        if z_filtered is not None and self.have_extrinsics:
            pelvis_px = coco_to_h36m_xy(keypoints_2d)[H36M_ROOT]
            root_camera_xyz = tf.pixel_depth_to_camera_point(self.K, pelvis_px, z_filtered)
            world_root_xyz = tf.camera_point_to_world(self.T_world_from_camera, root_camera_xyz)
            out["world_root_xyz"] = world_root_xyz
            # POSTURE (gravity-aligned shape) + LOCATION (absolute root position)
            # fused into one world-frame skeleton -- see module docstring.
            out["skeleton"] = world_root_xyz + tf.transform_directions(
                self.T_world_from_camera, root_relative)

        return out


class StreamingH36MFeatureExtractor:
    """Turns a stream of (17, 3) H36M skeletons into the same feature
    panels as skeleton_pipeline/features/h36m_features.py's
    compute_all_features (joint_speed, joint_acceleration,
    joint_velocity_x/y/z, joint_acceleration_x/y/z,
    position_x/y/z_relative_to_pelvis, polar_azimuth, polar_elevation,
    joint_angles, ratios, distance_from_center -- see that module's
    docstring for the formulas and why Savitzky-Golay smoothing is used),
    one frame at a time. The dict keys/value ordering (FEATURE_JOINTS,
    i.e. all 17 H36M joints except pelvis) must match h36m_features.py's
    exactly -- NormRealTime.normalize_features looks up mean/std for every
    key this returns by that exact name (see best_model/3d_skeleton's
    norm .npz).

    Streaming/causal difference from the offline version: each call fits
    over the trailing `window_length` frames seen so far (a causal window,
    not a centered one), and falls back to the raw last frame (zero
    velocity/acceleration) while the buffer is still shorter than the fit
    needs.

    Real (not fixed) frame spacing: h36m_features.py's offline
    _savgol_derivative assumes uniformly-spaced samples (a single `delta`
    scalar), which is a fine assumption for a video decoded at a constant
    fps. It is NOT a fine assumption here: run_recognition.py's live loop
    is a fixed-TICK scheduler, not a fixed-WORK one, so whenever
    recognition_manager.update() (YOLO+MotionBERT+depth+features+LSTM)
    overruns the tick budget, actual wall-clock frame spacing stretches
    unevenly -- assuming a constant delta would silently mis-scale
    velocity/acceleration (e.g. a slow frame's real 80ms gap treated as
    the nominal 33ms). So instead of scipy's savgol_filter (which only
    takes a fixed delta), this fits a degree-`polyorder` polynomial of
    POSITION vs. each frame's actual timestamp (see _fit_derivative) --
    the non-uniform-time generalization of the same "local polynomial fit,
    don't finite-difference the raw signal" idea.

    config: VisionConfig (see vision_model/vision_config.py) -- uses its
    feature_window_length/feature_polyorder fields.
    """

    def __init__(self, config: VisionConfig):
        self.window_length = config.feature_window_length
        self.polyorder = config.feature_polyorder
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=self.window_length)
        self._last_valid_skeleton: np.ndarray | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._last_valid_skeleton = None

    def update(self, skeleton_17x3: np.ndarray | None, timestamp: float) -> dict[str, np.ndarray]:
        """skeleton_17x3: (17, 3) or None (no detection this frame -- holds
        the last valid skeleton, same "hold" philosophy as
        KeypointOutlierHoldFilter upstream; zeros if there's no valid
        skeleton yet at all). timestamp: this frame's wall-clock/video
        time in seconds (e.g. RecognitionManager.update_from_frame's
        frame_timestamp) -- the ACTUAL time this skeleton was observed at,
        used for velocity/acceleration scaling instead of an assumed fixed
        frame interval (see class docstring)."""
        if skeleton_17x3 is None:
            skeleton_17x3 = self._last_valid_skeleton
        if skeleton_17x3 is None:
            skeleton_17x3 = np.zeros((17, 3), dtype=np.float64)
        self._last_valid_skeleton = skeleton_17x3
        self._buffer.append((float(timestamp), np.asarray(skeleton_17x3, dtype=np.float64)))

        times = np.array([t for t, _ in self._buffer], dtype=np.float64)
        window = np.stack([s for _, s in self._buffer], axis=0)  # (t<=window_length, 17, 3)
        position = self._fit_derivative(times, window, deriv=0)[None]  # (1, 17, 3)
        velocity = self._fit_derivative(times, window, deriv=1)[None]
        acceleration = self._fit_derivative(times, window, deriv=2)[None]

        azimuth, elevation = compute_polar(position)
        joint_angles = compute_joint_angles(position)
        ratios = compute_ratios(position)
        dist_ratio = compute_distance_from_center_ratio(position)
        speed = np.linalg.norm(velocity, axis=-1)
        accel_mag = np.linalg.norm(acceleration, axis=-1)

        joint_angles_vec = np.array(
            [joint_angles[key][0] for key in JOINT_ANGLE_KEYS], dtype=np.float32)
        ratios_vec = np.array(
            [ratios[key][0] for key in RATIO_KEYS], dtype=np.float32)

        features = {
            "joint_speed": speed[0, FEATURE_JOINTS].astype(np.float32),
            "joint_acceleration": accel_mag[0, FEATURE_JOINTS].astype(np.float32),
            "polar_azimuth": azimuth[0, FEATURE_JOINTS].astype(np.float32),
            "polar_elevation": elevation[0, FEATURE_JOINTS].astype(np.float32),
            "joint_angles": joint_angles_vec,
            "ratios": ratios_vec,
            "distance_from_center": dist_ratio[0, FEATURE_JOINTS].astype(np.float32),
        }
        for axis_idx, axis_name in enumerate("xyz"):
            features[f"joint_velocity_{axis_name}"] = velocity[0, FEATURE_JOINTS, axis_idx].astype(np.float32)
            features[f"joint_acceleration_{axis_name}"] = acceleration[0, FEATURE_JOINTS, axis_idx].astype(np.float32)
            features[f"position_{axis_name}_relative_to_pelvis"] = position[0, FEATURE_JOINTS, axis_idx].astype(np.float32)

        return features

    def _fit_derivative(self, times: np.ndarray, window: np.ndarray, deriv: int) -> np.ndarray:
        """times: (t<=window_length,) each buffered frame's actual
        timestamp, seconds -- NOT assumed uniformly spaced (see class
        docstring). window: (t, 17, 3), same order as times. Returns the
        deriv-th derivative (0=smoothed position, 1=velocity,
        2=acceleration) at the LATEST timestamp: fits one degree-
        `polyorder` least-squares polynomial per joint/axis of position
        vs. real elapsed time (np.polyfit's x can be arbitrarily spaced,
        unlike savgol_filter's fixed delta), then differentiates that
        polynomial analytically and evaluates at the most recent frame --
        centering time at the latest frame (t_rel = times - times[-1], so
        it's always 0) means "evaluate the deriv-th derivative at the
        latest sample" is just deriv! times the x**deriv coefficient, no
        separate evaluation step needed. Falls back to the raw last frame
        (deriv=0) / zero (deriv>0) while the buffer is still too short to
        fit reliably -- same 2-more-than-polyorder margin
        _savgol_derivative used, so a degenerate/overfit exact-interpolation
        polynomial (zero residual, unstable derivatives) is never fit."""
        t = times.shape[0]
        if t < self.polyorder + 2:
            return window[-1] if deriv == 0 else np.zeros_like(window[-1])
        if deriv > self.polyorder:
            return np.zeros_like(window[-1])

        t_rel = times - times[-1]  # latest frame at x=0
        flat = window.reshape(t, -1)  # (t, 17*3)
        coeffs = np.polyfit(t_rel, flat, self.polyorder)  # (polyorder+1, 17*3), highest power first

        # d^deriv/dx^deriv (a_n x^n + ... + a_0) at x=0 == deriv! * a_deriv;
        # coeffs is ordered highest power first, so a_deriv is row (polyorder - deriv).
        coeff_row = coeffs[self.polyorder - deriv]
        derivative_at_latest = math.factorial(deriv) * coeff_row
        return derivative_at_latest.reshape(window.shape[1:])
