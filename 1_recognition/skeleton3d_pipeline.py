"""Realtime 3D posture pipeline for RecognitionManager.

YOLO 2D pose -> KeypointOutlierHoldFilter -> MotionBERT streaming 2D->3D
lift -> (optional) world-frame fusion via MetricDepthEstimator + calibrated
extrinsics, plus a streaming version of skeleton_pipeline/features/
h36m_features.py's pol_angles/joint_angles/ratios panels (one frame at a
time instead of a whole clip).

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
    FEATURE_JOINTS, compute_joint_angles, compute_polar, compute_ratios,
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
    """Turns a stream of (17, 3) H36M skeletons into the same
    "pol_angles" / "joint_angles" / "ratios" feature panels as
    skeleton_pipeline/features/h36m_features.py's compute_all_features, one
    frame at a time -- see that module's docstring for the formulas and why
    Savitzky-Golay smoothing is used for the position these are derived
    from.

    Streaming/causal difference from the offline version: each call
    smooths over the trailing `window_length` frames seen so far (a causal
    window, not a centered one), and falls back to the raw last frame while
    the buffer is still shorter than the smoothing window needs.

    config: VisionConfig (see vision_model/vision_config.py) -- uses its
    fps/feature_window_length/feature_polyorder fields.
    """

    def __init__(self, config: VisionConfig):
        self.fps = config.fps
        self.window_length = config.feature_window_length
        self.polyorder = config.feature_polyorder
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window_length)
        self._last_valid_skeleton: np.ndarray | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._last_valid_skeleton = None

    def update(self, skeleton_17x3: np.ndarray | None) -> dict[str, np.ndarray]:
        """skeleton_17x3: (17, 3) or None (no detection this frame -- holds
        the last valid skeleton, same "hold" philosophy as
        KeypointOutlierHoldFilter upstream; zeros if there's no valid
        skeleton yet at all)."""
        if skeleton_17x3 is None:
            skeleton_17x3 = self._last_valid_skeleton
        if skeleton_17x3 is None:
            skeleton_17x3 = np.zeros((17, 3), dtype=np.float64)
        self._last_valid_skeleton = skeleton_17x3
        self._buffer.append(np.asarray(skeleton_17x3, dtype=np.float64))

        window = np.stack(self._buffer, axis=0)  # (t<=window_length, 17, 3)
        smoothed_last = self._smoothed_last_frame(window)[None]  # (1, 17, 3)

        azimuth, elevation = compute_polar(smoothed_last)
        joint_angles = compute_joint_angles(smoothed_last)
        ratios = compute_ratios(smoothed_last)

        pol_angles = np.concatenate([
            azimuth[0, FEATURE_JOINTS], elevation[0, FEATURE_JOINTS],
        ]).astype(np.float32)
        joint_angles_vec = np.array(
            [joint_angles[key][0] for key in JOINT_ANGLE_KEYS], dtype=np.float32)
        ratios_vec = np.array(
            [ratios[key][0] for key in RATIO_KEYS], dtype=np.float32)

        return {
            "pol_angles": pol_angles,
            "joint_angles": joint_angles_vec,
            "ratios": ratios_vec,
        }

    def _smoothed_last_frame(self, window: np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter

        t = window.shape[0]
        wl = min(self.window_length, t if t % 2 == 1 else t - 1)
        if wl < self.polyorder + 2:
            return window[-1]  # not enough data yet -- raw last frame
        smoothed = savgol_filter(
            window, window_length=wl, polyorder=self.polyorder, deriv=0,
            delta=1.0 / self.fps, axis=0)
        return smoothed[-1]
