"""Realtime 3D posture pipeline and its streaming feature extractor.

Two classes, used together by RecognitionManager:

  RealtimeSkeleton3DPipeline   frame -> 3D skeleton
  StreamingH36MFeatureExtractor  skeleton stream -> kinematic feature vectors

The pipeline runs YOLO 2D pose, holds outliers, lifts to 3D with MotionBERT, and
-- given calibrated extrinsics and a resolved depth -- fuses posture with an
absolute position into a world-frame skeleton:

    world_skeleton = world_root_xyz + R_world_from_camera @ root_relative

Two distinct quantities travel through here. **Posture** is the root-relative body
shape, with the pelvis at (0,0,0). **Location** is the absolute distance from the
camera, estimated from 2D keypoints, intrinsics and a known body height. Without
extrinsics only posture is available, tilted by however the camera is mounted,
since there is no gravity reference to straighten it against.

The extractor's feature names and shapes must match feature_utils/h36m_features.py
exactly: the trained model's norm stats were computed offline against those names,
so any divergence silently feeds the model mismatched inputs.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# src/ for the sibling packages, 1_recognition/ for vision_model -- the layer's public
# config surface, which lives outside src/.
_SRC_DIR = Path(__file__).resolve().parent
_RECOGNITION_DIR = _SRC_DIR.parent
for _p in (_SRC_DIR, _RECOGNITION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vision_model.vision_config import VisionConfig  # noqa: E402
from logging_setup import get_logger  # noqa: E402
from camera_utils.calibration_io import load_extrinsics, load_intrinsics  # noqa: E402
from camera_utils import transforms as tf  # noqa: E402
from skeleton_utils.body_calibration import NOSE_HEIGHT_RATIO, BodyCalibration  # noqa: E402
from skeleton_utils.bone_length_filter import BoneLengthConstraintFilter  # noqa: E402
from skeleton_utils.coco_h36m import coco_to_h36m_xy  # noqa: E402
from skeleton_utils.keypoint_filter import KeypointOutlierHoldFilter  # noqa: E402
from skeleton_utils.metric_depth_estimator import MetricDepthEstimator  # noqa: E402
from skeleton_utils.person_selection import select_tracked_person_keypoints  # noqa: E402
from feature_utils.h36m_features import (  # noqa: E402
    FEATURE_JOINTS, L_SHOULDER, R_SHOULDER, compute_distance_from_center_ratio,
    compute_joint_angles, compute_polar, compute_ratios,
)

logger = get_logger(__name__)

CALIB_DATA_DIR = _RECOGNITION_DIR / "calib_data"

# H36M-17 indices (see skeleton_utils/coco_h36m.py's docstring).
H36M_ROOT = 0
H36M_THORAX = 8

# Column order for the "joint_angles"/"ratios" feature vectors. Must match
# compute_joint_angles()/compute_ratios()' dict keys exactly.
JOINT_ANGLE_KEYS = [
    "left_elbow_angle_deg", "right_elbow_angle_deg",
    "left_shoulder_angle_deg", "right_shoulder_angle_deg",
    "left_hip_angle_deg", "right_hip_angle_deg",
    "left_knee_angle_deg", "right_knee_angle_deg",
    "neck_angle_deg",
]
RATIO_KEYS = ["elbow_over_shoulder_ratio", "wrist_over_shoulder_ratio"]

# Samples behind the running shoulder-width median. Deliberately long: shoulder width
# is an anatomical constant, so this is a per-subject calibration that should settle and
# stop moving, not a smoothing window. 3000 is ~5 min at 10 Hz.
SHOULDER_WIDTH_HISTORY = 3000


def estimate_pitch_from_torso_vector(root_relative: np.ndarray) -> float:
    """Unsigned angle in radians between the pelvis->thorax vector and vertical.

    Feeds the depth estimator's foreshortening correction: a leaning torso covers
    fewer pixels than an upright one at the same distance.
    """
    torso_vector = root_relative[H36M_THORAX]
    norm = float(np.linalg.norm(torso_vector))
    if norm < 1e-6:
        return 0.0
    cos_pitch = np.clip(torso_vector[2] / norm, -1.0, 1.0)
    return float(np.arccos(cos_pitch))


def run_yolo_2d(yolo, frame, imgsz=None, selection=None,
                previous_keypoints=None, previous_conf=None):
    """Detect 2D keypoints and return ONE person's (keypoints_xy, conf).

    selection: None (default) keeps the historical behaviour of taking YOLO's
        first listed detection -- fine when only one person is ever in frame.
        A dict of select_tracked_person_keypoints() kwargs instead picks the
        detection matching `previous_keypoints`, which is what stops a
        bystander from stealing the subject mid-run. See
        skeleton_utils/person_selection.py.
    """
    kwargs = {"imgsz": imgsz} if imgsz is not None else {}
    results = yolo(frame, verbose=False, **kwargs)
    if not results or results[0].keypoints is None or results[0].keypoints.xy.numel() == 0:
        return None, None

    keypoints = results[0].keypoints
    if selection is None:
        kpts_xy = keypoints.xy[0].cpu().numpy()
        conf = (keypoints.conf[0].cpu().numpy()
                if keypoints.conf is not None
                else np.ones(kpts_xy.shape[0], dtype=np.float32))
        return kpts_xy, conf

    all_xy = keypoints.xy.cpu().numpy()  # (n_people, 17, 2)
    all_conf = (keypoints.conf.cpu().numpy()
                if keypoints.conf is not None
                else np.ones(all_xy.shape[:2], dtype=np.float32))
    # Box confidence is the "is this a person at all" score, which is a better
    # filter for phantom detections than mean keypoint confidence alone.
    boxes = getattr(results[0], "boxes", None)
    det_scores = (boxes.conf.cpu().numpy()
                  if boxes is not None and getattr(boxes, "conf", None) is not None
                  else None)

    return select_tracked_person_keypoints(
        all_xy, all_conf, det_scores,
        previous_keypoints=previous_keypoints, previous_conf=previous_conf,
        **selection)


class RealtimeSkeleton3DPipeline:
    """Turns BGR frames into H36M-17 skeletons, one frame at a time.

    The skeleton is root-relative, and additionally world-located when the camera
    has calibrated extrinsics and depth resolves for that frame.

    config: VisionConfig, carrying every YOLO/calibration/MotionBERT/depth knob.
    """

    def __init__(self, config: VisionConfig):
        from ultralytics import YOLO
        from skeleton_utils.motionbert_lifter import (
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
                "calibrated camera -- run setup/calibrate_camera.py (see "
                "eval/pose_detection_live.py's module docstring) first.")
        # image_size is kept, not discarded: K is only valid at the resolution it
        # was solved for, so any caller feeding this pipeline its own frames
        # (setup/calibrate_body.py) needs to know what that resolution is.
        self.K, _dist, self.image_size = load_intrinsics(intrinsics_path)
        fy = float(self.K[1, 1])

        self.T_world_from_camera = None
        # Calibrated floor height. body_calibration measures a metric stature by
        # intersecting the ankle ray with this plane -- the one way to get a height
        # without already knowing one, which the depth estimator cannot do.
        self.ground_z = 0.0
        self.have_extrinsics = extrinsics_path.exists()
        if self.have_extrinsics:
            self.T_world_from_camera, self.ground_z, _robot_base = load_extrinsics(extrinsics_path)
        else:
            logger.warning(
                "No extrinsics at %s -- posture will stay camera-frame (tilted by however "
                "the camera is mounted, no gravity/world reference). Run "
                "setup/calibrate_camera.py's extrinsic step to fix this.", extrinsics_path)

        device = config.device or "cpu"
        self.yolo = YOLO(str(config.yolo_model_path))
        self.lifter = MotionBERTStreamingLifter(
            config_path=config.motionbert_config or DEFAULT_CONFIG,
            checkpoint_path=config.motionbert_checkpoint or DEFAULT_CHECKPOINT,
            clip_len=config.clip_len, device=device, half=config.motionbert_fp16)
        self.kp_filter = KeypointOutlierHoldFilter() if config.use_keypoint_filter else None
        # On by default because training data is bone-filtered too; turning it off feeds
        # the model a noisier skeleton than it was trained on.
        self.bone_filter = (BoneLengthConstraintFilter()
                            if config.use_bone_length_filter else None)

        # Sticky person selection (see run_yolo_2d / skeleton_utils/person_selection.py).
        # The kwargs are fixed at construction; only the previous pose changes per frame.
        self.person_selection = None
        if config.use_person_selection:
            self.person_selection = {
                "conf_threshold": config.conf_threshold,
                "min_valid_joints": config.person_min_valid_joints,
                "min_detection_score": config.person_min_detection_score,
                "max_jump_ratio": config.person_max_jump_ratio,
            }
        # Last ACCEPTED detection -- what the next frame's candidates are matched against.
        # Held (not cleared) on a frame with no detection, so a brief dropout does not
        # reset the identity and let a bystander win the next frame by default.
        self._previous_keypoints = None
        self._previous_conf = None

        self.calibration = None
        if config.body_calibration_file:
            self.calibration = BodyCalibration.load(config.body_calibration_file)
            if self.bone_filter is not None and self.calibration.bone_lengths:
                self.bone_filter.seed(self.calibration.bone_lengths)
                logger.info("Seeded bone lengths from %s (%d bones, %d T-pose frames).",
                            config.body_calibration_file, len(self.calibration.bone_lengths),
                            self.calibration.n_pose_frames)
            stature = self.calibration.effective_stature_m
            if stature:
                logger.info("Using calibrated stature %.3f m from %s (was %.3f m).",
                            stature, self.calibration.stature_source, self.user_height_m)
                self.user_height_m = float(stature)
                # Both present: the tape measurement is being used, and the difference
                # is the ground-plane method's error for this subject. Worth logging --
                # a consistent bias across subjects means NOSE_HEIGHT_RATIO needs
                # retuning for this population (see its comment in body_calibration.py).
                ratio = self.calibration.implied_nose_height_ratio
                if ratio is not None and self.calibration.stature_m:
                    logger.info("  ground-plane estimate was %.3f m (%+.1f%%); this subject's "
                                "true nose-height ratio is %.3f vs the assumed %.2f.",
                                self.calibration.stature_m,
                                100.0 * (self.calibration.stature_m / stature - 1.0),
                                ratio, NOSE_HEIGHT_RATIO)

        self.depth_estimator = MetricDepthEstimator(
            focal_length_y=fy, min_cutoff=config.min_cutoff, beta=config.beta,
            d_cutoff=config.d_cutoff)

    def reset(self) -> None:
        """Clear all per-subject state. Call between people.

        The bone filter's ratio gate defends whatever lengths it has converged on, so
        without this a second subject is forced into the first one's skeleton.
        """
        self.lifter.reset()
        # Forget who we were following, or the new subject is rejected as a "jump".
        self._previous_keypoints = None
        self._previous_conf = None
        if self.kp_filter is not None:
            self.kp_filter = KeypointOutlierHoldFilter()
        if self.bone_filter is not None:
            self.bone_filter.reset()
            if self.calibration is not None and self.calibration.bone_lengths:
                self.bone_filter.seed(self.calibration.bone_lengths)

    def process(self, frame, timestamp: float, K=None) -> dict[str, Any]:
        """Run one frame through the pipeline.

        K: this frame's intrinsics, for sources that report their own (an
            iPhone via Record3D does -- see FrameSource.last_intrinsics).
            None uses the calibrated file K, which is what a webcam needs.
            Both the depth estimate and the pixel->camera back-projection use
            it, since depth is linear in fy and a focus-driven change in fy
            would otherwise rescale every world position silently.

        Returns a dict with:
            keypoints_2d, keypoints_conf, kp_status  YOLO output, post-filter
            root_relative    camera-frame posture, or None if nothing detected
            world_root_xyz   (3,), nan-filled when position is unavailable
            skeleton         world-frame fusion when extrinsics and depth both
                             resolved this frame, else root_relative, else None
        """
        K_frame = self.K if K is None else np.asarray(K, dtype=np.float64)
        h, w = frame.shape[:2]
        keypoints_2d, keypoints_conf = run_yolo_2d(
            self.yolo, frame, imgsz=self.yolo_imgsz,
            selection=self.person_selection,
            previous_keypoints=self._previous_keypoints,
            previous_conf=self._previous_conf)
        # Anchor the next frame on the RAW selected detection, before the keypoint filter
        # holds/smooths it -- matching against filtered output would compare candidates to
        # a pose that was partly invented, and let the anchor drift away from any real person.
        if keypoints_2d is not None:
            self._previous_keypoints = keypoints_2d
            self._previous_conf = keypoints_conf

        kp_status = None
        if self.kp_filter is not None:
            keypoints_2d, keypoints_conf, kp_status = self.kp_filter.filter(keypoints_2d, keypoints_conf)

        out: dict[str, Any] = {
            "keypoints_2d": keypoints_2d,
            "keypoints_conf": keypoints_conf,
            "kp_status": kp_status,
            "root_relative": None,
            "root_relative_world": None,
            "world_root_xyz": np.full(3, np.nan),
            "skeleton": None,
        }
        if keypoints_2d is None or not np.any(keypoints_2d):
            return out

        root_relative = self.lifter.lift(keypoints_2d, image_size=(w, h), keypoints_conf=keypoints_conf)
        if root_relative is None:
            return out
        # Before any world-frame transform, matching training's lift -> filter ->
        # rotate order. Filtering after rotation gives the same lengths but a
        # different arrangement of the filter's accumulated state.
        if self.bone_filter is not None:
            root_relative = self.bone_filter.filter(root_relative)
        out["root_relative"] = root_relative
        out["skeleton"] = root_relative  # fallback if no extrinsics/depth this frame

        pitch_rad = estimate_pitch_from_torso_vector(root_relative)
        masked_kpts = keypoints_2d.astype(np.float64).copy()
        masked_kpts[keypoints_conf < self.conf_threshold] = np.nan
        z_filtered = self.depth_estimator.update(
            track_id=0, keypoints_2d=masked_kpts, user_height_meters=self.user_height_m,
            pitch_angle_rad=pitch_rad, timestamp=timestamp,
            focal_length_y=float(K_frame[1, 1]))

        if z_filtered is not None and self.have_extrinsics:
            pelvis_px = coco_to_h36m_xy(keypoints_2d)[H36M_ROOT]
            root_camera_xyz = tf.pixel_depth_to_camera_point(K_frame, pelvis_px, z_filtered)
            world_root_xyz = tf.camera_point_to_world(self.T_world_from_camera, root_camera_xyz)
            out["world_root_xyz"] = world_root_xyz
            # The same posture in three useful forms. root_relative is camera-frame:
            # its axes are the lens's, so it only means anything alongside the camera's
            # own orientation. root_relative_world is that posture ROTATED into world
            # axes but still pelvis-centred -- which is what a consumer needs if it is
            # going to place the skeleton at world_root_xyz itself, since mixing a
            # world-frame position with a camera-frame posture tilts the body by
            # whatever the extrinsics rotation is. skeleton is both applied at once.
            rotated = tf.transform_directions(self.T_world_from_camera, root_relative)
            out["root_relative_world"] = rotated
            out["skeleton"] = world_root_xyz + rotated

        return out


class StreamingH36MFeatureExtractor:
    """Turns a stream of (17, 3) skeletons into kinematic feature vectors.

    Emits the same panels as h36m_features.compute_all_features -- speeds,
    accelerations, per-axis velocities, pelvis-relative positions, polar angles,
    joint angles, ratios, distance from center -- one frame at a time. Keys and
    column order must match that module exactly, since normalization looks up
    mean/std by these names.

    Three things differ from the offline version it has to agree with:

    **Causal window.** Each call fits over the trailing `window_length` frames
    rather than a centered window, and falls back to the raw last frame with zero
    velocity while the buffer is too short to fit.

    **Real timestamps, not a fixed delta.** The live loop is a fixed-tick
    scheduler, so a slow frame stretches the real gap between samples. Assuming a
    constant delta would mis-scale velocity and acceleration -- an 80 ms gap read
    as the nominal 33 ms. So this fits a polynomial against each frame's actual
    timestamp instead of using savgol_filter, which only takes a fixed delta.

    **Running shoulder-width median.** The ratios panel divides by shoulder width,
    and offline that denominator is the median over a whole clip, not the current
    frame: when a subject turns edge-on the lifter can collapse both shoulders onto
    one point, and a per-frame denominator then spikes the ratios tenfold with the
    arms motionless. A whole-clip median is not available live, so this keeps a
    running one over the last SHOULDER_WIDTH_HISTORY widths. It converges over the
    first seconds of a session rather than being exact from frame one.

    config: VisionConfig; uses feature_window_length and feature_polyorder.
    """

    def __init__(self, config: VisionConfig):
        self.window_length = config.feature_window_length
        self.polyorder = config.feature_polyorder
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=self.window_length)
        self._last_valid_skeleton: np.ndarray | None = None
        self._shoulder_widths: deque[float] = deque(maxlen=SHOULDER_WIDTH_HISTORY)

    def reset(self) -> None:
        self._buffer.clear()
        self._last_valid_skeleton = None
        self._shoulder_widths.clear()

    def update(self, skeleton_17x3: np.ndarray | None, timestamp: float) -> dict[str, np.ndarray]:
        """Add one frame and return its feature vectors.

        skeleton_17x3: (17, 3), or None when nothing was detected -- the last valid
            skeleton is then held, matching the keypoint filter upstream, or zeros
            if none has arrived yet.
        timestamp: seconds, the actual time this skeleton was observed. Used to
            scale velocity and acceleration, so it must be real rather than a
            frame counter (see the class docstring).
        """
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
        ratios = compute_ratios(position, shoulder_width=self._running_shoulder_width(position))
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

    def _running_shoulder_width(self, position: np.ndarray) -> float:
        """Append this frame's shoulder width and return the running median.

        position: (1, 17, 3), the smoothed skeleton -- the same array the ratios are
        computed from, so numerator and denominator stay consistent.

        Only numerically degenerate widths (non-finite, or <=1e-8) are refused, to keep
        a division by zero out of the history. A partial edge-on collapse is still
        appended and is harmless, because it cannot move a median -- which is exactly
        why this is a median and not a mean, where a few near-zero samples would drag
        the denominator down and inflate every later ratio.

        Returns NaN until the first usable sample, making those frames' ratios NaN
        rather than inf.
        """
        width = float(np.linalg.norm(position[0, L_SHOULDER] - position[0, R_SHOULDER]))
        if np.isfinite(width) and width > 1e-8:
            self._shoulder_widths.append(width)
        if not self._shoulder_widths:
            return float("nan")
        return float(np.median(self._shoulder_widths))

    def _fit_derivative(self, times: np.ndarray, window: np.ndarray, deriv: int) -> np.ndarray:
        """The deriv-th derivative at the latest timestamp.

        deriv: 0 = smoothed position, 1 = velocity, 2 = acceleration.
        times: (t,) each buffered frame's real timestamp, not assumed evenly spaced.
        window: (t, 17, 3), same order as times.

        Fits one least-squares polynomial per joint/axis of position against elapsed
        time, then differentiates it analytically. Centering time on the latest frame
        makes the answer just deriv! * the x**deriv coefficient, with no separate
        evaluation step.

        Falls back to the raw last frame (deriv=0) or zero (deriv>0) until the buffer
        holds two more samples than polyorder, so an exact-interpolation polynomial --
        zero residual, wild derivatives -- is never fit.
        """
        t = times.shape[0]
        if t < self.polyorder + 2:
            return window[-1] if deriv == 0 else np.zeros_like(window[-1])
        if deriv > self.polyorder:
            return np.zeros_like(window[-1])

        t_rel = times - times[-1]  # latest frame at x=0
        flat = window.reshape(t, -1)  # (t, 17*3)
        coeffs = np.polyfit(t_rel, flat, self.polyorder)  # (polyorder+1, 17*3), highest power first

        # At x=0 the deriv-th derivative is deriv! * a_deriv, and coeffs runs highest
        # power first, so a_deriv is row (polyorder - deriv).
        coeff_row = coeffs[self.polyorder - deriv]
        derivative_at_latest = math.factorial(deriv) * coeff_row
        return derivative_at_latest.reshape(window.shape[1:])
