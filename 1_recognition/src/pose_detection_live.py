"""Live human posture + world-frame location from a calibrated camera.

Pipeline: YOLO 2D pose -> MotionBERT streaming 2D->3D lift (POSTURE, i.e.
root-relative body shape) + MetricDepthEstimator (LOCATION, i.e. absolute
distance from the camera, from 2D keypoints + known intrinsics + assumed
body height -- see skeleton_pipeline/metric_depth_estimator.py's docstring;
deliberately NOT a heavy monocular depth model like Depth Pro or
yolo26-depth, since this project's calibrated setup doesn't need one).

The camera must already be calibrated with app/calibrate_camera.py
(intrinsic + extrinsic, either --method board or --method marker -- see
that script). Extrinsic calibration is what defines the WORLD ORIGIN this
script reports position in: wherever the ChArUco board/ArUco marker was
physically placed during that calibration run (by default the target's own
origin, i.e. --board-xyz (0,0,0)) -- e.g. if you calibrated against an
ArUco marker taped to the floor, world (0,0,0) is that marker's location,
world +Z is up.

Per frame this combines the two into a single WORLD-FRAME skeleton:
    world_skeleton = world_root_xyz + R_world_from_camera @ root_relative
i.e. MotionBERT's root-relative shape, rotated into gravity-aligned world
axes and translated to the depth-estimator's absolute root position --
"posture and location with respect to the calibration origin" in one array.

Each frame's skeleton is also run through StreamingH36MFeatureExtractor
(skeleton3d_pipeline.py -- the same kinematic-feature extractor
RecognitionManager feeds its LSTM with) to compute causal-smoothed
pol_angles/joint_angles/ratios, shown as an extra overlay label and, if
--output is given, saved to the sibling .npz (keys: world_root_xyz,
world_skeleton, timestamps, pol_angles, joint_angles, joint_angle_keys,
ratios, ratio_keys -- see main()'s end for the full list).

Without calibrated extrinsics (only intrinsics.json), this still runs and
shows camera-relative depth (Z from the lens), but cannot report a WORLD
position -- there is no defined origin without extrinsics.

Usage, webcam:
    uvpython pose_detection_live.py --source 0 --device cuda:0 `
        --user-height-m 1.75 --capture-rotate90 90 --world-view-range 5.0

Usage, iPhone via Record3D (must match the --capture-rotate90 used when
calibrating iphone_intrinsics.json/iphone_extrinsics.json):
    uv run python pose_detection_live.py --source iphone --dev-idx 0 `
        --capture-rotate90 90 --device cuda:0 --world-view-range 5.0 --user-height-m 1.75

Usage, recorded video (for testing without a live camera):
    python pose_detection_live.py --source path/to/clip.mp4 --device cuda:0

Press Q or ESC in the preview window to stop.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 1_recognition/, for skeleton3d_pipeline/vision_model

import cv2
import numpy as np

from camera_utils.calibration_io import load_extrinsics, load_intrinsics
from camera_utils import transforms as tf
from skeleton_pipeline.coco_h36m import coco_to_h36m_xy
from skeleton_pipeline.keypoint_filter import KeypointOutlierHoldFilter
from skeleton_pipeline.metric_depth_estimator import MetricDepthEstimator
from skeleton_pipeline.render.skeleton_video import FastSkeleton3DRenderer, draw_2d_skeleton
from skeleton3d_pipeline import JOINT_ANGLE_KEYS, RATIO_KEYS, StreamingH36MFeatureExtractor
from vision_model.vision_config import VisionConfig

CALIB_DATA_DIR = Path(__file__).resolve().parents[1] / "calib_data"
DEFAULT_YOLO_MODEL = Path(__file__).resolve().parents[1] / "dataset" / "model" / "yolo26m-pose.pt"

# H36M-17 indices (see skeleton_pipeline/coco_h36m.py's docstring for the full order).
H36M_ROOT = 0
H36M_THORAX = 8


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


def estimate_pitch_from_torso_vector(root_relative):
    """Angle (radians, unsigned) between the pelvis->thorax vector and
    vertical (+Z, up) -- 0 = perfectly upright. root_relative[H36M_ROOT] is
    (0,0,0) by construction, so root_relative[H36M_THORAX] IS the
    pelvis->thorax vector already. Fed to MetricDepthEstimator's
    pitch_angle_rad (see its docstring's foreshortening-correction
    section) -- an unsigned magnitude is fine since that correction
    (cos(pitch)) is symmetric for forward/backward lean."""
    torso_vector = root_relative[H36M_THORAX]
    norm = float(np.linalg.norm(torso_vector))
    if norm < 1e-6:
        return 0.0
    cos_pitch = np.clip(torso_vector[2] / norm, -1.0, 1.0)
    return float(np.arccos(cos_pitch))


def draw_bottom_left_labels(img, label_lines, margin=16, line_height=28,
                             font_scale=0.7, thickness=2):
    h = img.shape[0]
    for i, line in enumerate(label_lines):
        y = h - margin - line_height * (len(label_lines) - 1 - i)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)


class WorldTrajectoryRenderer:
    """Top-down (world X-Y, i.e. floor-plan) view of the root's world-frame
    position -- where the person actually is relative to the calibration
    origin (the ChArUco board/ArUco marker's location, world +Z up), not
    their distance from the camera lens (the other panels show that). Also
    draws a fading trail of its last N frames of position.
    """

    def __init__(self, panel_size, view_range_m=3.0, camera_xy_world=None):
        self.panel_size = panel_size
        self.view_range_m = view_range_m
        self.camera_xy_world = camera_xy_world

    def _to_px(self, xy):
        w, h = self.panel_size
        scale = min(w, h) / (2.0 * self.view_range_m)
        # World +X -> screen right, world +Y -> screen up (image rows grow
        # downward, so world +Y needs the sign flip).
        px = w / 2 + xy[0] * scale
        py = h / 2 - xy[1] * scale
        return int(round(px)), int(round(py))

    def render(self, trajectory_xy, current_xy=None):
        w, h = self.panel_size
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        step = max(1, int(round(self.view_range_m / 3)))
        for m in range(-int(self.view_range_m), int(self.view_range_m) + 1, step):
            gx, _ = self._to_px((m, 0))
            _, gy = self._to_px((0, m))
            cv2.line(img, (gx, 0), (gx, h), (230, 230, 230), 1, cv2.LINE_AA)
            cv2.line(img, (0, gy), (w, gy), (230, 230, 230), 1, cv2.LINE_AA)

        ox, oy = self._to_px((0, 0))
        cv2.line(img, (0, oy), (w, oy), (195, 195, 195), 1, cv2.LINE_AA)
        cv2.line(img, (ox, 0), (ox, h), (195, 195, 195), 1, cv2.LINE_AA)
        cv2.circle(img, (ox, oy), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img, "origin", (ox + 8, oy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

        if self.camera_xy_world is not None:
            camx, camy = self._to_px(self.camera_xy_world)
            cv2.drawMarker(img, (camx, camy), (150, 0, 0), cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
            cv2.putText(img, "camera", (camx + 8, camy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 0, 0), 1, cv2.LINE_AA)

        n = len(trajectory_xy)
        for i in range(1, n):
            frac = i / max(n - 1, 1)
            color = (int(220 - 100 * frac), int(200 - 80 * frac), int(220 + 30 * frac))
            cv2.line(img, self._to_px(trajectory_xy[i - 1]), self._to_px(trajectory_xy[i]),
                     color, 2, cv2.LINE_AA)

        if current_xy is not None:
            cx_px, cy_px = self._to_px(current_xy)
            cv2.circle(img, (cx_px, cy_px), 7, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cx_px, cy_px), 7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"({current_xy[0]:.2f},{current_xy[1]:.2f})m", (cx_px + 10, cy_px + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(img, f"world XY (top-down), +-{self.view_range_m:.1f}m, trail={n}f",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return img


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, default="0",
                         help="Camera index (e.g. '0'), a video file path, or 'iphone' to stream "
                              "from an iPhone via Record3D.")
    parser.add_argument("--dev-idx", type=int, default=0,
                         help="Record3D device index, only used when --source iphone.")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="Only used when --source iphone. Must match the --capture-rotate90 "
                              "used when calibrating iphone_intrinsics.json/iphone_extrinsics.json, "
                              "or K and the world frame will be wrong for these frames.")
    parser.add_argument("--calib-dir", type=str, default=str(CALIB_DATA_DIR),
                         help=f"Directory with intrinsics/extrinsics JSON (default: {CALIB_DATA_DIR}, "
                              "see app/calibrate_camera.py).")
    parser.add_argument("--intrinsics-file", type=str, default=None,
                         help="Default: 'iphone_intrinsics.json' for --source iphone, else "
                              "'intrinsics.json'.")
    parser.add_argument("--extrinsics-file", type=str, default=None,
                         help="Default: 'iphone_extrinsics.json' for --source iphone, else "
                              "'extrinsics.json'. Defines the world origin (see module docstring); "
                              "if missing, this still runs with camera-relative depth only.")
    parser.add_argument("--yolo-model", type=str, default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--device", type=str, default="cpu",
                         help="'cpu', 'cuda:0', etc. -- passed to both YOLO and MotionBERT.")
    parser.add_argument("--motionbert-config", type=str, default=None,
                         help="Default: skeleton_pipeline/motionbert_lifter.py's DEFAULT_CONFIG.")
    parser.add_argument("--motionbert-checkpoint", type=str, default=None,
                         help="Default: skeleton_pipeline/motionbert_lifter.py's DEFAULT_CHECKPOINT "
                              "-- must be downloaded manually first, see that module's docstring.")
    parser.add_argument("--clip-len", type=int, default=81,
                         help="MotionBERT rolling-buffer window length. Default 81 to match this "
                              "project's live/deployment window (see generate_lstm_training_data.py's "
                              "--clip-len docstring for why NOT the higher-quality 243).")
    parser.add_argument("--user-height-m", type=float, default=1.70,
                         help="Assumed real height of the person in frame -- MetricDepthEstimator's "
                              "accuracy scales directly with how correct this is.")
    parser.add_argument("--min-cutoff", type=float, default=1.0,
                         help="OneEuroFilter param (depth smoothing) -- lower = smoother but laggier.")
    parser.add_argument("--beta", type=float, default=0.007,
                         help="OneEuroFilter param -- higher = reacts faster to real fast motion, "
                              "at the cost of passing through more noise while moving.")
    parser.add_argument("--d-cutoff", type=float, default=1.0, help="OneEuroFilter param.")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                         help="YOLO keypoint confidence below this is treated as not visible.")
    parser.add_argument("--no-keypoint-filter", action="store_true",
                         help="Disable KeypointOutlierHoldFilter and send YOLO's raw output "
                              "straight to the lifter/depth estimator.")
    parser.add_argument("--feature-window-length", type=int, default=9,
                         help="StreamingH36MFeatureExtractor's causal Savitzky-Golay smoothing "
                              "window (frames) -- see skeleton_pipeline/features/h36m_features.py's "
                              "'Why Savitzky-Golay' docstring.")
    parser.add_argument("--feature-polyorder", type=int, default=3,
                         help="StreamingH36MFeatureExtractor's Savitzky-Golay polynomial order.")
    parser.add_argument("--trajectory-frames", type=int, default=200,
                         help="How many past frames of world-XY position to show as a fading "
                              "trail in the world-trajectory panel.")
    parser.add_argument("--world-view-range", type=float, default=3.0,
                         help="Half-width/height in meters of the world-trajectory panel's view.")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(480, 480))
    parser.add_argument("--output", type=str, default=None,
                         help="If given, writes a 3-panel (2D overlay | 3D posture | world "
                              "trajectory) video here, plus a sibling .npz with per-frame "
                              "world_root_xyz/world_skeleton/timestamps. Not saved by default.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Hard cap on frames processed (omit to run until Q/ESC/stream end).")
    parser.add_argument("--no-preview", action="store_true",
                         help="Disable the live cv2.imshow preview window -- the only way to watch "
                              "a live run in real time; also the only way to stop it early (Q/ESC) "
                              "when enabled.")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    from skeleton_pipeline.motionbert_lifter import (
        DEFAULT_CHECKPOINT, DEFAULT_CONFIG, MotionBERTStreamingLifter,
    )

    is_iphone = args.source == "iphone"
    calib_dir = Path(args.calib_dir)
    intrinsics_file = args.intrinsics_file or ("iphone_intrinsics.json" if is_iphone else "intrinsics.json")
    extrinsics_file = args.extrinsics_file or ("iphone_extrinsics.json" if is_iphone else "extrinsics.json")
    intrinsics_path = calib_dir / intrinsics_file
    extrinsics_path = calib_dir / extrinsics_file

    if not intrinsics_path.exists():
        raise FileNotFoundError(
            f"No intrinsics found at {intrinsics_path}. Run app/calibrate_camera.py "
            f"({'iphone-intrinsic' if is_iphone else 'intrinsic'}) first.")
    K, _dist, _image_size = load_intrinsics(intrinsics_path)
    fy = float(K[1, 1])

    T_world_from_camera = None
    have_extrinsics = extrinsics_path.exists()
    if have_extrinsics:
        T_world_from_camera, _ground_z, _robot_base = load_extrinsics(extrinsics_path)
        camera_up_world = T_world_from_camera[:3, :3] @ np.array([0.0, -1.0, 0.0])
        tilt_deg = np.degrees(np.arccos(np.clip(camera_up_world[2], -1.0, 1.0)))
        print(f"Loaded extrinsics from {extrinsics_path} -- world origin is the calibration "
              f"target's (ChArUco board / ArUco marker) location. Camera is tilted "
              f"~{tilt_deg:.1f}deg from vertical per this calibration; if that doesn't match "
              f"how the camera is physically mounted right now, re-run extrinsic calibration.")
    else:
        print(f"NOTE: no extrinsics found at {extrinsics_path} -- only camera-relative depth "
              f"will be available (no defined world origin). Run app/calibrate_camera.py "
              f"({'iphone-extrinsic' if is_iphone else 'extrinsic'}) first to get world-frame "
              f"posture/location.")

    if is_iphone:
        from camera_utils.iphone_connection import IPhoneVideoCaptureAdapter
        cap = IPhoneVideoCaptureAdapter(dev_idx=args.dev_idx, capture_rotate90=args.capture_rotate90)
        fps = 30.0
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Could not open video/camera: {args.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Loading YOLO ({args.yolo_model}) on {args.device}...")
    yolo = YOLO(args.yolo_model)

    config_path = args.motionbert_config or DEFAULT_CONFIG
    checkpoint_path = args.motionbert_checkpoint or DEFAULT_CHECKPOINT
    print(f"Loading MotionBERT (config={config_path}, checkpoint={checkpoint_path}, "
          f"clip_len={args.clip_len}) on {args.device}...")
    lifter = MotionBERTStreamingLifter(
        config_path=config_path, checkpoint_path=checkpoint_path,
        clip_len=args.clip_len, device=args.device)

    kp_filter = None if args.no_keypoint_filter else KeypointOutlierHoldFilter()
    depth_estimator = MetricDepthEstimator(
        focal_length_y=fy, min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff)
    # Same streaming H36M kinematic-feature extractor RecognitionManager feeds the LSTM with
    # (see skeleton3d_pipeline.py's StreamingH36MFeatureExtractor / recognition_manager.py's
    # update_from_frame) -- run here too so this preview shows what downstream recognition sees.
    feature_extractor = StreamingH36MFeatureExtractor(VisionConfig(
        fps=fps, feature_window_length=args.feature_window_length,
        feature_polyorder=args.feature_polyorder))
    renderer_3d = FastSkeleton3DRenderer(args.panel_size)
    world_renderer = None
    if have_extrinsics:
        world_renderer = WorldTrajectoryRenderer(
            args.panel_size, view_range_m=args.world_view_range,
            camera_xy_world=tuple(T_world_from_camera[:2, 3]))
    trajectory = deque(maxlen=args.trajectory_frames)

    writer = None
    npz_path = None
    all_world_root, all_world_skeleton, all_timestamps = [], [], []
    all_pol_angles, all_joint_angles, all_ratios = [], [], []
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = output_path.with_suffix(".npz")

    panel_w, panel_h = args.panel_size
    frame_idx = 0
    t0 = time.time()
    print("Running -- press Q/ESC in the preview window to stop." if not args.no_preview
          else "Running (no preview) -- Ctrl+C to stop, or set --max-frames.")

    try:
        while args.max_frames is None or frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            timestamp = frame_idx / fps

            keypoints_2d, keypoints_conf = run_yolo_2d(yolo, frame)
            kp_status = None
            if kp_filter is not None:
                keypoints_2d, keypoints_conf, kp_status = kp_filter.filter(keypoints_2d, keypoints_conf)

            overlay = frame
            world_root_xyz = np.full(3, np.nan)
            world_skeleton = np.full((17, 3), np.nan)
            skeleton_for_render = None
            label_lines = []

            if kp_status is not None and kp_status != "accepted":
                label_lines.append(f"keypoint filter: {kp_status}")

            if keypoints_2d is not None and np.any(keypoints_2d):
                overlay = draw_2d_skeleton(frame, keypoints_2d, keypoints_conf,
                                            conf_threshold=args.conf_threshold)

                root_relative = lifter.lift(keypoints_2d, image_size=(w, h),
                                             keypoints_conf=keypoints_conf)
                pitch_rad = 0.0
                if root_relative is not None:
                    pitch_rad = estimate_pitch_from_torso_vector(root_relative)
                    skeleton_for_render = root_relative

                masked_kpts = keypoints_2d.astype(np.float64).copy()
                masked_kpts[keypoints_conf < args.conf_threshold] = np.nan

                z_filtered = depth_estimator.update(
                    track_id=0, keypoints_2d=masked_kpts, user_height_meters=args.user_height_m,
                    pitch_angle_rad=pitch_rad, timestamp=timestamp)

                if z_filtered is not None:
                    pelvis_px = coco_to_h36m_xy(keypoints_2d)[H36M_ROOT]
                    cv2.circle(overlay, (int(pelvis_px[0]), int(pelvis_px[1])), 6,
                               (0, 255, 255), 2, cv2.LINE_AA)
                    label_lines.append(f"Z(camera, filtered)={z_filtered:.2f}m")

                    if have_extrinsics:
                        root_camera_xyz = tf.pixel_depth_to_camera_point(K, pelvis_px, z_filtered)
                        world_root_xyz = tf.camera_point_to_world(T_world_from_camera, root_camera_xyz)
                        trajectory.append((world_root_xyz[0], world_root_xyz[1]))
                        label_lines.append(
                            f"World XYZ=({world_root_xyz[0]:.2f},{world_root_xyz[1]:.2f},"
                            f"{world_root_xyz[2]:.2f})m  (origin = calibration target)")
                        if root_relative is not None:
                            # POSTURE (gravity-aligned shape) + LOCATION (absolute root
                            # position) fused into one world-frame skeleton -- see module
                            # docstring's world_skeleton formula.
                            world_skeleton = world_root_xyz + tf.transform_directions(
                                T_world_from_camera, root_relative)
                            skeleton_for_render = world_skeleton  # gravity-aligned for the 3D panel
                    else:
                        label_lines.append("World XYZ=n/a (no extrinsics -- see module docstring)")

            # Streaming H36M kinematic features (pol_angles/joint_angles/ratios) from
            # skeleton_for_render -- same input RecognitionManager feeds its LSTM with (world-frame
            # fusion if available this frame, else camera-frame root_relative, else None/hold-last;
            # see skeleton3d_pipeline.py's StreamingH36MFeatureExtractor docstring).
            features = feature_extractor.update(skeleton_for_render)
            if skeleton_for_render is not None:
                label_lines.append(
                    f"elbow L/R={features['joint_angles'][JOINT_ANGLE_KEYS.index('left_elbow_angle_deg')]:.0f}/"
                    f"{features['joint_angles'][JOINT_ANGLE_KEYS.index('right_elbow_angle_deg')]:.0f}deg  "
                    f"wrist/shoulder={features['ratios'][RATIO_KEYS.index('wrist_over_shoulder_ratio')]:.2f}")

            draw_bottom_left_labels(overlay, label_lines)

            panel_3d = renderer_3d.render(skeleton_for_render)
            if world_renderer is not None:
                current_xy = ((world_root_xyz[0], world_root_xyz[1])
                              if not np.isnan(world_root_xyz).any() else None)
                world_vis = world_renderer.render(list(trajectory), current_xy=current_xy)
            else:
                world_vis = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
                cv2.putText(world_vis, "no calibrated extrinsics", (10, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
                cv2.putText(world_vis, "(see --calib-dir)", (10, panel_h // 2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)

            combined = np.hstack([cv2.resize(overlay, (panel_w, panel_h)),
                                  cv2.resize(panel_3d, (panel_w, panel_h)),
                                  cv2.resize(world_vis, (panel_w, panel_h))])

            if args.output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.output), fourcc, fps,
                                              (combined.shape[1], combined.shape[0]))
                writer.write(combined)
                all_world_root.append(world_root_xyz)
                all_world_skeleton.append(world_skeleton)
                all_timestamps.append(timestamp)
                all_pol_angles.append(features["pol_angles"])
                all_joint_angles.append(features["joint_angles"])
                all_ratios.append(features["ratios"])

            if not args.no_preview:
                cv2.imshow("pose_detection_live - Q/ESC to quit", combined)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("  stopped by user (Q/ESC).")
                    break

            frame_idx += 1
            if frame_idx % 60 == 0:
                elapsed = time.time() - t0
                print(f"  frame {frame_idx}  ({frame_idx / elapsed:.1f} fps)"
                      + (f"  world=({world_root_xyz[0]:.2f},{world_root_xyz[1]:.2f},"
                         f"{world_root_xyz[2]:.2f})m" if not np.isnan(world_root_xyz).any() else ""))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\nDone: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps).")

    if npz_path is not None and all_timestamps:
        world_root_arr = np.array(all_world_root)
        world_skeleton_arr = np.array(all_world_skeleton)
        np.savez_compressed(
            npz_path,
            world_root_xyz=world_root_arr,
            world_skeleton=world_skeleton_arr,
            timestamps=np.array(all_timestamps),
            fps=fps,
            have_extrinsics=have_extrinsics,
            pol_angles=np.array(all_pol_angles),
            joint_angles=np.array(all_joint_angles),
            joint_angle_keys=np.array(JOINT_ANGLE_KEYS),
            ratios=np.array(all_ratios),
            ratio_keys=np.array(RATIO_KEYS),
        )
        valid = ~np.isnan(world_root_arr).any(axis=1)
        if valid.any():
            mean_xyz = np.nanmean(world_root_arr, axis=0)
            print(f"World root position: mean=({mean_xyz[0]:.2f},{mean_xyz[1]:.2f},{mean_xyz[2]:.2f})m "
                  f"(n={int(valid.sum())}/{len(all_timestamps)} frames, origin = calibration target)")
        print(f"Saved: {args.output}\nSaved: {npz_path}")


if __name__ == "__main__":
    main()
