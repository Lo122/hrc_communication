"""Recognition pipeline integration."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
import logging

import numpy as np

import config
from models import RecognitionResult
from vision_model.vision_config import VisionConfig

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "best_model" / "3d_skeleton" 
logger = logging.getLogger(__name__)

# Feature keys a model config may ask for that StreamingH36MFeatureExtractor doesn't emit
# under that name, because they are a concatenation of several of its outputs. "pol_angles"
# is azimuth followed by elevation -- the same order src/pose_detection_live.py builds it in
# (see its all_pol_angles). Each component is normalized separately (the norm .npz carries
# per-component stats, e.g. polar_azimuth_mean/polar_elevation_mean) and only then
# concatenated here, so this stays consistent with training.
COMPOSITE_FEATURE_KEYS = {"pol_angles": ("polar_azimuth", "polar_elevation")}

STEP_SMOOTHING_WINDOW = 5
STEP_CONFIRMATION_COUNT = 3
STEP_MIN_CONFIDENCE = 0.6
STEP_MIN_MARGIN = 0.15

# video_source auto-detection (RecognitionManager._classify_video_source) -- recorded-file
# extensions vs. live-stream URL schemes, see that method's docstring.
_VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".wmv", ".mpg", ".mpeg"}
_STREAM_URL_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class RecognitionManager:
    """Converts passthrough data or camera frames into a RecognitionResult."""

    def __init__(
        self,
        step_stabilizer=None,
        *,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        model_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        norm_path: str | Path | None = None,
        video_source: str | int | None = None,
        feature_keys: list[str] | None = None,
        device: str | None = None,
        show_video: bool = False,
        display_window_name: str = "HRC Recognition",
        display_panel_size: tuple[int, int] = (480, 480),
        plot_window_name: str = "HRC Debug Plot",
        plot_panel_size: tuple[int, int] = (480, 320),
        plot_history_len: int = 300,
        enable_step_model: bool = True,
        vision_config: VisionConfig | None = None,
    ):
        self.step_stabilizer = step_stabilizer
        self.model_dir = Path(model_dir)
        self.enable_step_model = enable_step_model

        if enable_step_model:
            self.model_config_path = Path(model_config_path) if model_config_path is not None else self.model_dir / "config.json"
            self.model_config = self._load_model_config()
            self.model_path = Path(model_path) if model_path is not None else self._find_model_path()
            self.norm_path = Path(norm_path) if norm_path is not None else self._find_norm_path()
            self.feature_keys = feature_keys or list(self.model_config["feature_keys"])
        else:
            # Vision-only mode: YOLO 2D pose + MotionBERT 3D lift still run for real, but the
            # LSTM step classifier and its normalization stats are never touched -- so a
            # missing or mismatched norm .npz cannot stop the skeleton pipeline from running.
            self.model_config_path = None
            self.model_config = None
            self.model_path = None
            self.norm_path = None
            self.feature_keys = list(feature_keys or [])

        self.device_name = device
        self.show_video = show_video
        self.display_window_name = display_window_name
        self.display_panel_size = display_panel_size
        self.plot_window_name = plot_window_name
        self.plot_panel_size = plot_panel_size

        # Debug-overlay history for the live plot window (_update_debug_plot) -- progress/
        # confidence only accumulate once the LSTM window buffer is full (see update_from_frame),
        # world position accumulates every frame with a valid 3D lift, so these are independent
        # lengths/timelines by design; each is just plotted against its own sample index.
        self._progress_history: deque[float] = deque(maxlen=plot_history_len)
        self._confidence_history: deque[float] = deque(maxlen=plot_history_len)
        self._world_xyz_history: deque[tuple[float, float, float]] = deque(maxlen=plot_history_len)

        # 3D posture pipeline (YOLO/calibration/MotionBERT/depth-estimator/feature-window
        # knobs) -- see vision_model/vision_config.py and skeleton3d_pipeline.py. video_source/
        # live are convenience kwargs that, if given, override vision_config.camera's.
        self.vision_config = vision_config or VisionConfig()
        if video_source is not None:
            self.vision_config.camera.video_source = video_source

        self.window_size: int | None = None
        self.num_steps: int | None = None
        self.buffer = deque()

        self._torch = None
        self._cv2 = None
        self._pipeline_ready = False
        self._model = None
        self._norm_real_time = None
        self._skeleton_pipeline = None
        self._feature_extractor = None
        self._draw_2d_skeleton = None
        self._renderer_3d = None
        self._capture = None
        self._camera_K = None
        self._camera_dist = None
        self._camera_image_size = None
        self.T_world_from_camera = None
        self.have_extrinsics = False

        # Latest world-frame human position (see skeleton3d_pipeline.py's world_root_xyz),
        # updated every frame in update_from_frame -- independent of RecognitionResult,
        # which is only returned on confirmed step transitions (too sparse for a live
        # location feed). Holds the last valid value when a frame has no detection/no
        # extrinsics rather than clearing it; stays None until the first valid frame.
        self.last_world_xyz: tuple[float, float, float] | None = None
        self.last_location_timestamp: float | None = None

        # Pelvis-relative (pelvis forced to (0,0,0)) H36M-17 skeleton from the most recent
        # frame with a valid 3D lift -- posture only, NOT fused with world position (see
        # skeleton3d_pipeline.py's RealtimeSkeleton3DPipeline.process() docstring's
        # root_relative). Same "hold last valid value" policy as last_world_xyz. Joint names
        # (self._h36m_joint_names) are resolved lazily in _ensure_realtime_pipeline, since
        # they live in skeleton_pipeline/features/h36m_features.py under src/, which is only
        # added to sys.path once the realtime pipeline is set up.
        self.last_root_relative_skeleton: np.ndarray | None = None
        self._h36m_joint_names: list[str] | None = None

        self.round_id = 0
        self.piece_id = 0
        self.required_steps_per_round = self._load_required_steps_per_round()
        self.seen_trigger_steps_in_round: set[int] = set()
        self._last_recorded_step_id: int | None = None
        self.last_raw_step_id = None

    def update(self, input_data=None) -> RecognitionResult | None:
        """Return the latest standardized recognition result."""
        if input_data is None:
            frame = self._read_frame()
            return self.update_from_frame(frame) if frame is not None else None

        if self._looks_like_frame(input_data):
            return self.update_from_frame(input_data)

        if isinstance(input_data, dict) and "frame" in input_data:
            return self.update_from_frame(
                input_data["frame"],
                timestamp=input_data.get("timestamp"),
            )

        if isinstance(input_data, dict):
            return self._result_from_passthrough(input_data)

        raise TypeError("RecognitionManager.update expects None, a frame, or a dict input.")

    def update_from_frame(
        self,
        frame,
        *,
        round_id: int | None = None,
        piece_id: int | None = None,
        timestamp: float | None = None,
    ) -> RecognitionResult | None:
        """Run YOLO 2D pose -> MotionBERT 3D lift -> streaming H36M
        kinematic features -> LSTM inference for one frame. See
        skeleton3d_pipeline.py (mirrors src/pose_detection_live.py) for the
        posture pipeline this replaces the old 2D one with. Deliberately
        does NOT resize/downscale the frame the way the old 2D pipeline
        did: the calibrated intrinsics (K) and MetricDepthEstimator's
        world-position math assume pixel coordinates at the resolution
        calibration was run at -- resizing YOLO's speed knob instead
        (see RealtimeSkeleton3DPipeline's yolo_imgsz)."""
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Realtime recognition requires opencv-python.") from exc
            self._cv2 = cv2

        self._ensure_realtime_pipeline()

        torch = self._torch
        frame_timestamp = time.time() if timestamp is None else float(timestamp)

        pipeline_out = self._skeleton_pipeline.process(frame, frame_timestamp)

        world_xyz = pipeline_out.get("world_root_xyz")
        if world_xyz is not None and not np.isnan(world_xyz).any():
            self.last_world_xyz = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
            self.last_location_timestamp = frame_timestamp
            self._world_xyz_history.append(self.last_world_xyz)
        logger.info(f"Frame {frame_timestamp:.3f}: world_root_xyz={self.last_world_xyz}")

        root_relative = pipeline_out.get("root_relative")
        if root_relative is not None and not np.isnan(root_relative).any():
            self.last_root_relative_skeleton = root_relative

        if not self.enable_step_model:
            # Vision-only ("fake recognition"): skeleton, 3D lift and world position above are
            # all real and keep feeding the display/HUMAN_LOCATION_UPDATE stream. Step
            # classification is simply not run, so there is no RecognitionResult to return --
            # trigger events come from the manual/keyboard source in run_recognition.py instead.
            self._show_frame(frame, pipeline_out)
            return None

        features = self._feature_extractor.update(pipeline_out["skeleton"], frame_timestamp)
        features = self._norm_real_time.normalize_features(features)
        feature_vector = self._build_feature_vector(features)

        self.buffer.append(feature_vector)
        if len(self.buffer) < self.window_size:
            self._show_frame(frame, pipeline_out)
            return None

        x = np.stack(self.buffer).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            step_logits, progress_pred = self._model(x)
            step_probs = torch.softmax(step_logits, dim=1)
            raw_step_id = int(torch.argmax(step_probs, dim=1).item())
            confidence = float(torch.max(step_probs, dim=1).values.item())
            progress = float(progress_pred.item())

        probabilities = step_probs.squeeze(0).cpu().numpy()
        stable_step_id = self._stable_step_id(probabilities)
        self.last_raw_step_id = raw_step_id
        self._progress_history.append(progress)
        self._confidence_history.append(confidence)

        if stable_step_id is None:
            self._show_frame(frame, pipeline_out, raw_step_id=raw_step_id, progress=progress, confidence=confidence)
            return None

        self._record_step_and_advance_round(stable_step_id)
        result = RecognitionResult(
            round_id=self.round_id,
            step_id=int(stable_step_id),
            progress=progress,
            piece_id=self.piece_id,
            confidence=confidence,
            timestamp=frame_timestamp,
        )
        print(f"Raw Step: {raw_step_id} | Stable Step: {stable_step_id} | Progress: {progress}")
        self._show_frame(frame, pipeline_out, raw_step_id=raw_step_id, stable_step_id=stable_step_id, progress=progress, confidence=confidence)

        return result

    def get_last_keypoints(self) -> dict[str, dict[str, float]] | None:
        """Pelvis-relative H36M-17 skeleton keypoints from the most recent
        frame with a valid 3D lift, as {joint_name: {"x", "y", "z"}} --
        pelvis is included and is always exactly (0,0,0) by construction
        (see last_root_relative_skeleton). None until the first valid
        frame, or before the realtime pipeline has been set up (joint
        names are only resolved in _ensure_realtime_pipeline)."""
        if self.last_root_relative_skeleton is None or self._h36m_joint_names is None:
            return None
        return {
            name: {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
            for name, xyz in zip(self._h36m_joint_names, self.last_root_relative_skeleton)
        }

    def set_video_source(self, video_source: str | int | None, *, live: bool | None = None) -> None:
        """Set or replace the source used when update(None) is called.
        live=None (default) auto-detects whether video_source is a live
        feed (webcam index, stream URL, or "iphone" for a Record3D-
        connected iPhone -- see camera_utils/iphone_connection.py) vs. a
        recorded video file -- see _classify_video_source(). Pass True/
        False to override the auto-detection for an ambiguous source."""
        self.release()
        self.vision_config.camera.video_source = video_source
        self.vision_config.camera.live = live

    def release(self) -> None:
        """Release any owned OpenCV capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self.show_video and self._cv2 is not None:
            for window_name in (self.display_window_name, self.plot_window_name):
                try:
                    self._cv2.destroyWindow(window_name)
                except self._cv2.error:
                    pass

    def _show_frame(
        self,
        frame,
        pipeline_out: dict[str, Any] | None = None,
        *,
        raw_step_id: int | None = None,
        stable_step_id: int | None = None,
        progress: float | None = None,
        confidence: float | None = None,
    ) -> None:
        if not self.show_video:
            return

        overlay = frame
        skeleton = None
        if pipeline_out is not None:
            if pipeline_out.get("keypoints_2d") is not None:
                overlay = self._draw_2d_skeleton(
                    frame, pipeline_out["keypoints_2d"], pipeline_out["keypoints_conf"],
                    conf_threshold=self.vision_config.conf_threshold)
            skeleton = pipeline_out.get("skeleton")

        panel_w, panel_h = self.display_panel_size
        display = overlay
        if self._renderer_3d is not None:
            # Side-by-side: 2D overlay | 3D posture (oblique/front/side/top),
            # same layout as src/pose_detection_live.py's preview window.
            panel_3d = self._renderer_3d.render(skeleton)
            display = self._cv2.hconcat([
                self._cv2.resize(overlay, (panel_w, panel_h)),
                self._cv2.resize(panel_3d, (panel_w, panel_h)),
            ])

        display = self._draw_debug_overlay(
            display, raw_step_id=raw_step_id, stable_step_id=stable_step_id,
            progress=progress, confidence=confidence)

        self._cv2.imshow(self.display_window_name, display)
        self._update_debug_plot()
        if self._cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def _draw_debug_overlay(
        self,
        display,
        *,
        raw_step_id: int | None,
        stable_step_id: int | None,
        progress: float | None,
        confidence: float | None,
    ):
        """Burn live model output + absolute human position as text onto the
        top-left corner of the display frame, for debugging without needing
        to correlate a separate console/log stream against the video."""
        display = display.copy()

        if self.last_world_xyz is not None:
            x, y, z = self.last_world_xyz
            world_line = f"World XYZ: ({x:+.2f}, {y:+.2f}, {z:+.2f}) m"
        else:
            world_line = "World XYZ: --"

        if raw_step_id is not None:
            stable_text = str(stable_step_id) if stable_step_id is not None else "-"
            step_lines = [
                f"Raw step: {raw_step_id}  Stable step: {stable_text}",
                f"Progress: {progress:.2f}  Confidence: {confidence:.2f}",
            ]
        elif not self.enable_step_model:
            step_lines = ["Step model: OFF (manual triggers)"]
        else:
            buffered = len(self.buffer)
            window = self.window_size or "?"
            step_lines = [f"Buffering: {buffered}/{window}"]

        lines = [world_line, *step_lines]
        for i, text in enumerate(lines):
            origin = (10, 24 + i * 22)
            # Black outline then colored fill so the text stays legible over
            # any background (skeleton overlay, bright frame, etc.).
            self._cv2.putText(display, text, origin, self._cv2.FONT_HERSHEY_SIMPLEX,
                               0.55, (0, 0, 0), 3, self._cv2.LINE_AA)
            self._cv2.putText(display, text, origin, self._cv2.FONT_HERSHEY_SIMPLEX,
                               0.55, (0, 255, 0), 1, self._cv2.LINE_AA)
        return display

    def _update_debug_plot(self) -> None:
        """Scrolling time-series window (separate from the skeleton/overlay
        window) of step progress/confidence and world x/y/z -- the trend
        over time that a single-frame text overlay can't show."""
        panel_w, panel_h = self.plot_panel_size
        canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
        top_h = panel_h // 2

        self._draw_series_plot(
            canvas, row_range=(0, top_h), y_range=(0.0, 1.0), title="Progress / Confidence",
            series=[
                (list(self._progress_history), (0, 150, 0), "progress"),
                (list(self._confidence_history), (200, 0, 0), "confidence"),
            ],
        )
        world = list(self._world_xyz_history)
        self._draw_series_plot(
            canvas, row_range=(top_h, panel_h), y_range=None, title="World position (m)",
            series=[
                ([p[0] for p in world], (255, 0, 0), "x"),
                ([p[1] for p in world], (0, 150, 150), "y"),
                ([p[2] for p in world], (0, 0, 255), "z"),
            ],
        )
        self._cv2.imshow(self.plot_window_name, canvas)

    def _draw_series_plot(
        self,
        canvas: np.ndarray,
        *,
        row_range: tuple[int, int],
        y_range: tuple[float, float] | None,
        title: str,
        series: list[tuple[list[float], tuple[int, int, int], str]],
    ) -> None:
        """Draw one or more scrolling line series into canvas[row0:row1, :].
        y_range=None auto-scales to the min/max across all series (with a
        small margin), falling back to (-1, 1) if nothing has data yet.
        Newest sample is at the right edge, oldest scrolls off the left, in
        deque-maxlen-relative x -- so the plot doesn't jump width as history
        fills up."""
        row0, row1 = row_range
        height = row1 - row0
        width = canvas.shape[1]
        maxlen = self._progress_history.maxlen or 1

        if y_range is None:
            values = [v for values, _color, _label in series for v in values]
            if values:
                lo, hi = min(values), max(values)
                margin = max((hi - lo) * 0.1, 0.05)
                y_range = (lo - margin, hi + margin)
            else:
                y_range = (-1.0, 1.0)
        y_lo, y_hi = y_range
        y_span = (y_hi - y_lo) or 1.0

        def to_point(index: int, value: float, n: int) -> tuple[int, int]:
            x = int(round((width - 1) * (maxlen - n + index) / max(maxlen - 1, 1)))
            y = row0 + int(round((1.0 - (value - y_lo) / y_span) * (height - 1)))
            return x, y

        for values, color, _label in series:
            n = len(values)
            if n < 2:
                continue
            points = [to_point(i, v, n) for i, v in enumerate(values)]
            self._cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 1, self._cv2.LINE_AA)

        legend = f"{title}  [" + ", ".join(label for _v, _c, label in series) + f"]  y:[{y_lo:.2f},{y_hi:.2f}]"
        self._cv2.putText(canvas, legend, (6, row0 + 14), self._cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (40, 40, 40), 1, self._cv2.LINE_AA)
        if row0 > 0:
            self._cv2.line(canvas, (0, row0), (width, row0), (210, 210, 210), 1)

    def _result_from_passthrough(self, input_data: dict) -> RecognitionResult:
        step_id = input_data.get("step_id", 0)
        if self.step_stabilizer is not None and "step_probabilities" in input_data:
            step_id = self.step_stabilizer.update(input_data["step_probabilities"])

        self._record_step_and_advance_round(step_id)
        result = RecognitionResult(
            round_id=self.round_id,
            step_id=step_id,
            progress=input_data.get("progress", 0.0),
            piece_id=self.piece_id,
            confidence=input_data.get("confidence", 0.0),
            timestamp=input_data.get("timestamp", time.time()),
        )
        return result

    def _load_required_steps_per_round(self) -> set[int]:
        return {int(step_id) for step_id in config.TRIGGER_RULES}

    def _record_step_and_advance_round(self, step_id) -> None:
        if step_id is None:
            return

        step_id = int(step_id)
        if step_id == self._last_recorded_step_id:
            return

        # Keep all frames of the final step in the same round. Advance only
        # when the next cable-pulling step begins, before building its result.
        if step_id == config.HUMAN_PULL_CABLES and self.required_steps_per_round.issubset(self.seen_trigger_steps_in_round):
            self.round_id += 1
            self.piece_id = self.round_id
            self.seen_trigger_steps_in_round.clear()
        self._last_recorded_step_id = step_id
        if step_id in self.required_steps_per_round:
            self.seen_trigger_steps_in_round.add(step_id)

    def _ensure_realtime_pipeline(self) -> None:
        if self._pipeline_ready:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Realtime recognition requires torch to be installed.") from exc

        from dataclasses import replace

        from skeleton3d_pipeline import RealtimeSkeleton3DPipeline, StreamingH36MFeatureExtractor
        from skeleton_pipeline.features.h36m_features import H36M_JOINT_NAMES
        from skeleton_pipeline.render.skeleton_video import draw_2d_skeleton

        self._h36m_joint_names = H36M_JOINT_NAMES

        self._torch = torch
        self.device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        # Resolved device wins over vision_config.device without mutating a config the
        # caller may be reusing elsewhere.
        self.vision_config = replace(self.vision_config, device=self.vision_config.device or self.device)

        if self.enable_step_model:
            from norm_feat_rlt import NormRealTime

            self.window_size = int(self.model_config["window_size"])
            self.num_steps = int(self.model_config["num_steps"])
            self.buffer = deque(maxlen=self.window_size)

            #region: HERE WE INSTANTIATE THE TRAINED MODEL
            model_dir = str(self.model_dir.parent) if str(self.model_dir.name) in ["2d_skeleton", "3d_skeleton"] else str(self.model_dir)
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            from LSTM_model_train import AssistLSTM
            self._model = AssistLSTM(
                input_dim=int(self.model_config["input_dim"]),
                hidden_dim=int(self.model_config["hidden_dim"]),
                num_steps=self.num_steps,
            ).to(self.device)
            self._model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self._model.eval()
            #endregion

            self._feature_extractor = StreamingH36MFeatureExtractor(self.vision_config)
            self._norm_real_time = NormRealTime(str(self.norm_path), self.feature_keys)
            self._ensure_stabilizer()
        else:
            print("Vision-only mode: YOLO 2D pose + MotionBERT 3D lift are live; "
                  "LSTM step classification is disabled (no norm stats needed).")

        self._skeleton_pipeline = RealtimeSkeleton3DPipeline(self.vision_config)
        self._draw_2d_skeleton = draw_2d_skeleton

        if self.show_video:
            # Four-view (oblique/front/side/top) orthographic panel of the
            # world-frame/camera-frame skeleton, shown alongside the 2D
            # overlay -- see _show_frame() and skeleton_video.py's
            # FastSkeleton3DRenderer docstring.
            from skeleton_pipeline.render.skeleton_video import FastSkeleton3DRenderer
            self._renderer_3d = FastSkeleton3DRenderer(self.display_panel_size)

        self._pipeline_ready = True

    def _ensure_stabilizer(self) -> None:
        if self.step_stabilizer is not None:
            return

        try:
            from step_stablizier import StepIdStabilizer
        except ImportError:
            from step_stablizier import StepStabilizer as StepIdStabilizer

        self.step_stabilizer = StepIdStabilizer(
            num_steps=self.num_steps,
            smoothing_window=STEP_SMOOTHING_WINDOW,
            confirmation_count=STEP_CONFIRMATION_COUNT,
            min_confidence=STEP_MIN_CONFIDENCE,
            min_margin=STEP_MIN_MARGIN,
        )

    def _stable_step_id(self, probabilities):
        if self.step_stabilizer is None:
            return int(np.argmax(probabilities))
        return self.step_stabilizer.update(probabilities)

    def _find_model_path(self) -> Path:
        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected trained model at {model_path}.")
        return model_path

    def _find_norm_path(self) -> Path:
        candidates = sorted(self.model_dir.glob("*.npz"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No norm .npz file found in {self.model_dir}.")
        raise FileExistsError(f"Expected exactly one norm .npz file in {self.model_dir}, found {len(candidates)}.")

    def _load_model_config(self) -> dict[str, Any]:
        with self.model_config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _build_feature_vector(self, features: dict[str, Any]) -> np.ndarray:
        values = []
        for key in self.feature_keys:
            for part_key in COMPOSITE_FEATURE_KEYS.get(key, (key,)):
                try:
                    value = features[part_key]
                except KeyError:
                    raise KeyError(
                        f"Feature '{part_key}' (for model config feature_key '{key}') is not "
                        f"produced by the feature extractor. Available features: "
                        f"{sorted(features)}."
                    ) from None
                if self._torch is not None and isinstance(value, self._torch.Tensor):
                    value = value.detach().cpu().numpy()
                values.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(values, axis=0)

    def _read_frame(self):
        camera = self.vision_config.camera
        if camera.video_source is None:
            return None

        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Reading from a video source requires opencv-python.") from exc
            self._cv2 = cv2

        if self._capture is None:
            self._ensure_camera_calibration()
            self._capture = self._open_capture()

        ok, frame = self._capture.read()
        return frame if ok else None

    def _open_capture(self):
        """Open vision_config.camera.video_source -- a live iPhone
        (Record3D) connection for "iphone" (see
        camera_utils/iphone_connection.py), otherwise a plain
        cv2.VideoCapture (webcam index, video file path, or stream URL --
        cv2.VideoCapture already handles all three). camera.live is
        auto-detected via _classify_video_source() unless already set
        explicitly, and written back so it's introspectable afterwards."""
        camera = self.vision_config.camera
        is_live, is_iphone = self._classify_video_source(camera.video_source)
        if camera.live is None:
            camera.live = is_live

        if is_iphone:
            self._ensure_src_on_path()
            from camera_utils.iphone_connection import IPhoneVideoCaptureAdapter
            return IPhoneVideoCaptureAdapter(
                dev_idx=camera.dev_idx, capture_rotate90=camera.capture_rotate90)

        source = camera.video_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        capture = self._cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {camera.video_source}")
        return capture

    @staticmethod
    def _classify_video_source(video_source) -> tuple[bool, bool]:
        """Returns (is_live, is_iphone) for a video_source value, so
        callers don't have to pass live= explicitly:
          - an int, or a digit-only string (e.g. "0") -> live webcam index.
          - "iphone" (case-insensitive) -> live iPhone via Record3D.
          - a string starting with a known stream URL scheme (rtsp://,
            http(s)://, udp://, tcp://) -> live network stream.
          - a string ending in a known video file extension -> recorded file.
          - anything else (an unrecognized path/string) -> treated as a
            recorded file too, the same as cv2.VideoCapture's own default
            assumption for a plain path.
        """
        if isinstance(video_source, int):
            return True, False
        if not isinstance(video_source, str):
            return False, False

        normalized = video_source.strip().lower()
        if normalized == "iphone":
            return True, True
        if normalized.isdigit():
            return True, False
        if normalized.startswith(_STREAM_URL_PREFIXES):
            return True, False
        return False, False

    def _ensure_camera_calibration(self) -> None:
        """Load intrinsics (required for 3D posture) / extrinsics
        (optional -- defines the world origin, see
        src/pose_detection_live.py's module docstring) once per capture,
        so a missing/bad calibration fails fast at camera-open time
        instead of on the first processed frame. RealtimeSkeleton3DPipeline
        (see skeleton3d_pipeline.py) loads its own copy independently when
        the LSTM pipeline spins up -- this is just an early sanity check."""
        if self._camera_K is not None or self.have_extrinsics:
            return

        camera = self.vision_config.camera
        if camera.calib_dir is None:
            return

        self._ensure_src_on_path()
        from camera_utils.calibration_io import load_extrinsics, load_intrinsics

        calib_dir = Path(camera.calib_dir)
        intrinsics_path = calib_dir / camera.intrinsics_file
        extrinsics_path = calib_dir / camera.extrinsics_file
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"No intrinsics found at {intrinsics_path}. Run src/app/calibrate_camera.py "
                "first (see src/pose_detection_live.py's module docstring).")
        self._camera_K, self._camera_dist, self._camera_image_size = load_intrinsics(intrinsics_path)

        self.have_extrinsics = extrinsics_path.exists()
        if self.have_extrinsics:
            self.T_world_from_camera, _ground_z, _robot_base = load_extrinsics(extrinsics_path)
            camera_up_world = self.T_world_from_camera[:3, :3] @ np.array([0.0, -1.0, 0.0])
            tilt_deg = np.degrees(np.arccos(np.clip(camera_up_world[2], -1.0, 1.0)))
            print(f"Loaded extrinsics from {extrinsics_path} -- camera tilt "
                  f"~{tilt_deg:.1f}deg from vertical per this calibration.")
        else:
            print(f"NOTE: no extrinsics at {extrinsics_path} -- posture will stay "
                  "camera-frame, see src/pose_detection_live.py's module docstring.")

    @staticmethod
    def _ensure_src_on_path() -> None:
        src_dir = Path(__file__).resolve().parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

    @staticmethod
    def _looks_like_frame(input_data) -> bool:
        return hasattr(input_data, "shape") and len(getattr(input_data, "shape", [])) >= 2
