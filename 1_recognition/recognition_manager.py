"""Facade over the recognition pipeline.

RecognitionManager organises three collaborators -- FrameSource (where frames come
from), RealtimeSkeleton3DPipeline (frame -> 3D skeleton -> features) and DebugView
(optional preview windows) -- and owns what is left: the LSTM step classifier, the
feature buffer, and the round/step bookkeeping. One call to update() advances all
of it by a frame and returns a RecognitionResult on a confirmed step transition.
"""

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


import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "0_core"))
# src/ holds this layer's internals, on the path here so the collaborators below
# import at module level like everything else.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "best_model"))
import config
from models import RecognitionResult
from vision_model.vision_config import VisionConfig
from camera_utils.frame_source import FrameSource
from logging_setup import get_logger
from render_utils.debug_view import DebugView

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "best_model" / "3d_skeleton"
logger = get_logger(__name__)

# Model-config feature keys that are a concatenation of several extractor outputs, in
# this exact order. Each component is normalized separately first (the norm .npz holds
# per-component stats), then concatenated -- the order training used.
COMPOSITE_FEATURE_KEYS = {"pol_angles": ("polar_azimuth", "polar_elevation")}

STEP_SMOOTHING_WINDOW = 5
STEP_CONFIRMATION_COUNT = 3
STEP_MIN_CONFIDENCE = 0.2
STEP_MIN_MARGIN = 0.10


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
        render_world_skeleton: bool = False,
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
            # Vision-only: the 3D pipeline runs for real, but the LSTM and its norm stats
            # are never touched, so a missing .npz cannot stop the skeleton pipeline.
            self.model_config_path = None
            self.model_config = None
            self.model_path = None
            self.norm_path = None
            self.feature_keys = list(feature_keys or [])

        self.device_name = device
        self.show_video = show_video
        self.display_panel_size = display_panel_size

        # Every knob for the vision pipeline. The video_source kwarg, if given, overrides
        # whatever the config carries.
        self.vision_config = vision_config or VisionConfig()
        if video_source is not None:
            self.vision_config.camera.video_source = video_source

        # Both are inert until used: neither imports cv2 at construction, and the view is
        # a no-op entirely when show_video is False.
        self._frames = FrameSource(self.vision_config.camera,
                                   fallback_fps=self.vision_config.fps)
        self._view = DebugView(
            enabled=show_video,
            window_name=display_window_name, panel_size=display_panel_size,
            plot_window_name=plot_window_name, plot_panel_size=plot_panel_size,
            history_len=plot_history_len,
            conf_threshold=self.vision_config.conf_threshold,
            render_world_skeleton=render_world_skeleton)

        self.window_size: int | None = None
        self.num_steps: int | None = None
        self.buffer = deque()

        self._torch = None
        self._pipeline_ready = False
        self._model = None
        self._norm_real_time = None
        self._skeleton_pipeline = None
        self._feature_extractor = None

        # Per-frame snapshot of what the model saw and predicted, for eval/run_logger.py.
        # None until the first frame is processed.
        self.last_frame_record: dict[str, Any] | None = None

        # Latest world-frame position, refreshed every frame. Kept separate from
        # RecognitionResult, which only appears on confirmed step transitions and is far
        # too sparse for a live location feed. Holds its last valid value through frames
        # with no detection; None until the first valid one.
        self.last_world_xyz: tuple[float, float, float] | None = None
        self.last_location_timestamp: float | None = None

        # Posture only: a pelvis-relative H36M-17 skeleton, not fused with world position.
        # Same hold-last-valid policy as last_world_xyz. Joint names are resolved in
        # _ensure_realtime_pipeline, where h36m_features is first imported -- it pulls in
        # scipy, which a manager that never processes a frame need not pay for.
        self.last_root_relative_skeleton: np.ndarray | None = None
        # The same posture rotated into world axes -- what get_last_keypoints() publishes,
        # so that a consumer placing it at last_world_xyz gets an upright body rather than
        # one tilted by the extrinsics rotation.
        self.last_world_relative_skeleton: np.ndarray | None = None
        self._h36m_joint_names: list[str] | None = None

        # Per-step softmax probabilities and the raw progress-head output from the most
        # recent frame the LSTM actually ran on (i.e. once self.buffer is full) -- updated
        # every such frame regardless of whether a stable step transition was confirmed, so
        # a live plot (see step_probability_plot.py) can show raw model output over time
        # rather than only the sparse, debounced RecognitionResult stream.
        self.last_step_probabilities: np.ndarray | None = None
        self.last_mistake_probabilities: np.ndarray | None = None
        # Argmax of the above, and 1 - P(no mistake) as a single 0..1 score. Both stay
        # None for a model trained without a mistake head (config.json's num_mistakes).
        self.last_mistake_id: int | None = None
        self.last_mistake_score: float | None = None
        self.last_progress: float | None = None
        self.last_step_probabilities_timestamp: float | None = None

        self.round_id = 0
        self.piece_id = 0
        self.required_steps_per_round = self._load_required_steps_per_round()
        self.seen_trigger_steps_in_round: set[int] = set()
        self._last_recorded_step_id: int | None = None
        self.last_raw_step_id = None

    # -- capture state, read by run_recognition.py and eval/run_logger.py ---

    @property
    def source_exhausted(self) -> bool:
        """True once a recorded source runs out of frames.

        update() returns None both here and on a live camera's hiccup, so a caller
        that wants to stop at the end of a video needs this to tell the two apart.
        """
        return self._frames.exhausted

    @property
    def playback_frame_index(self) -> int | None:
        """Source index of the frame just read, or None for a live camera."""
        return self._frames.frame_index

    @property
    def playback_dropped_before(self) -> int:
        return self._frames.dropped_before

    @property
    def playback_frames_read(self) -> int:
        return self._frames.frames_read

    @property
    def playback_frames_dropped(self) -> int:
        return self._frames.frames_dropped

    @property
    def live_frame_age_mean_s(self) -> float | None:
        """Mean capture-to-read age of live frames, for sources that report it
        (iPhone). Stays small when the loop is keeping up with the camera."""
        if not self._frames.frames_read or not self._frames.frame_age_sum_s:
            return None
        return self._frames.frame_age_sum_s / self._frames.frames_read

    @property
    def T_world_from_camera(self):
        return self._frames.T_world_from_camera

    @property
    def have_extrinsics(self) -> bool:
        return self._frames.have_extrinsics

    def update(self, input_data=None) -> RecognitionResult | None:
        """Return the latest standardized recognition result."""
        if input_data is None:
            frame = self._frames.read()
            if frame is None:
                return None
            # last_timestamp is the frame's place on the recording's timeline, None for a
            # live source -- update_from_frame then falls back to time.time().
            # last_intrinsics is this frame's K for sources that report one (iPhone),
            # None for everything else, which keeps using the calibrated file K.
            return self.update_from_frame(frame, timestamp=self._frames.last_timestamp,
                                          K=self._frames.last_intrinsics)

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
        K=None,
    ) -> RecognitionResult | None:
        """Run one frame through pose -> 3D lift -> features -> LSTM.

        Returns a RecognitionResult only on a confirmed step transition; None while
        the window buffer fills, when the stabilizer has not committed, and in
        vision-only mode.

        The frame is passed at full resolution on purpose: the calibrated intrinsics
        and the depth estimator's world-position math assume pixel coordinates at the
        resolution calibration ran at. Use VisionConfig.yolo_imgsz as the speed knob
        instead.
        """
        self._ensure_realtime_pipeline()

        torch = self._torch
        frame_timestamp = time.time() if timestamp is None else float(timestamp)

        # Per-frame snapshot for offline analysis: step transitions alone are far too
        # sparse to study what frame loss does to the model. Published immediately and
        # mutated in place below, so every return path leaves a complete record behind.
        record: dict[str, Any] = {
            "timestamp": frame_timestamp,
            "skeleton": None,
            "detected": False,
            "warmup": True,
            "raw_step_id": None,
            "stable_step_id": None,
            "confidence": None,
            "progress": None,
            "world_xyz": None,
        }
        self.last_frame_record = record

        pipeline_out = self._skeleton_pipeline.process(frame, frame_timestamp, K=K)
        record["skeleton"] = pipeline_out.get("root_relative")
        record["detected"] = pipeline_out.get("root_relative") is not None

        world_xyz = pipeline_out.get("world_root_xyz")
        if world_xyz is not None and not np.isnan(world_xyz).any():
            self.last_world_xyz = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
            self.last_location_timestamp = frame_timestamp
            self._view.record_world(self.last_world_xyz)
            record["world_xyz"] = self.last_world_xyz
        # DEBUG, not INFO: fires every frame (~20/s) and would bury everything else.
        logger.debug("Frame %.3f: world_root_xyz=%s", frame_timestamp, self.last_world_xyz)

        root_relative = pipeline_out.get("root_relative")
        if root_relative is not None and not np.isnan(root_relative).any():
            self.last_root_relative_skeleton = root_relative
            # Published alongside last_world_xyz, so it has to share that frame -- see
            # get_last_keypoints(). Falls back to the camera-frame posture when there are
            # no extrinsics, which is the best available and is what world position does too.
            rotated = pipeline_out.get("root_relative_world")
            self.last_world_relative_skeleton = (
                rotated if rotated is not None and not np.isnan(rotated).any() else root_relative)

        if not self.enable_step_model:
            # Vision-only: the lift and world position above are real and keep feeding the
            # display and location stream, but with no classifier there is no result to
            # return -- triggers are typed in instead.
            self._show(frame, pipeline_out)
            return None

        # root_relative, NOT "skeleton" -- only this one matches the training data,
        # which is pelvis-centred at exactly (0,0,0), while "skeleton" is world-absolute
        # once extrinsics and depth resolve. Swapping them silently redefines most of the
        # feature set (positions stop being pelvis-relative, polar angles measure about
        # the world origin, velocities pick up whole-body locomotion), and worse, the
        # pipeline falls back to root_relative whenever depth fails -- so the
        # representation would flip mid-run and the derivative fit would read the
        # metres-scale jump as an enormous spurious velocity.
        features = self._feature_extractor.update(pipeline_out["root_relative"], frame_timestamp)
        features = self._norm_real_time.normalize_features(features)
        feature_vector = self._build_feature_vector(features)

        self.buffer.append(feature_vector)
        if len(self.buffer) < self.window_size:
            self._show(frame, pipeline_out)
            return None

        x = np.stack(self.buffer).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Models trained with a mistake head return a third output; models without
            # one return two.
            outputs = self._model(x)
            step_logits, progress_pred = outputs[0], outputs[1]
            mistake_logits = outputs[2] if len(outputs) > 2 else None
            step_probs = torch.softmax(step_logits, dim=1)
            raw_step_id = int(torch.argmax(step_probs, dim=1).item())
            confidence = float(torch.max(step_probs, dim=1).values.item())
            progress = float(progress_pred.item())
            mistake_probs = (None if mistake_logits is None
                             else torch.softmax(mistake_logits, dim=1).squeeze(0).cpu().numpy())

        probabilities = step_probs.squeeze(0).cpu().numpy()
        self.last_step_probabilities = probabilities
        self.last_mistake_probabilities = mistake_probs
        # Class 0 is "no mistake", so the mistake SCORE is 1 - P(class 0) -- with
        # num_mistakes=2 that is just P(class 1), but writing it this way keeps working
        # if a later model splits mistakes into several kinds.
        self.last_mistake_id = None if mistake_probs is None else int(np.argmax(mistake_probs))
        self.last_mistake_score = (None if mistake_probs is None
                                   else float(1.0 - mistake_probs[0]))
        self.last_progress = progress
        self.last_step_probabilities_timestamp = frame_timestamp
        stable_step_id = self._stable_step_id(probabilities)
        self.last_raw_step_id = raw_step_id
        self._view.record_prediction(progress, confidence)

        record.update(warmup=False, raw_step_id=raw_step_id, confidence=confidence,
                      progress=progress, stable_step_id=stable_step_id,
                      mistake_id=self.last_mistake_id,
                      mistake_score=self.last_mistake_score)


        if stable_step_id is None:
            self._show(frame, pipeline_out, raw_step_id=raw_step_id,
                       progress=progress, confidence=confidence)
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
        logger.info("Raw step: %s | Stable step: %s | Progress: %.3f | Confidence: %.3f",
                    raw_step_id, stable_step_id, progress, confidence)
        self._show(frame, pipeline_out, raw_step_id=raw_step_id, stable_step_id=stable_step_id,
                   progress=progress, confidence=confidence)

        return result

    def get_last_keypoints(self) -> dict[str, dict[str, float]] | None:
        """Latest pelvis-relative skeleton as {joint_name: {"x", "y", "z"}}, in WORLD axes.

        Pelvis is included and is always exactly (0,0,0). None until the first valid
        frame, or before the pipeline has been set up and joint names resolved.

        World axes, not the camera's, because this is published next to last_world_xyz
        (see run_recognition.py's HUMAN_LOCATION_UPDATE): a consumer that draws the body
        at that position needs a posture in the same frame, or the skeleton comes out
        rotated by however the camera is mounted. Without extrinsics there is no world
        frame and this falls back to the camera-frame posture.
        """
        skeleton = self.last_world_relative_skeleton
        if skeleton is None or self._h36m_joint_names is None:
            return None
        return {
            name: {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
            for name, xyz in zip(self._h36m_joint_names, skeleton)
        }

    def set_video_source(self, video_source: str | int | None, *, live: bool | None = None) -> None:
        """Set or replace the source used when update(None) is called.

        live=None auto-detects live (webcam index, stream URL, "iphone") vs. recorded
        file; pass True/False to override that for an ambiguous source.
        """
        self.release()
        self.vision_config.camera.video_source = video_source
        self.vision_config.camera.live = live

    def release(self) -> None:
        """Release the capture and close the debug windows."""
        self._frames.release()
        self._view.close()

    # -- debug display -----------------------------------------------------

    def _show(self, frame, pipeline_out, **prediction) -> None:
        """Hand one frame to the debug view, adding the context it cannot know:
        the latest world position, and what to say when there is no prediction yet."""
        # Mistake comes from the manager rather than the call sites: it is refreshed on
        # every frame the LSTM runs on, independently of whether a step was confirmed.
        prediction.setdefault("mistake_id", self.last_mistake_id)
        prediction.setdefault("mistake_score", self.last_mistake_score)
        self._view.show(frame, pipeline_out, world_xyz=self.last_world_xyz,
                        status_line=self._status_line(), **prediction)

    def _status_line(self) -> str:
        if not self.enable_step_model:
            return "Step model: OFF (manual triggers)"
        return f"Buffering: {len(self.buffer)}/{self.window_size or '?'}"

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

        # Advance only when the next cable-pulling step begins, so every frame of the
        # final step stays in the round it belongs to.
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
        from feature_utils.h36m_features import H36M_JOINT_NAMES
        from render_utils.skeleton_video import draw_2d_skeleton

        self._h36m_joint_names = H36M_JOINT_NAMES

        self._torch = torch
        self.device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        # replace() rather than assignment: the caller may be reusing this config.
        self.vision_config = replace(self.vision_config, device=self.vision_config.device or self.device)

        if self.enable_step_model:
            from feature_utils.feature_normalizer import NormRealTime

            self.window_size = int(self.model_config["window_size"])
            self.num_steps = int(self.model_config["num_steps"])
            self.buffer = deque(maxlen=self.window_size)

            # AssistLSTM's definition lives beside the checkpoint in best_model/, one
            # level above the per-variant subdirectory the weights sit in.
            model_dir = str(self.model_dir.parent) if str(self.model_dir.name) in ["2d_skeleton", "3d_skeleton"] else str(self.model_dir)
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            from LSTM_model_train import AssistLSTM
            # num_mistakes/num_layers/dropout must mirror the values the checkpoint was
            # trained with, or load_state_dict rejects the weights (a mistake_head trained
            # into the checkpoint has no place to go in a head-less model). Older configs
            # predate these keys, hence the defaults.
            num_mistakes = self.model_config.get("num_mistakes")
            self._model = AssistLSTM(
                input_dim=int(self.model_config["input_dim"]),
                hidden_dim=int(self.model_config["hidden_dim"]),
                num_steps=self.num_steps,
                num_layers=int(self.model_config.get("num_layers", 1)),
                dropout=float(self.model_config.get("dropout", 0.5)),
                num_mistakes=int(num_mistakes) if num_mistakes is not None else None,
            ).to(self.device)
            self._model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self._model.eval()

            self._feature_extractor = StreamingH36MFeatureExtractor(self.vision_config)
            self._norm_real_time = NormRealTime(str(self.norm_path), self.feature_keys)
            self._ensure_stabilizer()
        else:
            logger.info("Vision-only mode: YOLO 2D pose + MotionBERT 3D lift are live; "
                        "LSTM step classification is disabled (no norm stats needed).")

        self._skeleton_pipeline = RealtimeSkeleton3DPipeline(self.vision_config)

        renderer_3d = None
        if self.show_video:
            # Four-view (oblique/front/side/top) orthographic panel of the skeleton,
            # shown beside the 2D overlay.
            from render_utils.skeleton_video import FastSkeleton3DRenderer
            renderer_3d = FastSkeleton3DRenderer(self.display_panel_size)
        self._view.attach_renderers(draw_2d_skeleton, renderer_3d)

        self._pipeline_ready = True

    def _ensure_stabilizer(self) -> None:
        if self.step_stabilizer is not None:
            return

        from step_stabilizer import StepIdStabilizer

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

    @staticmethod
    def _looks_like_frame(input_data) -> bool:
        return hasattr(input_data, "shape") and len(getattr(input_data, "shape", [])) >= 2
