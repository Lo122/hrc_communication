"""Facade over the recognition pipeline.

RecognitionManager organises three collaborators -- FrameSource (where frames come
from), RealtimeSkeleton3DPipeline (frame -> 3D skeleton -> features) and DebugView
(optional preview windows) -- and owns what is left: the step classifier (in either
format, see src/step_models.py), the feature buffer, and the round/step bookkeeping.
One call to update() advances all of it by a frame and returns a RecognitionResult on
a confirmed step transition.
"""

from __future__ import annotations

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
# For task_sequence_model (the transition table the stabilizer filters with).
sys.path.append(str(Path(__file__).resolve().parents[1] / "2_decision_making" / "src"))
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
from render_utils.video_recorder import VideoRecorder
from step_models import StepModel, open_step_model

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "best_model" / "3d_skeleton"
logger = get_logger(__name__)

# A model fed at its own rate (step_models: sample_period_s) takes a frame up to this
# share of a period early, so a loop running at about that rate feeds every frame
# despite jitter, and a faster one every n-th.
SAMPLE_EARLY_TOLERANCE = 0.25

STEP_SMOOTHING_WINDOW = 7
STEP_CONFIRMATION_COUNT = 3
STEP_MIN_CONFIDENCE = 0.5
STEP_MIN_MARGIN = 0.10

# Log one in every N frames that carry non-finite features or predictions -- see the
# counters in __init__.
NONFINITE_FEATURE_LOG_EVERY = 100


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
        plot_panel_size: tuple[int, int] = (600, 640),
        plot_history_len: int = 300,
        world_view_range_m: float = 3.0,
        trajectory_frames: int = 200,
        enable_step_model: bool = True,
        render_world_skeleton: bool = False,
        record_path: str | Path | None = None,
        record_fps: float = 20.0,
        raw_record_path: str | Path | None = None,
        vision_config: VisionConfig | None = None,
    ):
        self.step_stabilizer = step_stabilizer
        self.model_dir = Path(model_dir)
        self.enable_step_model = enable_step_model

        # The step classifier, in whichever format model_dir holds (step_models.py): the
        # legacy AssistLSTM or the 4-head GRU. Vision-only runs never open it, so a
        # missing file cannot stop the skeleton pipeline.
        self.step_model: StepModel | None = (
            open_step_model(self.model_dir, model_path=model_path,
                            model_config_path=model_config_path, norm_path=norm_path,
                            feature_keys=feature_keys)
            if enable_step_model else None)

        self.device_name = device
        self.show_video = show_video
        self.display_panel_size = display_panel_size
        self.world_view_range_m = world_view_range_m

        # Every knob for the vision pipeline. The video_source kwarg, if given, overrides
        # whatever the config carries.
        self.vision_config = vision_config or VisionConfig()
        if video_source is not None:
            self.vision_config.camera.video_source = video_source

        # Both are inert until used: neither imports cv2 at construction, and the view is
        # a no-op entirely when show_video is False AND no record_path is given (a
        # recording still needs the compositing work done, just not the windows).
        self._frames = FrameSource(self.vision_config.camera,
                                   fallback_fps=self.vision_config.fps)
        self._view = DebugView(
            enabled=show_video,
            window_name=display_window_name, panel_size=display_panel_size,
            plot_window_name=plot_window_name, plot_panel_size=plot_panel_size,
            history_len=plot_history_len,
            conf_threshold=self.vision_config.conf_threshold,
            render_world_skeleton=render_world_skeleton,
            record_path=record_path, record_fps=record_fps,
            step_names=config.STEP_NAMES, mistake_names=config.MISTAKE_NAMES,
            trajectory_len=trajectory_frames)
        # The unrendered camera frames, as the model gets them (after any capture
        # rotation), one per frame read -- so frame N here is frame N of record_path.
        # The timestamps CSV beside it says when each was taken, since the loop rate
        # the .mp4's fps assumes is not always the rate it actually ran at.
        self._raw_recorder = (
            VideoRecorder(raw_record_path, record_fps, label="raw camera video",
                          write_timestamps=True)
            if raw_record_path is not None else None)

        self.window_size: int | None = None
        self.num_steps: int | None = None
        self.buffer = deque()
        # Feeding a model at its own rate: when the next feature vector is due, and the
        # times of those in the buffer (to warn once if the loop is too slow for it).
        self._next_sample_due: float | None = None
        self._sample_times: deque[float] = deque()
        self._rate_checked = False

        self._torch = None
        self._pipeline_ready = False
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
        # World-frame velocity (m/s) from the position Kalman filter, refreshed together
        # with last_world_xyz. None when the filter is off or before the first position.
        self.last_world_velocity: tuple[float, float, float] | None = None

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
        # Multi-head models only (None for the legacy format): progress per step, the
        # background head's P(nobody is working), and each step's own 0-1 probability
        # (the task sigmoids, not scaled by 1 - P(idle) as last_step_probabilities is).
        self.last_step_progress: np.ndarray | None = None
        self.last_idle_probability: float | None = None
        self.last_task_probabilities: np.ndarray | None = None
        self.last_step_probabilities_timestamp: float | None = None
        # The overlay's step readout, held between predictions: a model fed at its own
        # rate skips loop frames (S3 at 10 Hz in a 13 fps loop: 1 in 4), and redrawing
        # those without it made the text flicker between the readout and "Buffering".
        self._last_readout: dict[str, Any] = {}

        self.round_id = 0
        self.piece_id = 0
        self.required_steps_per_round = self._load_required_steps_per_round()
        self.seen_trigger_steps_in_round: set[int] = set()
        self._last_recorded_step_id: int | None = None
        self.last_raw_step_id = None

        # Counters behind the throttled non-finite warnings below. Both conditions
        # persist for as long as nobody is in frame, so logging every frame would bury
        # the run log at loop_hz.
        self._nonfinite_feature_frames = 0
        self._nonfinite_prediction_frames = 0

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
            if self._raw_recorder is not None:
                # Before any processing, so nothing drawn later can end up in it.
                frame_time = self._frames.last_timestamp
                self._raw_recorder.write(frame, time.time() if frame_time is None else frame_time)
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
            velocity = pipeline_out.get("world_root_velocity")
            self.last_world_velocity = (
                (float(velocity[0]), float(velocity[1]), float(velocity[2]))
                if velocity is not None and np.isfinite(velocity).all() else None)
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
        if not self._sample_due(frame_timestamp):
            # A model fed at its own rate skips this frame: the features above still
            # advance (their velocity/acceleration fits need every frame), but nothing
            # is buffered or predicted until the next sample is due.
            record["warmup"] = len(self.buffer) < self.window_size
            self._show(frame, pipeline_out)
            return None
        feature_vector = self.step_model.feature_vector(features)
        if not np.all(np.isfinite(feature_vector)):
            # With nobody in frame the feature extractor holds a zero skeleton, and
            # panels that divide by a body dimension (the ratios panel's shoulder
            # width, the polar angles' radius) then divide by zero. Left alone that
            # NaN survives the LSTM as NaN logits, so a single undetected frame makes
            # every prediction NaN for the whole window_size that follows -- and the
            # debug plot's int(round(nan)) turns that into a crash.
            #
            # Zeros match what the extractor already substitutes for an absent body,
            # so the damage stays on this frame instead of spreading through the window.
            self._nonfinite_feature_frames += 1
            if self._nonfinite_feature_frames % NONFINITE_FEATURE_LOG_EVERY == 1:
                logger.warning(
                    "Non-finite features at t=%.3f (%d frame(s) so far); substituting "
                    "zeros. Normally means no person has been detected.",
                    frame_timestamp, self._nonfinite_feature_frames)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        self.buffer.append(feature_vector)
        self._sample_times.append(frame_timestamp)
        if len(self.buffer) < self.window_size:
            self._show(frame, pipeline_out)
            return None
        self._check_sample_rate()

        prediction = self.step_model.predict(np.stack(self.buffer).astype(np.float32))
        probabilities = prediction.step_scores
        raw_step_id = prediction.raw_step_id
        confidence = prediction.confidence
        progress = prediction.progress_of(raw_step_id)
        mistake_probs = prediction.mistake_probabilities
        if not prediction.is_finite():
            # Belt to the feature guard's braces: whatever produced it, a non-finite
            # prediction is not a prediction. Publishing it would seed the stabilizer's
            # smoothing window with NaN (which then never recovers, since NaN loses
            # every comparison) and push NaN into the debug plot. The frame is treated
            # exactly like the warm-up case: displayed, not predicted from.
            self._nonfinite_prediction_frames += 1
            if self._nonfinite_prediction_frames % NONFINITE_FEATURE_LOG_EVERY == 1:
                logger.warning(
                    "Model produced a non-finite prediction at t=%.3f (%d frame(s) so "
                    "far); skipping it.", frame_timestamp, self._nonfinite_prediction_frames)
            self._show(frame, pipeline_out)
            return None

        self.last_step_probabilities = probabilities
        self.last_mistake_probabilities = mistake_probs
        # Class 0 is "no mistake", so the mistake SCORE is 1 - P(class 0) -- with
        # num_mistakes=2 that is just P(class 1), but writing it this way keeps working
        # if a later model splits mistakes into several kinds.
        self.last_mistake_id = None if mistake_probs is None else int(np.argmax(mistake_probs))
        self.last_mistake_score = (None if mistake_probs is None
                                   else float(1.0 - mistake_probs[0]))
        self.last_progress = progress
        self.last_step_progress = prediction.step_progress
        self.last_idle_probability = prediction.idle_probability
        self.last_task_probabilities = prediction.task_probabilities
        self.last_step_probabilities_timestamp = frame_timestamp
        stable_step_id = self._stable_step_id(probabilities)
        self.last_raw_step_id = raw_step_id
        self._view.record_prediction(progress, confidence, raw_step_id=raw_step_id,
                                     stable_step_id=stable_step_id,
                                     mistake_id=self.last_mistake_id)

        record.update(warmup=False, raw_step_id=raw_step_id, confidence=confidence,
                      progress=progress, stable_step_id=stable_step_id,
                      mistake_id=self.last_mistake_id,
                      mistake_score=self.last_mistake_score,
                      idle_probability=self.last_idle_probability)


        if stable_step_id is None:
            self._show(frame, pipeline_out, raw_step_id=raw_step_id,
                       progress=progress, confidence=confidence)
            return None

        self._record_step_and_advance_round(stable_step_id)
        # The stable step's own progress: its lane for a multi-head model, whose raw
        # winner may be another step; the one value for the legacy format.
        progress = prediction.progress_of(int(stable_step_id))
        result = RecognitionResult(
            round_id=self.round_id,
            step_id=int(stable_step_id),
            progress=progress,
            piece_id=self.piece_id,
            confidence=confidence,
            timestamp=frame_timestamp,
            **self._step_outputs(probabilities, prediction.step_progress),
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
        """Release the capture, finish the recordings and close the debug windows."""
        self._frames.release()
        if self._raw_recorder is not None:
            self._raw_recorder.close()
        self._view.close()

    # -- debug display -----------------------------------------------------

    def _show(self, frame, pipeline_out, **prediction) -> None:
        """Hand one frame to the debug view, adding the context it cannot know:
        the latest world position, and what to say when there is no prediction yet."""
        if "raw_step_id" in prediction:
            self._last_readout = {key: prediction.get(key) for key in
                                  ("raw_step_id", "stable_step_id", "progress", "confidence")}
        else:  # no prediction this frame: keep showing the last one
            prediction = {**self._last_readout, **prediction}
        # Mistake and idle come from the manager rather than the call sites: they are
        # refreshed on every frame the LSTM runs on, whether or not a step was confirmed.
        prediction.setdefault("mistake_id", self.last_mistake_id)
        prediction.setdefault("mistake_score", self.last_mistake_score)
        prediction.setdefault("idle_probability", self.last_idle_probability)
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
            step_probabilities=input_data.get("step_probabilities"),
            step_progress=input_data.get("step_progress"),
        )
        return result

    def _step_outputs(self, probabilities, step_progress) -> dict:
        """Every step's score and own progress for the decision layer, which picks the
        step from them against the task sequence (sequence_step_selector.py). The
        stabilizer's smoothed scores -- what the stable step came from, and steadier
        than one frame's. Only over config.STEP_NAMES, the ids the decision layer uses."""
        if list(self.step_labels or []) != list(config.STEP_NAMES):
            return {}
        smoothed = (self.step_stabilizer.smoothed_probabilities
                    if self.step_stabilizer is not None else None)
        scores = probabilities if smoothed is None else smoothed
        return {
            "step_probabilities": [float(value) for value in scores],
            "step_progress": (None if step_progress is None
                              else [float(value) for value in step_progress]),
        }

    def _load_required_steps_per_round(self) -> set[int]:
        # Informational only: the decision layer's TaskTracker owns which piece is current.
        return {config.HUMAN_PULL_CABLES, config.HUMAN_CONNECT_PIPES, config.HUMAN_CLAMP_TOOL}

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
            self.step_model.load(torch, self.device)
            self.window_size = self.step_model.window_size
            self.num_steps = self.step_model.num_steps
            self.buffer = deque(maxlen=self.window_size)
            self._sample_times = deque(maxlen=self.window_size)
            logger.info("Loaded %s.", self.step_model.describe())

            self._feature_extractor = StreamingH36MFeatureExtractor(self.vision_config)
            self._ensure_stabilizer()
        else:
            logger.info("Vision-only mode: YOLO 2D pose + MotionBERT 3D lift are live; "
                        "LSTM step classification is disabled (no norm stats needed).")

        self._skeleton_pipeline = RealtimeSkeleton3DPipeline(self.vision_config)

        renderer_3d = None
        world_renderer = None
        if self.show_video:
            # Four-view (oblique/front/side/top) orthographic panel of the skeleton,
            # shown beside the 2D overlay, then the top-down world location.
            from render_utils.skeleton_video import FastSkeleton3DRenderer
            from render_utils.world_trajectory import WorldTrajectoryRenderer
            renderer_3d = FastSkeleton3DRenderer(self.display_panel_size)
            if self._skeleton_pipeline.have_extrinsics:
                T = self._skeleton_pipeline.T_world_from_camera
                world_renderer = WorldTrajectoryRenderer(
                    self.display_panel_size, view_range_m=self.world_view_range_m,
                    camera_xy_world=(float(T[0, 3]), float(T[1, 3])))
        self._view.attach_renderers(draw_2d_skeleton, renderer_3d, world_renderer)

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
            allowed_transitions=self._observed_transitions(),
            override_factor=config.RECOGNITION_FILTER_OVERRIDE_FACTOR,
            min_confidence_by_step=self._step_confidence_thresholds(),
        )

    def _step_confidence_thresholds(self) -> dict[int, float]:
        """{step id: min confidence} from the task database's "Action Confidence
        Threshold"; steps it leaves out use STEP_MIN_CONFIDENCE."""
        if self.num_steps != len(config.STEP_NAMES):
            return {}
        from task_database import TaskDatabase

        path = Path(__file__).resolve().parents[1] / config.TASK_DATABASE_PATH
        try:
            thresholds = TaskDatabase.from_json(path).confidence_thresholds
        except (OSError, ValueError) as exc:
            logger.warning("No per-step confidence from %s (%s); every step uses %.2f.",
                           path, exc, STEP_MIN_CONFIDENCE)
            return {}
        by_step = {config.STEP_NAMES.index(name): value for name, value in thresholds.items()
                   if name in config.STEP_NAMES}
        logger.info("Step confidence: %s; other steps %.2f.",
                    ", ".join(f"{config.STEP_NAMES[k]} {v:.2f}" for k, v in sorted(by_step.items())),
                    STEP_MIN_CONFIDENCE)
        return by_step

    def _observed_transitions(self) -> dict[int, list[int]] | None:
        """Step changes the annotated task sequences make plausible (see
        2_decision_making/src/task_sequence_model.py). None -> the stabilizer's default."""
        if self.num_steps != len(config.STEP_NAMES):
            logger.warning("Model has %s steps but config.STEP_NAMES has %s; "
                           "step transitions are not filtered.", self.num_steps, len(config.STEP_NAMES))
            return None
        from task_sequence_model import TransitionModel

        table_path = Path(__file__).resolve().parents[1] / config.TASK_TRANSITION_TABLE_PATH
        try:
            model = TransitionModel.from_csv(table_path)
        except OSError:
            logger.warning("Transition table %s not found; step transitions are not filtered.", table_path)
            return None
        allowed = model.allowed_transitions(config.STEP_NAMES, config.RECOGNITION_FILTER_MIN_PROBABILITY)
        # Reachable from any step (config.RECOGNITION_FILTER_OPEN_STEPS).
        open_steps = [config.STEP_NAMES.index(name) for name in config.RECOGNITION_FILTER_OPEN_STEPS
                      if name in config.STEP_NAMES]
        return {step: sorted(set(targets) | set(open_steps)) for step, targets in allowed.items()}

    def _stable_step_id(self, probabilities):
        if self.step_stabilizer is None:
            return int(np.argmax(probabilities))
        return self.step_stabilizer.update(probabilities)

    @property
    def step_labels(self) -> list[str] | None:
        """The model's step names (config.STEP_NAMES when they line up), or None
        without a step model."""
        return None if self.step_model is None else self.step_model.step_labels

    def _sample_due(self, timestamp: float) -> bool:
        """Whether this frame feeds the model. Every frame for a model without a rate
        of its own (legacy); otherwise one frame per sample_period_s, on a fixed grid so
        a loop between rates still averages the model's rate (a 15 Hz loop feeds 2 of
        every 3 frames: 10 Hz, not 7.5)."""
        period = self.step_model.sample_period_s
        if period is None:
            return True
        due = self._next_sample_due
        if due is not None and timestamp < due - SAMPLE_EARLY_TOLERANCE * period:
            return False
        # A loop that fell more than a period behind starts a new grid rather than
        # feeding a burst of catch-up samples.
        self._next_sample_due = (timestamp + period if due is None or timestamp > due + period
                                 else due + period)
        return True

    def _check_sample_rate(self) -> None:
        """Once, when the buffer first fills: warn if the loop fed the model much slower
        than it was trained at, since its window then spans more time than in training."""
        period = self.step_model.sample_period_s
        if self._rate_checked or period is None or len(self._sample_times) < 2:
            return
        self._rate_checked = True
        spacing = (self._sample_times[-1] - self._sample_times[0]) / (len(self._sample_times) - 1)
        if spacing > 1.5 * period:
            logger.warning("The %s model expects a feature vector every %.2f s but got one "
                           "every %.2f s: run the loop at %g Hz or more (--loop-hz).",
                           self.step_model.model_dir.name, period, spacing, 1.0 / period)

    @staticmethod
    def _looks_like_frame(input_data) -> bool:
        return hasattr(input_data, "shape") and len(getattr(input_data, "shape", [])) >= 2
