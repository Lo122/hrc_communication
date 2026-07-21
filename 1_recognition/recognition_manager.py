"""Recognition pipeline integration."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import config
from models import RecognitionResult

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "best_model"
DEFAULT_YOLO_MODEL = Path(__file__).resolve().parent / "yolo26n-pose.pt"

STEP_SMOOTHING_WINDOW = 5
STEP_CONFIRMATION_COUNT = 3
STEP_MIN_CONFIDENCE = 0.6
STEP_MIN_MARGIN = 0.15


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
        yolo_model_path: str | Path = DEFAULT_YOLO_MODEL,
        video_source: str | int | None = None,
        feature_keys: list[str] | None = None,
        device: str | None = None,
        lstm_hrc_root: str | Path | None = None,
        show_video: bool = False,
        display_window_name: str = "HRC Recognition",
    ):
        self.step_stabilizer = step_stabilizer
        self.model_dir = Path(model_dir)
        self.model_config_path = Path(model_config_path) if model_config_path is not None else self.model_dir / "config.json"
        self.model_config = self._load_model_config()
        self.model_path = Path(model_path) if model_path is not None else self._find_model_path()
        self.norm_path = Path(norm_path) if norm_path is not None else self._find_norm_path()
        self.yolo_model_path = str(yolo_model_path)
        self.feature_keys = feature_keys or list(self.model_config["feature_keys"])
        self.device_name = device
        self.lstm_hrc_root = Path(lstm_hrc_root) if lstm_hrc_root is not None else self._find_lstm_hrc_root()
        self.show_video = show_video
        self.display_window_name = display_window_name

        self.window_size: int | None = None
        self.num_steps: int | None = None
        self.buffer = deque()

        self._torch = None
        self._cv2 = None
        self._model = None
        self._yolo_model = None
        self._norm_real_time = None
        self._kinematic_tracker = None
        self._setup_filtering = None
        self._smooth_kpt = None
        self._extract_features = None
        self._capture = None

        self.video_source = video_source
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
        """Run YOLO pose + LSTM inference for one frame."""
        #reduce the resolution of the frame to make recognition faster.
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Realtime recognition requires opencv-python.") from exc
            self._cv2 = cv2
        frame = self._cv2.resize(frame, (640, 480))
        
        self._ensure_realtime_pipeline()

        torch = self._torch
        results = self._yolo_model(frame)
        result = results[0]

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            keypoints = result.keypoints.xyn[0].cpu()
        else:
            keypoints = torch.zeros((17, 2))

        smoothed_keypoints = self._smooth_kpt(keypoints, self._kinematic_tracker)
        features = self._extract_features(smoothed_keypoints, selected_feats=self.feature_keys)
        features = self._norm_real_time.normalize_features(features)
        feature_vector = self._build_feature_vector(features)

        self.buffer.append(feature_vector)
        if len(self.buffer) < self.window_size:
            self._show_frame(frame, result)
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

        if stable_step_id is None:
            self._show_frame(frame, result, raw_step_id=raw_step_id, progress=progress, confidence=confidence)
            return None

        result = RecognitionResult(
            round_id=self.round_id,
            step_id=int(stable_step_id),
            progress=progress,
            piece_id=self.piece_id,
            confidence=confidence,
            timestamp=time.time() if timestamp is None else float(timestamp),
        )
        print(f"Raw Step: {raw_step_id} | Stable Step: {stable_step_id} | Progress: {progress}")
        self._show_frame(frame, result, raw_step_id=raw_step_id, stable_step_id=stable_step_id, progress=progress, confidence=confidence)

        self._record_step_and_advance_round(stable_step_id)
        return result

    def set_video_source(self, video_source: str | int | None) -> None:
        """Set or replace the OpenCV source used when update(None) is called."""
        self.release()
        self.video_source = video_source

    def release(self) -> None:
        """Release any owned OpenCV capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self.show_video and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.display_window_name)
            except self._cv2.error:
                pass

    def _show_frame(
        self,
        frame,
        yolo_result=None,
        *,
        raw_step_id: int | None = None,
        stable_step_id: int | None = None,
        progress: float | None = None,
        confidence: float | None = None,
    ) -> None:
        if not self.show_video:
            return

        self._cv2.imshow(self.display_window_name, frame)
        if self._cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def _result_from_passthrough(self, input_data: dict) -> RecognitionResult:
        step_id = input_data.get("step_id", 0)
        if self.step_stabilizer is not None and "step_probabilities" in input_data:
            step_id = self.step_stabilizer.update(input_data["step_probabilities"])

        result = RecognitionResult(
            round_id=self.round_id,
            step_id=step_id,
            progress=input_data.get("progress", 0.0),
            piece_id=self.piece_id,
            confidence=input_data.get("confidence", 0.0),
            timestamp=input_data.get("timestamp", time.time()),
        )
        self._record_step_and_advance_round(step_id)
        return result

    def _load_required_steps_per_round(self) -> set[int]:
        return {int(step_id) for step_id in config.TRIGGER_RULES}

    def _record_step_and_advance_round(self, step_id) -> None:
        if step_id is None:
            return

        step_id = int(step_id)
        if step_id == self._last_recorded_step_id:
            return

        self._last_recorded_step_id = step_id
        if step_id not in self.required_steps_per_round:
            return

        self.seen_trigger_steps_in_round.add(step_id)
        if self.required_steps_per_round.issubset(self.seen_trigger_steps_in_round):
            self.round_id += 1
            self.piece_id = self.round_id
            self.seen_trigger_steps_in_round.clear()

    def _ensure_realtime_pipeline(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Realtime recognition requires torch and ultralytics to be installed."
            ) from exc

        from norm_feat_rlt import NormRealTime

        self._import_feature_pipeline()

        self._torch = torch
        self.device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = int(self.model_config["window_size"])
        self.num_steps = int(self.model_config["num_steps"])
        self.buffer = deque(maxlen=self.window_size)

        #region: HERE WE INSTANTIATE THE TRAINED MODEL
        model_dir = str(self.model_dir)
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

        self._yolo_model = YOLO(self.yolo_model_path)
        self._norm_real_time = NormRealTime(str(self.norm_path), self.feature_keys)
        self._kinematic_tracker = self._setup_filtering()
        self._ensure_stabilizer()

    def _import_feature_pipeline(self) -> None:
        if self.lstm_hrc_root is not None:
            app_path = self.lstm_hrc_root / "application"
            data_proc_path = self.lstm_hrc_root / "data_proc_2d" / "src"
            for path in [app_path, data_proc_path]:
                if path.exists():
                    sys.path.insert(0, str(path))

        try:
            from extract_feat_rlt import extract_features, setup_filtering, smooth_kpt
        except ImportError as exc:
            raise RuntimeError(
                "Realtime recognition requires extract_feat_rlt.py and its data_proc_2d dependencies. "
                "Pass lstm_hrc_root=... if they are not next to this repository."
            ) from exc

        self._setup_filtering = setup_filtering
        self._smooth_kpt = smooth_kpt
        self._extract_features = extract_features

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
            value = features[key]
            if self._torch is not None and isinstance(value, self._torch.Tensor):
                value = value.detach().cpu().numpy()
            values.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(values, axis=0)

    def _read_frame(self):
        if self.video_source is None:
            return None

        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Reading from a video source requires opencv-python.") from exc
            self._cv2 = cv2

        if self._capture is None:
            self._capture = self._cv2.VideoCapture(self.video_source)
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open video source: {self.video_source}")

        ok, frame = self._capture.read()
        return frame if ok else None

    @staticmethod
    def _looks_like_frame(input_data) -> bool:
        return hasattr(input_data, "shape") and len(getattr(input_data, "shape", [])) >= 2

    @staticmethod
    def _find_lstm_hrc_root() -> Path | None:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root.parent / "LSTM_HRC"
        return candidate if candidate.exists() else None
