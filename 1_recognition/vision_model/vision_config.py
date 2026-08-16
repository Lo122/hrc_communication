"""Configuration for the realtime 3D vision/posture pipeline (YOLO 2D pose
-> MotionBERT 3D lift -> world-frame fusion -> streaming H36M kinematic
features) used by RecognitionManager/skeleton3d_pipeline.py.

Two dataclasses bundle what used to be ~15+ separate keyword args spread
across RecognitionManager, RealtimeSkeleton3DPipeline and
StreamingH36MFeatureExtractor:
  - CameraConfig: how to open the frame source (live camera vs. recorded
    video) and where its calibration (intrinsics/extrinsics) lives.
  - VisionConfig: everything downstream of "we have a frame" -- YOLO,
    MotionBERT, depth estimation, streaming feature extraction -- and
    nests a CameraConfig since the posture pipeline needs the calibration
    too (see skeleton3d_pipeline.py's RealtimeSkeleton3DPipeline).

Defaults match src/pose_detection_live.py's CLI defaults.
"""
from dataclasses import dataclass, field
from pathlib import Path

_VISION_MODEL_DIR = Path(__file__).resolve().parent
_RECOGNITION_DIR = _VISION_MODEL_DIR.parent

DEFAULT_YOLO_MODEL = _VISION_MODEL_DIR / "yolo26m-pose.pt"
DEFAULT_CALIB_DIR = _RECOGNITION_DIR / "calib_data"


@dataclass
class CameraConfig:
    """How to open the frame source, and where its calibration lives --
    see camera_utils/iphone_connection.py and src/pose_detection_live.py's
    module docstring (the extrinsics file is what fixes the WORLD ORIGIN
    posture/location get reported in)."""

    video_source: str | int | None = None  # webcam index, video file path, stream URL, or "iphone"
    live: bool | None = None  # None (default) = auto-detect from video_source (see
                               # RecognitionManager._classify_video_source): an int/digit
                               # string or "iphone" -> live, a stream URL -> live, anything
                               # else -> a recorded video file. Set True/False to override
                               # the auto-detection for an ambiguous source. Also gets
                               # written back with the detected value once a capture is
                               # opened, so it's readable afterwards either way.
    dev_idx: int = 0  # Record3D device index, only used when video_source == "iphone".
    capture_rotate90: int = 0  # one of 0, 90, 180, 270 -- iPhone only, MUST match whatever
                                # was used when calibrating iphone_intrinsics.json/
                                # iphone_extrinsics.json, or K and the world frame will be
                                # wrong for these frames.

    calib_dir: str | Path | None = DEFAULT_CALIB_DIR
    intrinsics_file: str = "intrinsics.json"
    extrinsics_file: str = "extrinsics.json"


@dataclass
class VisionConfig:
    # YOLO 2D pose.
    yolo_model_path: str | Path = DEFAULT_YOLO_MODEL
    yolo_imgsz: int | None = 640  # YOLO speed knob -- does NOT resize the frame itself, see
                                   # recognition_manager.py's update_from_frame docstring for why.
    device: str | None = None  # None = resolved by RecognitionManager ("cuda" if available else "cpu")

    camera: CameraConfig = field(default_factory=CameraConfig)

    # MotionBERT streaming 2D->3D lifter.
    motionbert_config: str | Path | None = None  # None = MotionBERTStreamingLifter's own default
    motionbert_checkpoint: str | Path | None = None  # None = MotionBERTStreamingLifter's own default
    clip_len: int = 81

    # MetricDepthEstimator (absolute distance from the camera).
    user_height_m: float = 1.70
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

    # KeypointOutlierHoldFilter + shared confidence gate.
    conf_threshold: float = 0.3
    use_keypoint_filter: bool = True

    # StreamingH36MFeatureExtractor's smoothing window -- see that class's
    # docstring / skeleton_pipeline/features/h36m_features.py's "Why
    # Savitzky-Golay" note.
    fps: float = 30.0
    feature_window_length: int = 9
    feature_polyorder: int = 3
