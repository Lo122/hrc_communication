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

Defaults match eval/pose_detection_live.py's CLI defaults.
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
    see camera_utils/iphone_connection.py and eval/pose_detection_live.py's
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
    realtime_playback: bool = False  # Recorded files only (ignored when live). False (default)
                                      # = read the file frame by frame, so the model sees EVERY
                                      # frame no matter how slow it is. True = play the file
                                      # against the wall clock like a real camera: frames that
                                      # "arrived" while the model was busy are dropped, and the
                                      # reader waits when the model is faster than the recording.
                                      # See RecognitionManager._read_frame_on_wall_clock().
    playback_speed: float = 1.0  # realtime_playback only: how fast the video clock runs relative
                                  # to the wall clock. 1.0 = the recording's own rate. >1.0
                                  # consumes the file faster, which is the same thing as
                                  # simulating a model 1/playback_speed as fast (more frames
                                  # dropped per inference); <1.0 simulates a faster model. Frame
                                  # TIMESTAMPS stay on the recording's own timeline either way,
                                  # so measured velocities/accelerations are unaffected.
    dev_idx: int = 0  # Record3D device index, only used when video_source == "iphone".
    capture_rotate90: int = 0  # one of 0, 90, 180, 270 -- iPhone only, MUST match whatever
                                # was used when calibrating iphone_intrinsics.json/
                                # iphone_extrinsics.json, or K and the world frame will be
                                # wrong for these frames.

    # Requested capture mode, live devices only (a recorded file plays at the size it was
    # written at). None = request the resolution intrinsics_file was calibrated at, which
    # is almost always what is wanted: K does not transfer between resolutions, so frames
    # at any other size scale fx, fy, cx and cy and put every world position out by that
    # factor. Asking matters because nothing negotiates it for us -- Windows/DirectShow
    # settles on 640x480 however capable the device is, so an OBS Virtual Camera emitting
    # 1920x1080 arrives as 640x480 unless asked otherwise. Set explicitly only to capture
    # at a size the calibration was NOT solved for; FrameSource then warns, and
    # setup/rescale_intrinsics.py converts the calibration to match.
    capture_width: int | None = None
    capture_height: int | None = None
    capture_backend: str | None = None  # None/"auto" = DirectShow on Windows (far more reliable
                                         # for virtual cameras), OpenCV's choice elsewhere. Also
                                         # "dshow", "msmf", "any". Ignored for recorded files.

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
    motionbert_fp16: bool = False  # Run the DSTformer forward pass in fp16 (CUDA only; ignored with
                                    # a warning on CPU). Measured 31.0 -> 17.9 ms per lift at
                                    # clip_len=81 on an RTX 3060 Laptop, a 1.73x speedup on the
                                    # single most expensive stage of the live loop. Safe to flip
                                    # WITHOUT regenerating training data -- see
                                    # MotionBERTStreamingLifter's docstring for the A/B numbers
                                    # (0.200 mm rms vs fp32, 2% of the augmentation jitter) and for
                                    # why reducing clip_len is NOT an equivalent knob.

    # MetricDepthEstimator (absolute distance from the camera). Depth scales
    # LINEARLY with user_height_m (z = f_y * height * TORSO_HEIGHT_RATIO /
    # torso_px), so a wrong height is a proportional error in every absolute
    # world position -- a 1.90 m person left at this 1.70 m default is ~10%
    # out. body_calibration_file overrides it with a measured value.
    user_height_m: float = 1.70
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

    # BoneLengthConstraintFilter -- rescales each bone to a slowly-adapting
    # per-subject target, countering the frame-to-frame shrink/stretch a
    # monocular lifter produces from depth ambiguity. ON by default because
    # generate_lstm_training_data.py applies it to every TRAINING frame: with
    # it off, the model is fed a systematically noisier skeleton live than the
    # one it was trained on.
    use_bone_length_filter: bool = True
    # Optional body_calibration.py JSON (see calibrate_body.py). When present
    # it seeds the bone filter's targets and, if it carries a ground-plane
    # stature, overrides user_height_m below.
    body_calibration_file: str | Path | None = None

    # KeypointOutlierHoldFilter + shared confidence gate.
    conf_threshold: float = 0.3
    use_keypoint_filter: bool = True

    # StreamingH36MFeatureExtractor's smoothing window -- see that class's
    # docstring / feature_utils/h36m_features.py's "Why
    # Savitzky-Golay" note. fps is informational only (e.g. saved into
    # pose_detection_live.py's output .npz) -- StreamingH36MFeatureExtractor
    # fits velocity/acceleration against each frame's REAL timestamp, not
    # an assumed constant fps, since live capture spacing isn't uniform
    # (see that class's docstring).
    fps: float = 30.0
    feature_window_length: int = 9
    feature_polyorder: int = 3
