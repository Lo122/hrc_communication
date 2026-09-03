# 1_recognition

Realtime human step recognition for the HRC pipeline: a camera frame goes in,
a `RecognitionResult` (round/step/progress/confidence) comes out.

## Pipeline

```
frame -> YOLO 2D pose -> MotionBERT 2D->3D lift -> world-frame fusion
      -> streaming H36M kinematic features -> LSTM step classifier
      -> RecognitionResult
```

1. **YOLO 2D pose** (`ultralytics`, `vision_model/yolo26m-pose.pt`) detects
   17 COCO keypoints per frame.
2. **MotionBERT** (`src/skeleton_pipeline/motionbert_lifter.py`) lifts the 2D
   keypoints to a root-relative 3D body shape ("posture"), using a rolling
   causal window (`clip_len`, default 81 frames). This is a from-source
   PyTorch-only port of MotionBERT's `DSTformer` (no `mmpose`/`mmcv`
   dependency) — the actual model code and checkpoints live in an external
   clone of the [MotionBERT repo](https://github.com/Walter0807/MotionBERT),
   not in this project. See [MotionBERT setup](#motionbert-setup) below.
3. **`MetricDepthEstimator`** (`src/skeleton_pipeline/metric_depth_estimator.py`)
   estimates absolute distance from the camera ("location") from the 2D
   keypoints, calibrated intrinsics, and an assumed body height
   (`VisionConfig.user_height_m`).
4. If the camera has calibrated **extrinsics**, posture + location are fused
   into a single world-frame skeleton (`skeleton3d_pipeline.py`); otherwise
   posture stays camera-relative and location is skipped. See
   [Camera calibration](#camera-calibration).
5. **`StreamingH36MFeatureExtractor`** turns the fused skeleton into the same
   `pol_angles` / `joint_angles` / `ratios` kinematic features the LSTM was
   trained on (Savitzky–Golay smoothed), one frame at a time.
6. **`RecognitionManager`** (`recognition_manager.py`) buffers `window_size`
   feature vectors, runs the trained `AssistLSTM` (`best_model/`), and
   stabilizes the raw per-frame step prediction into a confirmed step
   transition (`step_stablizier.py`) before returning a `RecognitionResult`.

## Layout

| Path | Purpose |
|---|---|
| `recognition_manager.py` | Main entry point — wires the pipeline together, see `RecognitionManager`. |
| `skeleton3d_pipeline.py` | `RealtimeSkeleton3DPipeline` / `StreamingH36MFeatureExtractor` — per-frame posture+location+features. |
| `norm_feat_rlt.py` | Feature normalization at inference time, matching training-time stats (`best_model/*.npz`). |
| `step_stablizier.py` | Debounces raw per-frame step logits into confirmed step transitions. |
| `trigger_manager.py` | Turns confirmed steps into `Event`s for the rest of the pipeline. |
| `vision_model/vision_config.py` | `VisionConfig`/`CameraConfig` dataclasses — every knob for the pipeline above. |
| `vision_model/yolo26m-pose.pt` | YOLO 2D pose weights. |
| `best_model/` | Trained `AssistLSTM` checkpoint (`best_model.pth`), its `config.json` (window size, feature keys, step count, ...), and the feature-normalization `.npz`. |
| `calib_data/` | Camera intrinsics/extrinsics JSON (webcam + iPhone variants). See [Camera calibration](#camera-calibration). |
| `src/skeleton_pipeline/` | Posture/feature building blocks: COCO↔H36M remap, keypoint outlier filter, MotionBERT lifter, metric depth estimator, H36M kinematic features, rendering. |
| `src/camera_utils/` | Camera calibration (ChArUco/ArUco intrinsic+extrinsic) and iPhone (Record3D) capture. |
| `src/calibrate_camera.py` | CLI to run intrinsic/extrinsic calibration and write `calib_data/*.json`. |
| `src/pose_detection_live.py` | Standalone posture+location debug/preview tool (same pipeline, outside `RecognitionManager`). |

## Dependencies

Managed via `uv` from the repo-root `pyproject.toml` (`uv sync`). Key
packages for this part of the pipeline:

- `numpy`, `opencv-python`, `scipy` (Savitzky-Golay feature smoothing), `tqdm`
- `torch` / `torchvision` (CUDA build, `pytorch-cu128` index — needs a CUDA-
  capable GPU for realtime framerates; CPU works but is slow)
- `ultralytics` + `lap` (YOLO 2D pose + tracking support)
- MotionBERT's own minimal inference-only deps: `tensorboardX`, `easydict`,
  `prettytable`, `imageio-ffmpeg`, `roma`, `setuptools<81` (pinned — newer
  `setuptools` dropped `pkg_resources`, which `easydict`/`tensorboardX` still
  import). **Not** installed: `mmcv`/`mmpose`/`mmdet` — this pipeline only
  needs YOLO 2D + MotionBERT, not any of mmpose's own models.
- `matplotlib`, `pandas` (feature/label plotting, `src/skeleton_pipeline/plotting/`, `dataset/`)
- optional extra `iphone` → `record3d` (only needed for `video_source="iphone"`, see [iPhone (Record3D) capture](#iphone-record3d-capture))

```powershell
uv sync                    # base deps
uv sync --extra iphone     # + Record3D iPhone capture
```

### MotionBERT setup

The MotionBERT model code and checkpoints are **not vendored** in this repo
— `motionbert_lifter.py` imports `lib.utils.learning`/`lib.utils.tools` from
an external clone at runtime. Set it up once:

```powershell
git clone https://github.com/Walter0807/MotionBERT.git
```

By default it's expected 4 levels up from `motionbert_lifter.py`
(`.../MotionBERT`, i.e. a sibling of this repo's root); override with the
`MOTIONBERT_REPO_DIR` environment variable if you keep it elsewhere.

Inside that clone you need the fine-tuned H36M pose3d checkpoint (see
MotionBERT's own README for the download link — Google Drive/OneDrive):

```
MotionBERT/
├── configs/pose3d/MB_ft_h36m.yaml
└── checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin
```

This project uses the **full-size, `rootrel:True`** checkpoint
(`FT_MB_release_MB_ft_h36m`), not MotionBERT's smaller/faster "lite"
variant — only the root-relative body *shape* is used here (see
`motionbert_lifter.py`'s module docstring for why `rootrel:True` was picked
over `MB_ft_h36m_global`, which shares the same architecture).

### Model artifacts (this repo)

Already checked in / expected under `1_recognition/`:
- `vision_model/yolo26m-pose.pt` — YOLO 2D pose weights.
- `best_model/best_model.pth` + `best_model/config.json` + `best_model/*.npz` — trained LSTM step classifier + its training-time feature normalization stats. `RecognitionManager` fails fast at construction if any of these are missing.

## Camera calibration

3D posture needs calibrated **intrinsics** (`calib_data/intrinsics.json`);
world-frame location/fusion additionally needs calibrated **extrinsics**
(`calib_data/extrinsics.json`), which define the world origin everything
else is reported relative to. Without extrinsics the pipeline still runs but
stays camera-frame (posture only, tilted by however the camera is mounted).

```powershell
# Webcam, both in one go:
uv run python 1_recognition/src/calibrate_camera.py full --camera-index 0 `
    --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19

# iPhone (Record3D), both in one go:
uv run python 1_recognition/src/calibrate_camera.py iphone-full --capture-rotate90 90 `
    --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19
```

See `src/calibrate_camera.py`'s module docstring for the full subcommand
list (`intrinsic`/`extrinsic`/`full` and their `iphone-*` counterparts) and
`vision_model/vision_config.py`'s `CameraConfig` for where the resulting
JSON is expected to live.

## iPhone (Record3D) capture

Live iPhone capture goes through Apple's USB video stream via the
[Record3D](https://record3d.app/) app (paid, USB streaming mode), **not**
DroidCam/Wi-Fi — see `src/camera_utils/iphone_connection.py`.

1. Install the `iphone` extra: `uv sync --extra iphone`.
2. Install Record3D on the iPhone, connect it to the PC over USB, enable
   "USB Streaming" mode in the app.
3. Set `video_source="iphone"` (`RecognitionManager(video_source="iphone")`
   or `VisionConfig.camera.video_source`), and pick a `dev_idx` if more than
   one Record3D device is attached.
4. Pass the same `capture_rotate90` (0/90/180/270) you calibrated
   `iphone_intrinsics.json`/`iphone_extrinsics.json` with — it corrects the
   frame orientation *before* YOLO/MotionBERT see it, since both are trained
   on upright people and degrade badly on rotated input. Changing it later
   invalidates the existing calibration; recalibrate both intrinsics and
   extrinsics together whenever it changes.
5. USB drops (cable jostled, phone locked, app backgrounded) are
   auto-retried by a background watchdog in `IPhoneCamera`; callers just
   keep polling, no restart needed.

## Running

```powershell
# From the repo root, live webcam:
uv run python run_recognition.py --camera

# Recorded video file:
uv run python run_recognition.py --video-source path\to\clip.mp4

# Standalone posture/location debug preview (no LSTM, no event publishing):
uv run python 1_recognition/src/pose_detection_live.py --source 0 --device cuda:0 --user-height-m 1.75
```

Press `Q` to stop the preview window in either case.
