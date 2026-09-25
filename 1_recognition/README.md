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
2. **MotionBERT** (`src/skeleton_utils/motionbert_lifter.py`) lifts the 2D
   keypoints to a root-relative 3D body shape ("posture"), using a rolling
   causal window (`clip_len`, default 81 frames). This is a from-source
   PyTorch-only port of MotionBERT's `DSTformer` (no `mmpose`/`mmcv`
   dependency) — the actual model code and checkpoints live in an external
   clone of the [MotionBERT repo](https://github.com/Walter0807/MotionBERT),
   not in this project. See [MotionBERT setup](#motionbert-setup) below.
3. **`MetricDepthEstimator`** (`src/skeleton_utils/metric_depth_estimator.py`)
   estimates absolute distance from the camera ("location") from the 2D
   keypoints, calibrated intrinsics, and an assumed body height
   (`VisionConfig.user_height_m`).
4. If the camera has calibrated **extrinsics**, posture + location are fused
   into a single world-frame skeleton (`src/skeleton3d_pipeline.py`); otherwise
   posture stays camera-relative and location is skipped. See
   [Camera calibration](#camera-calibration).
5. **`StreamingH36MFeatureExtractor`** turns the fused skeleton into the same
   `pol_angles` / `joint_angles` / `ratios` kinematic features the LSTM was
   trained on (Savitzky–Golay smoothed), one frame at a time.
6. **`RecognitionManager`** (`recognition_manager.py`) buffers `window_size`
   feature vectors, runs the trained `AssistLSTM` (`best_model/`), and
   stabilizes the raw per-frame step prediction into a confirmed step
   transition (`src/step_stabilizer.py`) before returning a `RecognitionResult`.

## Layout

Four roles, one directory each. The line between the top level and `src/` is
**what other layers import**: the top level is this layer's public surface,
`src/` is everything nobody outside `1_recognition/` reaches for. `setup/`
runs once before the system to produce calibration files, and `eval/` is run
to exercise or measure the system and is never imported by it.

Inside `src/`, there is one package per **pipeline stage**, in the order a
frame travels through them. Nothing is nested more than one level.

```
1_recognition/
├── recognition_manager.py      public  ─┐ imported by run_recognition.py
├── trigger_manager.py                  │ (repo root) and tests/
├── vision_model/vision_config.py      ─┘
│
├── src/                        internals — not imported outside this layer
│   ├── camera_utils/               frames in, camera math
│   │   ├── frame_source.py
│   │   ├── iphone_connection.py
│   │   ├── calibration_io.py
│   │   └── transforms.py
│   ├── skeleton_utils/             2D keypoints → 3D skeleton
│   ├── feature_utils/              skeleton → feature vectors
│   ├── render_utils/               anything drawn on screen
│   ├── skeleton3d_pipeline.py      assembles the four packages above
│   └── step_stabilizer.py          LSTM output → confirmed step
│
├── setup/                      run once, writes calib_data/*.json
│   ├── calibrate_camera.py
│   ├── calibrate_body.py
│   └── calibration/                the solvers calibrate_camera.py drives
├── eval/                       run to measure/debug, never imported at runtime
│   ├── pose_detection_live.py
│   ├── run_logger.py
│   └── analyse_runs.py
├── vision_model/               VisionConfig + YOLO weights
├── best_model/                 trained AssistLSTM + its config/norm stats
└── calib_data/                 camera intrinsics/extrinsics JSON
```

### Public surface

The three things outside this layer import. `run_recognition.py` — the
process entry point — lives at the **repo root** next to
`run_communication.py` / `run_system.py`, because it is a live process that
publishes UDP events, not a tool, even when its input is a recorded file.

| Path | Purpose |
|---|---|
| `recognition_manager.py` | Facade — organises `FrameSource`, `RealtimeSkeleton3DPipeline` and `DebugView` into one `update()` that returns a `RecognitionResult`. Holds the round/step bookkeeping and the LSTM; everything else it delegates. |
| `trigger_manager.py` | Turns confirmed steps into `Event`s for the rest of the pipeline. Stays out of `src/` because it is the adapter to `0_core`'s event bus (`config`/`events`/`models`), not a recognition library — nothing inside this layer imports it. |
| `vision_model/vision_config.py` | `VisionConfig`/`CameraConfig` dataclasses — every knob for the pipeline, and the canonical `DEFAULT_CALIB_DIR`/`DEFAULT_YOLO_MODEL` paths. |

### Internals (`src/`)

Self-contained apart from `vision_model.vision_config`, which is a pure
dataclass leaf. Nothing here imports `config`, `events` or `models`.

| Path | Purpose |
|---|---|
| `src/camera_utils/` | **Frames in.** `frame_source.py` opens the source (webcam / iPhone / recorded file), owns the wall-clock playback clock and the frame-drop counters, and preflights the calibration; plus the Record3D connection, calibration JSON I/O and world-frame transforms. |
| `src/skeleton_utils/` | **2D → 3D.** COCO↔H36M remap, keypoint outlier filter, MotionBERT lifter, metric depth estimator, bone-length filter, per-subject body calibration. |
| `src/feature_utils/` | **Skeleton → features.** `h36m_features.py` computes the kinematic panels; `feature_normalizer.py` normalizes them against the training-time stats in `best_model/*.npz`. They live together because they must agree on feature names or the model silently gets wrong input. |
| `src/render_utils/` | **Anything drawn.** `skeleton_video.py` draws 2D overlays and the four-view 3D panel; `debug_view.py` owns the two optional OpenCV debug windows (preview + scrolling progress/confidence/world-XYZ plot) and consumes the former. |
| `src/skeleton3d_pipeline.py` | `RealtimeSkeleton3DPipeline` / `StreamingH36MFeatureExtractor` — assembles the four packages above into per-frame posture+location+features. |
| `src/step_stabilizer.py` | Debounces raw per-frame step logits into confirmed step transitions. |

### Setup (`setup/`)

Run these once per camera rig / per subject, before the live system. Both
write JSON into `calib_data/`.

| Path | Purpose |
|---|---|
| `setup/calibrate_camera.py` | Intrinsic/extrinsic calibration → `calib_data/*intrinsics.json`, `calib_data/*extrinsics.json`. See [Camera calibration](#camera-calibration). |
| `setup/calibrate_body.py` | Per-subject T-pose calibration → `calib_data/body_<subject>.json` (bone lengths + metric stature). See [Body calibration](#body-calibration). |
| `setup/calibration/` | The ChArUco/ArUco solvers `calibrate_camera.py` drives, plus the board detector. Each is also runnable on its own. They sit here rather than in `src/` because nothing at runtime imports them. |

The **printable target** is not in this repo: `generate_charuco_board.py` and
`generate_calibration_targets.py` live in the data-processing repo
(`LSTM_HRC/data_proc_3d/src/camera_utils/`). This repo only *detects* a
target and solves the camera pose from it. Print the board there, then keep
`--squares-x`/`--squares-y`/`--square-length-mm`/`--marker-length-mm`/
`--aruco-dict` identical to whatever you passed that script.

### Evaluation (`eval/`)

Tools for looking at and measuring the pipeline. Nothing here is imported by
the runtime layer — `run_recognition.py` loads `run_logger.py` lazily, only
when `--log-dir` is passed.

| Path | Purpose |
|---|---|
| `eval/pose_detection_live.py` | Standalone posture+location debug/preview tool (same pipeline, outside `RecognitionManager`). |
| `eval/run_logger.py` | Per-frame CSV logging for one run (`frames.csv`/`events.csv`/`run.json`), enabled by `run_recognition.py --log-dir`. |
| `eval/analyse_runs.py` | Compares logged runs on the recording's timeline and plots the result. See [Comparing runs](#comparing-runs). |

### Data and model artifacts

| Path | Purpose |
|---|---|
| `vision_model/yolo26m-pose.pt` | YOLO 2D pose weights. |
| `best_model/` | Trained `AssistLSTM` checkpoint (`3d_skeleton/best_model.pth`), its `config.json` (window size, feature keys, step count, ...), the feature-normalization `.npz`, and `LSTM_model_train.py` — which stays here because it defines the `AssistLSTM` class `RecognitionManager` imports at load time. |
| `calib_data/` | Camera intrinsics/extrinsics JSON (webcam + iPhone variants) and per-subject body calibrations. See [Camera calibration](#camera-calibration). |

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
- `matplotlib`, `pandas` (run comparison plots, `eval/analyse_runs.py`)
- `record3d` (for `video_source="iphone"`, see [iPhone (Record3D) capture](#iphone-record3d-capture))

```powershell
uv sync                    # everything, incl. Record3D iPhone capture
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
uv run python 1_recognition/setup/calibrate_camera.py full --camera-index 0 `
    --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19

# iPhone (Record3D), both in one go:
uv run python 1_recognition/setup/calibrate_camera.py iphone-full --capture-rotate90 90 `
    --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19
```

See `setup/calibrate_camera.py`'s module docstring for the full subcommand
list (`intrinsic`/`extrinsic`/`full` and their `iphone-*` counterparts) and
`vision_model/vision_config.py`'s `CameraConfig` for where the resulting
JSON is expected to live.

## Body calibration

Optional, per subject. Without it the bone-length filter seeds itself from
whatever the first live frame happened to measure, and absolute world
positions are scaled by `VisionConfig.user_height_m`'s 1.70 m assumption.

```powershell
# Hold a T-pose, then turn slowly through 180 deg:
uv run python 1_recognition/setup/calibrate_body.py --device cuda:0 --camera `
    --subject uid-01

# Then point the run at the resulting JSON:
uv run python run_recognition.py --camera `
    --body-calibration 1_recognition/calib_data/body_uid-01.json
```

The stature half needs calibrated **extrinsics**; without them the bone
lengths are still calibrated and stature is left null. See
`src/skeleton_utils/body_calibration.py` for why the turn is sampled
throughout rather than at its endpoints.

## iPhone (Record3D) capture

Live iPhone capture goes through Apple's USB video stream via the
[Record3D](https://record3d.app/) app (paid, USB streaming mode), **not**
DroidCam/Wi-Fi — see `src/camera_utils/iphone_connection.py`.

1. `uv sync` (record3d is a regular dependency).
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
uv run python 1_recognition/eval/pose_detection_live.py --source 0 --device cuda:0 --user-height-m 1.75
```

Press `Q` to stop the preview window in either case.

## Logging

Diagnostics go through `logging`, not `print`, so a run leaves a record you can
read afterwards. `src/logging_setup.py` configures it; every entry point calls
`configure_logging()` once at startup.

- **Console** gets `INFO` and above — what `print` used to show.
- **File** gets `DEBUG` and above, including the per-frame world-position trace
  that is far too noisy to watch live.

Where the file lands:

| Situation | Path |
|---|---|
| `run_recognition.py --log-dir results --run-name X` | `results/X/run.log`, beside that run's `frames.csv`/`events.csv`/`run.json` |
| any other run, or a `setup/`/`eval/` tool | `1_recognition/logs/<tool>_<YYYYmmdd_HHMMSS>.log` |

Both paths are gitignored. Interactive prompts ("press SPACE to capture…",
"Hold a T-pose…") stay on `print`, since a timestamp and level prefix only gets
in the way of a live console.

Handlers attach to a `"recognition"` parent logger with `propagate = False`, never
the root logger. That keeps this layer's records out of everyone else's handlers
and, just as importantly, keeps `ultralytics`/`torch`/`matplotlib` chatter out of
ours. It is **unrelated to `0_core/logger.py`'s `EventLogger`**, which is a separate
JSON-lines writer for the communication dialog (`hrc_communication_events.log`) and
does not use `logging` at all.

## Comparing runs

Processing latency makes the live system miss frames. To measure whether
that changes what the model *predicts*, run the same recording twice — once
frame by frame, once against the wall clock — and compare:

```powershell
uv run python run_recognition.py --video-source take01.mp4 `
    --log-dir results --run-name baseline --no-display

uv run python run_recognition.py --video-source take01.mp4 `
    --realtime-playback --loop-hz 1000 `
    --log-dir results --run-name realtime_1x --no-display

uv run python 1_recognition/eval/analyse_runs.py results --baseline baseline
```

`--log-dir` writes `frames.csv`/`events.csv`/`run.json` per run (see
`eval/run_logger.py` for the column list); `analyse_runs.py` aligns every run
on the *recording's* timeline and writes `summary.csv`, `triggers.csv` and
five figures into `<input>/analysis`.
