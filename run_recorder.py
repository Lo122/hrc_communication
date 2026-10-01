"""Record the camera during a run, without recognition: the reactive system's demo takes
(run_system.py --reactive --record-camera starts it beside communication).

The reactive system reads no camera, but every test still needs its video. This writes
the raw camera frames of the whole run to <log-dir>/<run-name>/camera.mp4 and, beside
it, camera.timestamps.csv: when each frame was taken, in epoch seconds -- the clock
communication logs with (communication_events.jsonl, timeline.csv) -- so the video lines
up with what the robot and the human did. The .mp4 is stamped with the frame rate the
camera actually delivers, measured over its first frames, so it plays back at
wall-clock speed.

Nothing but a capture and a writer runs: no pose model, no GPU. Both are recognition's
own (FrameSource, VideoRecorder), so the camera flags are run_recognition.py's and a
take is the same kind of file its --record-raw writes -- it can be run through
recognition afterwards (run_recognition.py --video-source .../camera.mp4 with the same
calibration flags).

Camera:
    --camera                     webcam 0 (the default)
    --video-source INDEX|URL     another webcam, or a stream
    --iphone [--iphone-rotate]   an iPhone over Record3D
A webcam is opened at the resolution --intrinsics-file (in --calib-dir) was calibrated
at, as recognition opens it, so a take can be replayed through recognition;
--capture-width/--capture-height override that. Without an intrinsics file the camera
picks.

Stopping: q in the preview window, Ctrl+C in this window, or the --stop-file appearing
(how run_system.py ends it when the run ends). Each finishes the .mp4. Closing this
window with its X, or killing the process, does not -- the .mp4 is then unplayable.

Usage:
    uv run python run_recorder.py --run-name demo_p01
    uv run python run_recorder.py --video-source 6 --intrinsics-file intrinsics_640x360_obs.json
    uv run python run_recorder.py --iphone --iphone-rotate 270 --output takes/p01.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "0_core", ROOT / "1_recognition", ROOT / "1_recognition" / "src"):
    sys.path.insert(0, str(path))

import config
from camera_utils.frame_source import BACKEND_NAMES, FrameSource
from logging_setup import configure_logging, get_logger
from render_utils.video_recorder import VideoRecorder
from vision_model.vision_config import DEFAULT_CALIB_DIR, CameraConfig

logger = get_logger(__name__)

VIDEO_NAME = "camera.mp4"
LOG_NAME = "recorder.log"
# As run_recognition.py's.
REALTIME_CAMERA_INDEX = 0
IPHONE_SOURCE = "iphone"
IPHONE_CAPTURE_ROTATE90 = 90
IPHONE_INTRINSICS_FILE = "iphone_intrinsics.json"
# The frame rate is measured over this many frames before the .mp4 is opened...
RATE_FRAMES = 20
# ...or taken as this if too few frames came to measure it, or the rate is not a
# camera's (a source repeating frames as fast as they are read).
FALLBACK_FPS = 30.0
FPS_RANGE = (1.0, 120.0)
PREVIEW_WINDOW = "camera recording (q stops)"
PREVIEW_MAX_WIDTH = 1280


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the camera during a run, without recognition.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", action="store_true", help="Record webcam 0 (the default).")
    source.add_argument("--video-source", default=None, help="Webcam index or stream URL to record.")
    source.add_argument("--iphone", action="store_true", help="Record an iPhone over Record3D.")
    parser.add_argument("--iphone-dev-idx", type=int, default=0, help="Record3D device index.")
    parser.add_argument("--iphone-rotate", type=int, choices=(0, 90, 180, 270), default=IPHONE_CAPTURE_ROTATE90,
                        help="Quarter turns applied to the iPhone frames, as for recognition.")
    parser.add_argument("--calib-dir", default=str(DEFAULT_CALIB_DIR),
                        help="Where --intrinsics-file is (default 1_recognition/calib_data).")
    parser.add_argument("--intrinsics-file", default=None,
                        help="A webcam is opened at the resolution this was calibrated at (default "
                             "intrinsics.json). Not needed: without it the camera picks.")
    parser.add_argument("--capture-width", type=int, default=None, help="Ask a webcam for this width.")
    parser.add_argument("--capture-height", type=int, default=None, help="Ask a webcam for this height.")
    parser.add_argument("--capture-backend", choices=BACKEND_NAMES, default=None,
                        help="OpenCV capture backend for a webcam (default: DirectShow on Windows).")
    parser.add_argument("--log-dir", default=str(ROOT / config.RUN_LOG_DIR),
                        help="Record to <LOG_DIR>/<RUN_NAME>/camera.mp4.")
    parser.add_argument("--run-name", default=None, help="Run directory name; defaults to run_<date>_<time>.")
    parser.add_argument("--output", default=None, help="Record to this .mp4 instead of the run directory.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Stamp the .mp4 with this frame rate instead of measuring the camera's.")
    parser.add_argument("--stop-file", default=None,
                        help="Stop, finishing the .mp4, once this file exists (run_system.py's stop).")
    parser.add_argument("--no-display", action="store_true", help="No preview window.")
    args = parser.parse_args(argv)
    if (args.capture_width is None) != (args.capture_height is None):
        parser.error("Give --capture-width and --capture-height together.")
    return args


def output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path(args.log_dir) / run_name / VIDEO_NAME


def camera_config(args: argparse.Namespace) -> CameraConfig:
    """What to open. A recording needs no calibration: an intrinsics file only sets the
    resolution a webcam is asked for."""
    camera = CameraConfig(calib_dir=None)
    if args.iphone:
        camera.video_source = IPHONE_SOURCE
        camera.dev_idx, camera.capture_rotate90 = args.iphone_dev_idx, args.iphone_rotate
        return camera  # Record3D delivers the phone's own resolution
    camera.video_source = args.video_source if args.video_source is not None else REALTIME_CAMERA_INDEX
    camera.capture_backend = args.capture_backend
    if args.capture_width is not None:
        camera.capture_width, camera.capture_height = args.capture_width, args.capture_height
        return camera
    intrinsics = Path(args.calib_dir) / (args.intrinsics_file or camera.intrinsics_file)
    if intrinsics.exists():
        from camera_utils.calibration_io import load_intrinsics
        _K, _dist, (camera.capture_width, camera.capture_height) = load_intrinsics(intrinsics)
    else:
        logger.warning("No intrinsics at %s: recording at the resolution the camera picks.", intrinsics)
    return camera


def record(frames, output: Path, *, stop, fps: float | None = None, preview=None, clock=time.time) -> float | None:
    """Write every frame `frames` (a FrameSource) delivers to output, until stop() is
    true, preview() returns False or a recorded source runs out. A camera's frame is
    stamped with clock() as it arrives, a recorded file's with its place on the file's
    own timeline. Returns the frame rate the .mp4 was stamped with (None if no frame
    came)."""
    recorder, pending = None, []  # frames held until the camera's rate is known
    try:
        while not stop():
            frame = frames.read()
            if frame is None:
                if frames.exhausted:
                    break
                continue  # a live camera's hiccup
            timeline = getattr(frames, "last_timestamp", None)
            now = clock() if timeline is None else timeline
            if recorder is None:
                pending.append((frame, now))
                if fps is not None or len(pending) >= RATE_FRAMES:
                    recorder = _start(output, fps, pending)
            else:
                recorder.write(frame, now)
            if preview is not None and not preview(frame):
                break
    finally:
        if recorder is None and pending:  # stopped before the rate was measured
            recorder = _start(output, fps, pending)
        if recorder is not None:
            recorder.close()
    return None if recorder is None else recorder.fps


def _start(output: Path, fps: float | None, pending: list) -> VideoRecorder:
    """Open the .mp4 at fps, or at the rate the pending frames came in, and write them."""
    if fps is None:
        span = pending[-1][1] - pending[0][1]
        fps = round((len(pending) - 1) / span, 2) if len(pending) > 1 and span > 0 else FALLBACK_FPS
        if FPS_RANGE[0] <= fps <= FPS_RANGE[1]:
            logger.info("The camera delivers %.2f frames per second.", fps)
        else:
            logger.warning("Measured %.2f frames per second, not a camera's rate: stamping %.0f; "
                           "the timestamps CSV has when each frame came.", fps, FALLBACK_FPS)
            fps = FALLBACK_FPS
    recorder = VideoRecorder(output, fps, label="camera video", write_timestamps=True)
    for frame, taken in pending:
        recorder.write(frame, taken)
    pending.clear()
    return recorder


def _preview(started: float):
    """Show each frame with a REC mark -- on a copy: the .mp4 gets the frame as taken.
    Returns False once q is pressed."""
    import cv2

    def show(frame) -> bool:
        height, width = frame.shape[:2]
        scale = min(1.0, PREVIEW_MAX_WIDTH / width)
        view = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1.0 else frame.copy()
        minutes, seconds = divmod(int(time.time() - started), 60)
        cv2.circle(view, (22, 22), 9, (0, 0, 255), -1)
        cv2.putText(view, f"REC {minutes:02d}:{seconds:02d}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(PREVIEW_WINDOW, view)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    return show


def main(argv=None) -> None:
    args = _parse_args(argv)
    output = output_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    configure_logging("run_recorder", log_file=output.parent / LOG_NAME)
    camera = camera_config(args)
    frames = FrameSource(camera, fallback_fps=FALLBACK_FPS)
    stop_file = Path(args.stop_file) if args.stop_file else None
    preview = None if args.no_display else _preview(time.time())

    print(f"Recording {camera.video_source} to {output}")
    print("Stop with q in the preview or Ctrl+C here -- not with this window's X, which "
          "leaves the video unplayable.")
    try:
        record(frames, output, stop=lambda: stop_file is not None and stop_file.exists(),
               fps=args.fps, preview=preview)
    except KeyboardInterrupt:
        pass
    finally:
        frames.release()
        if preview is not None:
            import cv2
            cv2.destroyAllWindows()
    print(f"Recording finished: {output}")


if __name__ == "__main__":
    main()
