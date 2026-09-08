"""
    Run realtime recognition in its own process and publish trigger events.
    Usage:
        uv run python run_recognition.py --loop-hz 15 [--camera] [--video-source VIDEO_SOURCE]`
          --model-dir MODEL_DIR [--host HOST] [--port PORT] [--no-display] [--loop-hz HZ]

        uv run python run_recognition.py --loop-hz 15 --video-source iphone `
          --capture-rotate90 90 --dev-idx 0 --show-probabilities

        uv run python run_recognition.py --loop-hz 15 --show-probabilities

        uv run python run_recognition.py --loop-hz 15 --video-source iphone --capture-rotate90 90 `
          --extrinsics-file iphone_extrinsics_synthetic.json --show-probabilities
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parent
for layer in ["0_core", "1_recognition"]:
    sys.path.insert(0, str(ROOT / layer))

import config
from event_transport import UDPEventSender
from events import Event, EventType
from recognition_manager import DEFAULT_MODEL_DIR, RecognitionManager
from step_probability_plot import StepProbabilityPlot
from trigger_manager import TriggerManager
from vision_model.vision_config import VisionConfig

logger = logging.getLogger(__name__)


REALTIME_CAMERA_INDEX = 0
LOOP_HZ = 20.0  # target recognition loop rate -- matches the previous flat 50ms sleep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recognition and publish trigger events.")
    parser.add_argument("--camera", action="store_true", help="Use realtime camera input.")
    parser.add_argument("--video-source", default=None, help="Video path, stream URL, webcam index, or \"iphone\" for a live Record3D-connected iPhone. Defaults to config.test_vid_path.")
    parser.add_argument("--dev-idx", type=int, default=0,
                         help="--video-source iphone only. Index into Record3D's connected-device list.")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="--video-source iphone only. Rotate the working frame at the source "
                              "before it's used at all -- MUST match whatever --capture-rotate90 "
                              "was passed to calibrate_camera.py's iphone-intrinsic/iphone-extrinsic "
                              "when producing the iphone_intrinsics.json/iphone_extrinsics.json this "
                              "run reads, or K and the world frame will be wrong.")
    parser.add_argument("--calib-dir", default=None,
                         help="Directory holding calibration files. Defaults to VisionConfig's "
                              "own default (1_recognition/calib_data).")
    parser.add_argument("--intrinsics-file", default=None,
                         help="Intrinsics filename, relative to --calib-dir. Defaults to "
                              "\"iphone_intrinsics.json\" for --video-source iphone, else "
                              "\"intrinsics.json\".")
    parser.add_argument("--extrinsics-file", default=None,
                         help="Extrinsics filename, relative to --calib-dir -- this is what fixes "
                              "the world origin/orientation (see vision_model/vision_config.py's "
                              "CameraConfig docstring). Defaults to \"iphone_extrinsics.json\" for "
                              "--video-source iphone, else \"extrinsics.json\". Pass "
                              "\"iphone_extrinsics_synthetic.json\" to use a hand-authored, "
                              "assumed-level-camera extrinsics instead of a marker/board-calibrated "
                              "one -- see that file's \"notes\" field before trusting it.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing config.json and best_model.pth.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Communication event receiver host.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Communication event receiver port.")
    parser.add_argument("--no-display", action="store_true", help="Disable the recognition video preview window.")
    parser.add_argument("--show-probabilities", action="store_true",
                         help="Show a live line-graph preview window of each task step's "
                              "softmax probability and the progress value over time -- see "
                              "step_probability_plot.py. Independent of --no-display.")
    parser.add_argument("--probability-history-seconds", type=float, default=12.0,
                         help="--show-probabilities only. How many seconds of history the "
                              "line graph keeps on screen.")
    parser.add_argument("--loop-hz", type=float, default=LOOP_HZ,
                         help="Target recognition loop rate (Hz). Each iteration sleeps only "
                              "enough to hold this rate -- see the main loop's fixed-tick "
                              "scheduling -- instead of a flat per-iteration delay, so the "
                              "actual cadence stays consistent regardless of how long "
                              "recognition_manager.update() itself takes.")
    return parser.parse_args()


def _recognition_source(args: argparse.Namespace):
    if args.camera:
        return REALTIME_CAMERA_INDEX
    return args.video_source or config.test_vid_path


if __name__ == "__main__":
    args = _parse_args()
    source = _recognition_source(args)
    sender = UDPEventSender(args.host, args.port)
    is_iphone_source = isinstance(source, str) and source.strip().lower() == "iphone"
    vision_config = VisionConfig()
    vision_config.camera.dev_idx = args.dev_idx
    vision_config.camera.capture_rotate90 = args.capture_rotate90
    if args.calib_dir is not None:
        vision_config.camera.calib_dir = args.calib_dir
    vision_config.camera.intrinsics_file = args.intrinsics_file or (
        "iphone_intrinsics.json" if is_iphone_source else "intrinsics.json")
    vision_config.camera.extrinsics_file = args.extrinsics_file or (
        "iphone_extrinsics.json" if is_iphone_source else "extrinsics.json")
    recognition_manager = RecognitionManager(
        model_dir=args.model_dir, video_source=source, show_video=not args.no_display,
        vision_config=vision_config)
    trigger_manager = TriggerManager()
    probability_plot: StepProbabilityPlot | None = None
    last_plotted_probabilities_timestamp = None

    print(f"Recognition running with source: {source}")
    print(f"Publishing recognition events to {args.host}:{args.port}")

    # Fixed-tick loop scheduling (like an embedded millis()-based scheduler: accumulate the
    # IDEAL next tick time and sleep only the remainder, rather than sleeping a flat amount
    # after each iteration) -- keeps the loop's actual cadence close to --loop-hz regardless
    # of how long each recognition_manager.update() call takes, and resyncs instead of firing
    # a catch-up burst if an iteration overruns its budget.
    period_s = 1.0 / args.loop_hz
    next_tick = time.perf_counter()
    overrun_warned_at = 0.0

    frame_count = 0
    try:
        while True:
            result = recognition_manager.update()
            print(f"[recognition] step {result.step_id} round {result.round_id} piece {result.piece_id} "
                  f"progress {result.progress:.3f} confidence {result.confidence:.3f}" if result is not None else "[recognition] no result")
            if result is not None:
                for event in trigger_manager.update(result):
                    print(f"[recognition event] sending {event.event_type.name} {event.payload}", flush=True)
                    sender.send(event)
                    print(f"[recognition event] sent {event.event_type.name} {event.payload}", flush=True)

            # Live line-graph preview of raw per-frame model output (see
            # step_probability_plot.py) -- fed from RecognitionManager.last_step_probabilities/
            # last_progress, which update every frame the LSTM runs on regardless of whether
            # a stable step transition was confirmed (unlike the sparse `result` above).
            if args.show_probabilities:
                probs_timestamp = recognition_manager.last_step_probabilities_timestamp
                if probs_timestamp is not None and probs_timestamp != last_plotted_probabilities_timestamp:
                    if probability_plot is None:
                        probability_plot = StepProbabilityPlot(
                            recognition_manager.num_steps,
                            history_seconds=args.probability_history_seconds)
                    probability_plot.update(
                        probs_timestamp,
                        recognition_manager.last_step_probabilities,
                        recognition_manager.last_progress)
                    last_plotted_probabilities_timestamp = probs_timestamp
                elif probability_plot is None and recognition_manager.window_size is not None \
                        and frame_count % 15 == 0:
                    # The LSTM doesn't run at all -- so last_step_probabilities never gets
                    # set -- until self.buffer holds a full window_size frames. Log progress
                    # so a several-second wait before the window first appears doesn't look
                    # like a hang.
                    print(f"[recognition] buffering for step probabilities: "
                          f"{len(recognition_manager.buffer)}/{recognition_manager.window_size} frames")

            # Human location is a continuous, much-higher-frequency stream than the
            # discrete task events above -- throttled to config.HUMAN_LOCATION_
            # PUBLISH_EVERY_N_FRAMES (see recognition_manager.py's last_world_xyz,
            # updated every frame regardless of step-recognition state).
            frame_count += 1
            if frame_count % config.HUMAN_LOCATION_PUBLISH_EVERY_N_FRAMES == 0:
                world_xyz = recognition_manager.last_world_xyz
                if world_xyz is not None:
                    sender.send(Event(
                        event_type=EventType.HUMAN_LOCATION_UPDATE,
                        source="recognition",
                        payload={
                            "x": world_xyz[0], "y": world_xyz[1], "z": world_xyz[2],
                            "timestamp": recognition_manager.last_location_timestamp,
                            # Pelvis-relative (pelvis at (0,0,0)) H36M-17 skeleton -- see
                            # recognition_manager.py's get_last_keypoints()/
                            # last_root_relative_skeleton. Posture only, separate from the
                            # world-frame x/y/z location above.
                            "keypoints": recognition_manager.get_last_keypoints(),
                        },
                    ))

            next_tick += period_s
            now = time.perf_counter()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # This iteration overran its budget -- resync to now instead of letting the
                # deficit accumulate (which would otherwise fire a burst of iterations back
                # to back later to "catch up").
                if now - overrun_warned_at > 5.0:
                    logger.warning(f"[recognition] loop overran budget by {-sleep_s * 1000:.0f}ms "
                          f"(target {period_s * 1000:.0f}ms/iteration) -- update() is the "
                          "bottleneck, not the sleep.")
                    overrun_warned_at = now
                next_tick = now
    except KeyboardInterrupt:
        print("Recognition stopped.")
    finally:
        recognition_manager.release()
        if probability_plot is not None:
            probability_plot.release()
        sender.close()
