"""
    Run realtime recognition in its own process and publish trigger events.
    Usage:
        uv run python run_recognition.py --loop-hz 15 [--camera] [--video-source VIDEO_SOURCE]`
          --model-dir MODEL_DIR [--host HOST] [--port PORT] [--no-display] [--loop-hz HZ]

        Or, to send RECOGNITION_TRIGGER events by hand (no camera/video/model) and time how
        the communication layer reacts:
            uv run python run_recognition.py --manual-trigger [--host HOST] [--port PORT]
        then at the "> " prompt type: step_id[,piece_id[,round_id[,progress]]]
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
from trigger_manager import TriggerManager

DEFAULT_MANUAL_PIECE_ID = 1
DEFAULT_MANUAL_ROUND_ID = 0
DEFAULT_MANUAL_PROGRESS = 1.0

logger = logging.getLogger(__name__)


REALTIME_CAMERA_INDEX = 0
LOOP_HZ = 20.0  # target recognition loop rate -- matches the previous flat 50ms sleep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recognition and publish trigger events.")
    parser.add_argument("--camera", action="store_true", help="Use realtime camera input.")
    parser.add_argument("--video-source", default=None, help="Video path or stream URL. Defaults to config.test_vid_path.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing config.json and best_model.pth.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Communication event receiver host.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Communication event receiver port.")
    parser.add_argument("--no-display", action="store_true", help="Disable the recognition video preview window.")
    parser.add_argument("--manual-trigger", action="store_true",
                         help="Skip the camera/video/model pipeline entirely. Instead, prompt on the "
                              "terminal for step_id[,piece_id[,round_id[,progress]]] and send that "
                              "RECOGNITION_TRIGGER event over the real UDP path on Enter -- for timing "
                              "how fast the communication layer reacts without needing real model output.")
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


def _parse_manual_trigger_line(line: str) -> tuple[int, int, int, float] | None:
    """Parse "step_id[,piece_id[,round_id[,progress]]]" typed at the prompt.

    Missing fields fall back to DEFAULT_MANUAL_*. Returns None if the line is
    blank or not parseable (the caller re-prompts instead of crashing the loop).
    """
    parts = [p.strip() for p in line.split(",")]
    if not parts or not parts[0]:
        return None
    try:
        step_id = int(parts[0])
        piece_id = int(parts[1]) if len(parts) > 1 and parts[1] else DEFAULT_MANUAL_PIECE_ID
        round_id = int(parts[2]) if len(parts) > 2 and parts[2] else DEFAULT_MANUAL_ROUND_ID
        progress = float(parts[3]) if len(parts) > 3 and parts[3] else DEFAULT_MANUAL_PROGRESS
    except ValueError:
        print(f"Could not parse '{line}' as step_id[,piece_id[,round_id[,progress]]] -- try again.")
        return None
    return step_id, piece_id, round_id, progress


def _run_manual_trigger_loop(sender: UDPEventSender, host: str, port: int) -> None:
    """Interactively send RECOGNITION_TRIGGER events over the real UDP path.

    Bypasses RecognitionManager/TriggerManager entirely (no camera, no video,
    no model) so you can time how long the communication layer takes to react
    to a given step, independent of recognition inference time.
    """
    configured_steps = ", ".join(str(step) for step in sorted(config.TRIGGER_RULES))
    print(f"Manual trigger mode -- sending RECOGNITION_TRIGGER events to {host}:{port}")
    print(f"Configured human steps: {configured_steps}")
    print(f"Defaults when omitted: piece_id={DEFAULT_MANUAL_PIECE_ID}, "
          f"round_id={DEFAULT_MANUAL_ROUND_ID}, progress={DEFAULT_MANUAL_PROGRESS}")
    print("Enter: step_id[,piece_id[,round_id[,progress]]]  (Ctrl+C or 'q' to quit)")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line.lower() in ("q", "quit", "exit"):
            break

        parsed = _parse_manual_trigger_line(line)
        if parsed is None:
            continue
        step_id, piece_id, round_id, progress = parsed
        if step_id not in config.TRIGGER_RULES:
            print(f"Step {step_id} has no recognition trigger configured. "
                  f"Configured human steps: {configured_steps}.")
            continue

        event = Event(
            event_type=EventType.RECOGNITION_TRIGGER,
            source="manual_recognition",
            payload={
                "step_id": step_id,
                "piece_id": piece_id,
                "round_id": round_id,
                "progress": progress,
            },
        )
        send_time = time.time()
        sender.send(event)
        print(f"[manual trigger] sent step_id={step_id} piece_id={piece_id} "
              f"round_id={round_id} progress={progress} at t={send_time:.6f}")


if __name__ == "__main__":
    args = _parse_args()
    sender = UDPEventSender(args.host, args.port)

    if args.manual_trigger:
        try:
            _run_manual_trigger_loop(sender, args.host, args.port)
        except KeyboardInterrupt:
            pass
        finally:
            sender.close()
        sys.exit(0)

    source = _recognition_source(args)
    recognition_manager = RecognitionManager(model_dir=args.model_dir, video_source=source, show_video=not args.no_display)
    trigger_manager = TriggerManager()

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
            if result is not None:
                for event in trigger_manager.update(result):
                    print(f"[recognition event] sending {event.event_type.name} {event.payload}", flush=True)
                    sender.send(event)
                    print(f"[recognition event] sent {event.event_type.name} {event.payload}", flush=True)

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
        sender.close()
