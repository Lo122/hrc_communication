"""Run realtime recognition in its own process and publish trigger events."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for layer in ["0_core", "1_recognition"]:
    sys.path.insert(0, str(ROOT / layer))

import config
from event_transport import UDPEventSender
from events import Event, EventType
from recognition_manager import DEFAULT_MODEL_DIR, RecognitionManager
from trigger_manager import TriggerManager

REALTIME_CAMERA_INDEX = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recognition and publish trigger events.")
    parser.add_argument("--camera", action="store_true", help="Use realtime camera input.")
    parser.add_argument("--video-source", default=None, help="Video path or stream URL. Defaults to config.test_vid_path.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing config.json and best_model.pth.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Communication event receiver host.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Communication event receiver port.")
    parser.add_argument("--no-display", action="store_true", help="Disable the recognition video preview window.")
    return parser.parse_args()


def _recognition_source(args: argparse.Namespace):
    if args.camera:
        return REALTIME_CAMERA_INDEX
    return args.video_source or config.test_vid_path


if __name__ == "__main__":
    args = _parse_args()
    source = _recognition_source(args)
    sender = UDPEventSender(args.host, args.port)
    recognition_manager = RecognitionManager(model_dir=args.model_dir, video_source=source, show_video=not args.no_display)
    trigger_manager = TriggerManager()

    print(f"Recognition running with source: {source}")
    print(f"Publishing recognition events to {args.host}:{args.port}")

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
                        },
                    ))

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Recognition stopped.")
    finally:
        recognition_manager.release()
        sender.close()
