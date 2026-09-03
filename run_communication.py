"""
    Run communication, CLI, TaskManager, and ROS without realtime recognition.
    Usage:
        uv run python run_communication.py [--host HOST] [--port PORT]
          [--debug-trigger] [--debug-step-id STEP_ID]
          [--debug-progress PROGRESS] [--debug-round-id ROUND_ID]
          [--debug-piece-id PIECE_ID]     
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for layer in ["0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"]:
    sys.path.insert(0, str(ROOT / layer))

import config
from event_transport import UDPEventReceiver
from events import Event, EventType
from communication_runtime import build_system


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run communication and receive recognition events.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Host to bind for recognition events.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Port to bind for recognition events.")
    parser.add_argument("--debug-trigger", action="store_true", help="Inject one fake recognition trigger for communication debugging.")
    parser.add_argument("--debug-step-id", type=int, default=0, help="Step id for --debug-trigger.")
    parser.add_argument("--debug-progress", type=float, default=1.0, help="Progress value for --debug-trigger.")
    parser.add_argument("--debug-round-id", type=int, default=0, help="Round id for --debug-trigger.")
    parser.add_argument("--debug-piece-id", type=int, default=1, help="Piece id for --debug-trigger.")
    return parser.parse_args()

#python run_communication.py --debug-trigger
if __name__ == "__main__":
    args = _parse_args()
    receiver = UDPEventReceiver(args.host, args.port)
    system = build_system()
    system.system_running = True
    system.start_cli_thread()

    print(f"Communication running. Listening for recognition events on {args.host}:{args.port}")
    print("HRC communication started. Waiting for recognition trigger...")

    if args.debug_trigger:
        system.event_queue.put(
            Event(
                event_type=EventType.RECOGNITION_TRIGGER,
                source="debug_recognition",
                payload={
                    "step_id": args.debug_step_id,
                    "piece_id": args.debug_piece_id,
                    "round_id": args.debug_round_id,
                    "progress": args.debug_progress,
                },
            )
        )
        print(f"Injected debug recognition trigger for step {args.debug_step_id}.")

    try:
        while system.system_running:
            for event in receiver.poll():
                
                if (
                    event.event_type == EventType.RECOGNITION_TRIGGER
                    and system.task_manager.active_task is not None
                ):
                    continue

                system.event_queue.put(event)
            system.process_events()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Communication stopped.")
    finally:
        receiver.close()
        system.close()
