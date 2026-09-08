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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run communication and receive recognition events.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Host to bind for recognition events.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Port to bind for recognition events.")
    parser.add_argument("--debug-trigger", action="store_true", help="Inject one fake recognition trigger for communication debugging.")
    parser.add_argument("--debug-step-id", type=int, default=4, help="Human step id for --debug-trigger (configured: 0, 4, 5); not a robot task id.")
    parser.add_argument("--debug-progress", type=float, default=1.0, help="Progress value for --debug-trigger.")
    parser.add_argument("--debug-round-id", type=int, default=0, help="Round id for --debug-trigger.")
    parser.add_argument("--debug-piece-id", type=int, default=1, help="Piece id for --debug-trigger.")
    args = parser.parse_args()
    if args.debug_trigger and args.debug_step_id not in config.TRIGGER_RULES:
        configured = ", ".join(str(step) for step in sorted(config.TRIGGER_RULES))
        parser.error(
            f"Human step {args.debug_step_id} has no recognition trigger. "
            f"Configured human steps: {configured}. "
            "H3 is command-only: start with --debug-step-id 0, complete the lift "
            "and adjustment, then say or type 'screw done' while holding to request R2. "
            "R3 is requested only after R2 completes."
        )
    return args

#python run_communication.py --debug-trigger
if __name__ == "__main__":
    args = _parse_args()
    from communication_runtime import build_system

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
                system.event_queue.put(event)
            system.process_events()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Communication stopped.")
    finally:
        receiver.close()
        system.close()
