"""Run communication, CLI, TaskManager, and ROS without realtime recognition."""

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
from communication_runtime import build_system


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run communication and receive recognition events.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Host to bind for recognition events.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Port to bind for recognition events.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    receiver = UDPEventReceiver(args.host, args.port)
    system = build_system()
    system.system_running = True
    system.start_cli_thread()

    print(f"Communication running. Listening for recognition events on {args.host}:{args.port}")

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
