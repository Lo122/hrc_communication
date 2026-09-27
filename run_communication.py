"""
    Run communication, CLI, TaskManager, and ROS without realtime recognition.
    Usage:
        uv run python run_communication.py [--host HOST] [--port PORT] [--demo]
          [--debug-trigger] [--debug-step-id STEP_ID]
          [--debug-progress PROGRESS] [--debug-round-id ROUND_ID]
          [--log-dir DIR] [--run-name NAME] [--no-run-log]

    --debug-trigger injects one HUMAN_TASK_UPDATE, as if recognition had seen the
    human on that step. Whether a robot task follows is up to the task database's
    trigger rules (e.g. step 0 = Pull Cables at progress >= 0.5 offers Lift).

    Every run is logged to logs/runs/<run name>/ (--log-dir, --run-name, --no-run-log):
    communication_events.jsonl holds every event, transition and message with its
    time; timeline.csv the same, readable, without the human-location stream.

    --demo opens with a scripted dialogue for the first panel -- the robot offers to
    pull the cables, then asks about the lift -- and only then hands over to
    recognition, the trigger rules and the task detectors
    (2_decision_making/demo_opening.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for layer in ["0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"]:
    sys.path.insert(0, str(ROOT / layer))

import config
from event_transport import UDPEventReceiver
from events import Event, EventType


def default_run_name() -> str:
    """run_<date>_<time>, as run_recognition.py names its runs."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run communication and receive recognition events.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Host to bind for recognition events.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Port to bind for recognition events.")
    parser.add_argument("--demo", action="store_true", help="Open with the scripted demo dialogue for the first panel.")
    parser.add_argument("--debug-trigger", action="store_true", help="Inject one fake human task update for communication debugging.")
    parser.add_argument("--debug-step-id", type=int, default=0, help="Human step id (config.STEP_NAMES index) for --debug-trigger; not a robot task id.")
    parser.add_argument("--debug-progress", type=float, default=1.0, help="Progress value for --debug-trigger.")
    parser.add_argument("--debug-round-id", type=int, default=0, help="Round id for --debug-trigger.")
    parser.add_argument("--log-dir", default=str(ROOT / config.RUN_LOG_DIR),
                        help="Log this run to <LOG_DIR>/<RUN_NAME>/: communication_events.jsonl and timeline.csv.")
    parser.add_argument("--run-name", default=None,
                        help="Run directory name; defaults to run_<date>_<time>. run_system.py gives "
                             "recognition the same one.")
    parser.add_argument("--no-run-log", action="store_true",
                        help=f"Append to {config.LOG_FILE_PATH} instead of a run directory.")
    args = parser.parse_args()
    if args.debug_trigger and not 0 <= args.debug_step_id < len(config.STEP_NAMES):
        steps = ", ".join(f"{index}={name}" for index, name in enumerate(config.STEP_NAMES))
        parser.error(f"Human step {args.debug_step_id} does not exist. Steps: {steps}.")
    return args

#python run_communication.py --debug-trigger
if __name__ == "__main__":
    args = _parse_args()
    from communication_runtime import build_system

    receiver = UDPEventReceiver(args.host, args.port)
    run_dir = None if args.no_run_log else Path(args.log_dir) / (args.run_name or default_run_name())
    system = build_system(demo=args.demo, run_dir=run_dir)
    if run_dir is not None:
        print(f"Logging this run to {run_dir}")
    system.system_running = True
    system.start_cli_thread()

    print(f"Communication running. Listening for recognition events on {args.host}:{args.port}")
    if args.demo:
        system.start_demo()
        when = (f"once recognition has run for {config.RECOGNITION_ACTIVATION_S:g} s"
                if config.RECOGNITION_ACTIVATION_S > 0 else "now")
        print(f"HRC communication started in demo mode: asking to pull the first panel's cables {when}.")
    else:
        print("HRC communication started. Waiting for recognition trigger...")

    if args.debug_trigger:
        system.event_queue.put(
            Event(
                event_type=EventType.HUMAN_TASK_UPDATE,
                source="debug_recognition",
                payload={
                    "step_id": args.debug_step_id,
                    "round_id": args.debug_round_id,
                    "progress": args.debug_progress,
                },
            )
        )
        print(f"Injected debug task update for step {args.debug_step_id} "
              f"({config.STEP_NAMES[args.debug_step_id]}).")

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
