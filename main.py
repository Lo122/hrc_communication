"""Application bootstrap for the HRC communication system."""

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for layer in [
    "0_core",
    "1_recognition",
    "2_decision_making",
    "3_communication",
    "4_execution",
]:
    sys.path.insert(0, str(ROOT / layer))

import config
from cmd_parser import CommandParser
from cli_interface import CLIInterface
from event_queue import EventQueue
from gh_dispatcher import GHDispatcher
from logger import EventLogger
from message_manager import MessageManager
from pending_task import PendingTaskPool
from recognition_manager import DEFAULT_MODEL_DIR, RecognitionManager
from ros_communication import ROSCommunication
from state_machine import StateMachine
from step_stablizier import StepStabilizer
from task_manager import TaskManager
from timer_manager import TimerManager
from trigger_manager import TriggerManager
from udp_sender import UDPSender


REALTIME_CAMERA_INDEX = 0


class HRCSystem:
    """Wires all layers together around one shared event queue."""

    def __init__(
        self,
        *,
        recognition_video_source=None,
        recognition_model_dir: str | Path = DEFAULT_MODEL_DIR,
    ):
        self.event_queue = EventQueue()

        self.logger = EventLogger(config.LOG_FILE_PATH)
        self.command_parser = CommandParser()
        self.cli = CLIInterface(self.command_parser)
        self.message_manager = MessageManager()

        self.udp_sender = UDPSender(config.UDP_HOST, config.UDP_PORT)
        self.gh_dispatcher = GHDispatcher(self.udp_sender)
        self.ros = ROSCommunication()
        self.ros.set_event_callback(self.event_queue.put)

        self.timer_manager = TimerManager(event_callback=self.event_queue.put)
        self.pending_pool = PendingTaskPool()
        self.state_machine = StateMachine()
        self.task_manager = TaskManager(
            state_machine=self.state_machine,
            pending_pool=self.pending_pool,
            timer_manager=self.timer_manager,
            message_manager=self.message_manager,
            cli=self.cli,
            gh_dispatcher=self.gh_dispatcher,
            ros_communication=self.ros,
            logger=self.logger,
        )

        self.step_stabilizer = StepStabilizer()
        self.recognition_manager = RecognitionManager(
            self.step_stabilizer,
            model_dir=recognition_model_dir,
            video_source=recognition_video_source,
        )
        self.trigger_manager = TriggerManager()
        self.system_running = False

    def start_cli_thread(self) -> None:
        """Start a background CLI event producer."""
        thread = threading.Thread(target=self._cli_loop, daemon=True)
        thread.start()

    def _cli_loop(self) -> None:
        """Read CLI commands and publish parsed events."""
        while self.system_running:
            event = self.cli.read_input()
            if event is not None:
                self.event_queue.put(event)
            else:
                self.cli.show_message("Command not recognized.")

    def run_once(self, recognition_input=None) -> None:
        """Run one integration tick for simulations or live recognition."""
        result = self.recognition_manager.update(recognition_input)
        if result is not None:
            for event in self.trigger_manager.update(result):
                self.event_queue.put(event)

        while not self.event_queue.empty():
            event = self.event_queue.get()
            self.task_manager.handle_event(event)

    def run_forever(self) -> None:
        """Run the integrated system with RecognitionManager as the input source."""
        self.system_running = True
        self.start_cli_thread()

        try:
            while self.system_running:
                self.run_once()
                time.sleep(0.05)
        finally:
            self.system_running = False
            self.recognition_manager.release()
            self.ros.close()
            self.udp_sender.close()


def build_system(
    *,
    recognition_video_source=None,
    recognition_model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> HRCSystem:
    """Factory used by tests, manual simulations, or the CLI entry point."""
    return HRCSystem(
        recognition_video_source=recognition_video_source,
        recognition_model_dir=recognition_model_dir,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HRC communication system.")
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use realtime camera input instead of config.test_vid_path.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build the system without starting the live recognition loop.",
    )
    return parser.parse_args()


def _recognition_source(use_camera: bool):
    if use_camera:
        return REALTIME_CAMERA_INDEX
    return config.test_vid_path


if __name__ == "__main__":
    args = _parse_args()
    video_source = _recognition_source(args.camera)
    system = build_system(recognition_video_source=video_source)

    if args.init_only:
        print("HRC system initialized. Use build_system().run_once(...) for simulation.")
    else:
        source_name = "realtime camera" if args.camera else "configured test video"
        print(f"HRC system running with {source_name}: {video_source}")
        system.run_forever()
