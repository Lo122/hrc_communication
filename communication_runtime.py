"""Communication system wiring for the HRC runtime."""

import sys
import threading
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
from events import Event, EventType
from cmd_parser import CommandParser
from cli_interface import CLIInterface
from communication_manager import CommunicationManager
from event_queue import EventQueue
from gh_dispatcher import GHDispatcher
from logger import EventLogger
from message_manager import MessageManager
from tts_manager import NullTTSManager, TTSManager
from voice_interface import NullVoiceInterface, VoiceInterface
from pending_task import PendingTaskPool
from ros_communication import ROSCommunication
from state_machine import StateMachine
from task_manager import TaskManager
from timer_manager import TimerManager
from udp_sender import UDPSender


class HRCSystem:
    """Wires communication, decision making, and execution around one event queue."""

    def __init__(self):
        self.event_queue = EventQueue()

        self.logger = EventLogger(config.LOG_FILE_PATH)
        self.command_parser = CommandParser()
        self.cli = CLIInterface(self.command_parser)
        self.message_manager = MessageManager()

        if config.VOICE_ENABLED:
            self.voice = VoiceInterface(
                model_path=ROOT / config.VOICE_MODEL_PATH,
                gpt_enabled=config.VOICE_GPT_ENABLED,
                phrases=CommandParser._ALIASES,
                device_name=config.VOICE_INPUT_DEVICE_NAME,
                timeout=config.VOICE_LISTEN_TIMEOUT_SECONDS,
            )
            self.tts = TTSManager(config.VOICE_TTS_RATE)
        else:
            self.voice = NullVoiceInterface()
            self.tts = NullTTSManager()

        self.communication = CommunicationManager(
            cli=self.cli,
            parser=self.command_parser,
            voice=self.voice,
            tts=self.tts,
            event_sink=self.event_queue.put,
            state_provider=self._current_state,
            guard_seconds=config.VOICE_POST_TTS_GUARD_SECONDS,
            max_attempts=config.VOICE_MAX_ATTEMPTS,
        )

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
            cli=self.communication,
            gh_dispatcher=self.gh_dispatcher,
            ros_communication=self.ros,
            logger=self.logger,
        )

        self.system_running = False

    def start_cli_thread(self) -> None:
        """Start a background CLI event producer."""
        thread = threading.Thread(target=self._cli_loop, daemon=True)
        thread.start()

    def _cli_loop(self) -> None:
        """Read CLI commands and publish parsed events."""
        while self.system_running:
            raw_text = self.cli.read_raw()

            # Questions are answered here and never reach the event queue --
            # there is no state to transition to, the human just wants to
            # know something. Checked first, because the command parser
            # would otherwise reject them as unrecognized.
            if self._answer_query(raw_text):
                continue

            event = self.cli.parse_command(raw_text)
            if event is not None:
                self.event_queue.put(event)
            else:
                self.communication.show_message("Command not recognized.")

    def _answer_query(self, raw_text: str) -> bool:
        """Speak an answer to a question. True if it was one."""
        query = self.command_parser.parse_query(raw_text)
        if query is None:
            return False

        task = self.task_manager.active_task

        if query == "status":
            self.communication.show_message(
                self.message_manager.get_status_message(task)
            )
        elif query == "why":
            self.communication.show_message(
                self.message_manager.get_proposal_reason(task)
            )
        elif query == "queue":
            self.communication.show_message(
                self.message_manager.get_pending_summary(self.pending_pool.list_all())
            )
        elif query == "repeat":
            self.communication.show_message(
                self.message_manager.get_repeat_message(self.communication.last_message)
            )
        elif query == "run_pending":
            pending = self.pending_pool.list_all()
            if not pending:
                self.communication.show_message("Nothing is queued.")
            else:
                self.event_queue.put(
                    Event(
                        event_type=EventType.H_EXECUTE_PENDING_TASK,
                        source="human_cli",
                        task_instance_id=pending[0].task_instance_id,
                    )
                )
        return True

    def _current_state(self):
        task = getattr(self, "task_manager", None)
        return task.active_task.state if task and task.active_task else None

    def process_events(self) -> None:
        """Handle all queued recognition, CLI, timer, and ROS events."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            self.task_manager.handle_event(event)
            self.communication.sync_state(self._current_state())

    def close(self) -> None:
        """Release communication resources."""
        self.system_running = False
        self.communication.close()
        self.ros.close()
        self.udp_sender.close()


def build_system() -> HRCSystem:
    """Build the communication-side HRC runtime."""
    return HRCSystem()
