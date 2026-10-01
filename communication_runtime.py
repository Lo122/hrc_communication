"""Communication system wiring for the HRC runtime."""

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
from communication_manager import CommunicationManager
from decision_view import DecisionView, TimelineLogger, build_snapshot
from demo_opening import build_demo_opening
from event_queue import EventQueue
from events import Event, EventType, RobotTaskState
from gh_dispatcher import GHDispatcher
from logger import EventLogger
from message_manager import MessageManager
from tts_manager import NullTTSManager, TTSManager
from voice_interface import NullVoiceInterface, VoiceInterface
from voice_context import VoiceContext
from pending_task import PendingTaskPool
from reactive_task_manager import ReactiveTaskManager
from ros_communication import ROSCommunication
from state_machine import StateMachine
from task_manager import RECOGNITION_SOURCE, TaskManager
from task_tracker import build_task_tracking
from task_transition_detector import DetectorContext, build_detectors, signal_event, signal_payload
from timer_manager import TimerManager
from udp_sender import UDPSender

RECOGNITION_EVENTS = (EventType.HUMAN_TASK_UPDATE, EventType.RECOGNITION_TRIGGER)
# The step the detectors see when no task the sequence allows fits the scores.
IDLE_STEP = "Non Related Task"
# What a run directory gets from the communication side (run_recognition.py writes
# its frames.csv / events.csv / run.json / run.log beside them).
COMMUNICATION_LOG = "communication_events.jsonl"
TIMELINE_LOG = "timeline.csv"


class HRCSystem:
    """Wires communication, decision making, and execution around one event queue.

    demo: the first panel opens with a scripted dialogue (demo_opening.py) instead of
    recognition -- start it with start_demo() -- and the task detectors run in
    config.DEMO_DETECTORS_MODE.
    reactive: the robot offers nothing and acts only on the human's commands
    (reactive_task_manager.py). It runs without recognition and without the task
    detectors (no camera, no TCP force): the task tracker follows the robot's tasks and
    what the human says, so a command goes to the right piece. Voice input listens while
    the robot is idle, too.
    run_dir: log this run there -- communication_events.jsonl (every event, transition
    and message) and timeline.csv (the same, readable) -- instead of appending to
    config.LOG_FILE_PATH."""

    def __init__(self, demo: bool = False, run_dir: str | Path | None = None, reactive: bool = False):
        if demo and reactive:
            raise ValueError("The demo opening is proactive: it cannot run in reactive mode.")
        self.reactive = reactive
        if demo:
            # Read by the detectors, the ROS wrench subscription and the decision view.
            config.TASK_DETECTORS_MODE = config.DEMO_DETECTORS_MODE
        if reactive:
            # No camera and no TCP force: no detectors, no wrench subscription.
            config.TASK_DETECTORS_MODE = "off"
        self.event_queue = EventQueue()

        self.run_dir = Path(run_dir) if run_dir is not None else None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.logger = EventLogger(self.run_dir / COMMUNICATION_LOG,
                                      timeline_path=self.run_dir / TIMELINE_LOG)
        else:
            self.logger = EventLogger(config.LOG_FILE_PATH)
        if config.DECISION_VIEW_PORT:
            # Everything still goes to the log file; the view also shows the latest lines.
            self.logger = TimelineLogger(self.logger)
        self.command_parser = CommandParser()
        self.cli = CLIInterface(self.command_parser)
        self.message_manager = MessageManager(reactive=reactive)

        wake_word = config.VOICE_WAKE_WORD and not config.VOICE_GPT_ENABLED
        if config.VOICE_ENABLED:
            self.voice = VoiceInterface(
                model_path=ROOT / config.VOICE_MODEL_PATH,
                gpt_enabled=config.VOICE_GPT_ENABLED,
                phrases=CommandParser.voice_phrases(wake_word),
                device_name=config.VOICE_INPUT_DEVICE_NAME,
                output_device_name=config.VOICE_OUTPUT_DEVICE_NAME,
                timeout=config.VOICE_LISTEN_TIMEOUT_SECONDS,
            )
            self.tts = TTSManager(config.VOICE_TTS_RATE, voice=config.VOICE_TTS_VOICE)
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
            context_provider=self._current_voice_context,
            guard_seconds=config.VOICE_POST_TTS_GUARD_SECONDS,
            max_attempts=config.VOICE_MAX_ATTEMPTS,
            retry_seconds=config.VOICE_ERROR_RETRY_SECONDS,
            logger=self.logger,
            wake_word=wake_word,
            wake_window_seconds=config.VOICE_WAKE_WINDOW_S,
        )

        self.udp_sender = UDPSender(config.UDP_HOST, config.UDP_PORT)
        self.gh_dispatcher = GHDispatcher(self.udp_sender)
        detecting = config.TASK_DETECTORS_MODE != "off"
        self.ros = ROSCommunication(
            wrench_topic=config.ROS_TOPICS[config.FORCE_WRENCH_TOPIC] if detecting else None)
        self.ros.set_event_callback(self.event_queue.put)
        self.detectors = build_detectors() if detecting else None
        self._recognition: list[tuple[float, str, float]] = []  # for the detectors' next pass

        self.timer_manager = TimerManager(event_callback=self.event_queue.put)
        self.pending_pool = PendingTaskPool()
        self.state_machine = StateMachine()
        self.task_tracker, self.trigger_policy = build_task_tracking(ROOT, logger=self.logger)
        if reactive:
            self.trigger_policy = None  # nothing is offered: the tracker only finds the piece
        self.task_manager = (ReactiveTaskManager if reactive else TaskManager)(
            state_machine=self.state_machine,
            pending_pool=self.pending_pool,
            timer_manager=self.timer_manager,
            message_manager=self.message_manager,
            cli=self.communication,
            gh_dispatcher=self.gh_dispatcher,
            ros_communication=self.ros,
            logger=self.logger,
            task_tracker=self.task_tracker,
            trigger_policy=self.trigger_policy,
            status_callback=lambda line: print(line, flush=True),
            demo=build_demo_opening() if demo else None,
            recognition_activation_s=config.RECOGNITION_ACTIVATION_S,
        )

        self.system_running = False
        # Last, so a failure above never leaves its port taken.
        self.decision_view = (DecisionView.start(config.DECISION_VIEW_HOST, config.DECISION_VIEW_PORT)
                              if config.DECISION_VIEW_PORT else None)
        if self.decision_view is not None:
            # The page's manual control: applied by TaskManager between the other events.
            self.decision_view.set_submit(lambda payload: self.event_queue.put(
                Event(EventType.MANUAL_CONTROL, "decision_view", payload=payload)))

    def start_demo(self) -> None:
        """Open the demo: the robot offers to pull the first panel's cables -- once
        recognition is active (config.RECOGNITION_ACTIVATION_S)."""
        self.event_queue.put(Event(EventType.DEMO_START, "demo"))

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
                self.communication.queue_message("Command not recognized.")

    def _current_state(self):
        manager = getattr(self, "task_manager", None)
        if manager is None:
            return None
        if manager.active_task is not None:
            return manager.active_task.state
        # A question outside any robot task (the demo opening's) waits for a yes / no too.
        return RobotTaskState.R_WAITING_RESPONSE if manager.question is not None else None

    def _current_voice_context(self) -> VoiceContext:
        manager = getattr(self, "task_manager", None)
        task = manager.active_task if manager else None
        reactive = getattr(self, "reactive", False)
        if task is None:
            if manager is not None and manager.question is not None:
                return VoiceContext(RobotTaskState.R_WAITING_RESPONSE,
                                    task_instance_id=f"question {manager.question}", reactive=reactive)
            return VoiceContext(None, reactive=reactive)
        return VoiceContext(task.state, task.task_id, task.task_instance_id, reactive=reactive)

    def process_events(self) -> None:
        """Run the task detectors, then handle all queued recognition, CLI, timer,
        ROS and detector events."""
        self._run_detectors()
        while not self.event_queue.empty():
            event = self.event_queue.get()
            self.task_manager.handle_event(event)
            self._note_recognition(event)  # after: the step TaskManager chose for it
            self.communication.sync_state(self._current_state())
        self.communication.poll()
        view = getattr(self, "decision_view", None)
        if view is not None:
            view.update(lambda: build_snapshot(self.task_manager, timeline=self.logger,
                                               detectors=self.detectors))

    def _run_detectors(self) -> None:
        """One pass of the sensor-based task detectors (task_transition_detector.py).
        With TASK_DETECTORS_MODE "log" their signals are only logged."""
        detectors = getattr(self, "detectors", None)
        if detectors is None:
            return
        manager = self.task_manager
        recognition, self._recognition = getattr(self, "_recognition", []), []
        model_steps, self._model_steps = getattr(self, "_model_steps", []), []
        context = DetectorContext(time.time(), manager.active_task, manager.held_piece_id,
                                  manager.tracker, self.ros.drain_wrench(), recognition,
                                  model_steps)
        for signal in detectors.update(context):
            if config.TASK_DETECTORS_MODE == "on":
                self.event_queue.put(signal_event(signal))
            else:
                self.logger.log_message("Task detector signal (log only).",
                                        {"source": signal.source, **signal_payload(signal)})

    def _note_recognition(self, event) -> None:
        """Keep recognition's (task, progress) stream for the detectors' next pass: the
        step TaskManager chose for this update against the task sequence, and the model's
        own step. They read it even while TaskManager ignores recognition for a robot
        lift -- but not while recognition's model still warms up."""
        if getattr(self, "detectors", None) is None or event.event_type not in RECOGNITION_EVENTS:
            return
        manager = getattr(self, "task_manager", None)
        if event.source == RECOGNITION_SOURCE and manager is not None and not manager.recognition_active:
            return
        step = event.payload.get("step_id")
        if not (isinstance(step, int) and 0 <= step < len(config.STEP_NAMES)):
            return
        model_task = config.STEP_NAMES[step]
        progress = float(event.payload.get("progress") or 0.0)
        choice = getattr(manager, "last_recognition", None)
        if choice is not None:
            # No task the sequence allows fits: the human is on none, as far as the
            # detectors go (a climb under way ends as "recognition moved on").
            task_name = choice.task_name if choice.task_name is not None else IDLE_STEP
            progress = choice.progress if choice.task_name is not None else 0.0
        else:
            task_name = model_task
        self._recognition = getattr(self, "_recognition", [])
        self._recognition.append((event.timestamp, task_name, progress))
        self._model_steps = getattr(self, "_model_steps", [])
        self._model_steps.append((event.timestamp, model_task))

    def close(self) -> None:
        """Release communication resources."""
        self.system_running = False
        if getattr(self, "decision_view", None) is not None:
            self.decision_view.close()
        self.communication.close()
        self.ros.close()
        self.udp_sender.close()
        self.logger.close()


def build_system(demo: bool = False, run_dir: str | Path | None = None, reactive: bool = False) -> HRCSystem:
    """Build the communication-side HRC runtime."""
    return HRCSystem(demo=demo, run_dir=run_dir, reactive=reactive)
