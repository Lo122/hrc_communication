"""Coordinates CLI output, TTS playback, and state-driven voice input."""

from enum import Enum, auto
from queue import Empty, SimpleQueue
import time

from events import EventType, RobotTaskState
from voice_result import VoiceOutcome
from voice_context import VoiceContext, build_instructions


class ListeningMode(Enum):
    OFF = auto()
    SINGLE = auto()
    CONTINUOUS = auto()


STATE_MODES = {
    RobotTaskState.R_WAITING_RESPONSE: ListeningMode.SINGLE,
    RobotTaskState.R_DEFER: ListeningMode.CONTINUOUS,
    RobotTaskState.R_EXECUTING: ListeningMode.CONTINUOUS,
    RobotTaskState.R_PAUSED: ListeningMode.CONTINUOUS,
    RobotTaskState.R_WAITING_FREE_DRIVE: ListeningMode.SINGLE,
    RobotTaskState.R_FREE_DRIVE: ListeningMode.CONTINUOUS,
    RobotTaskState.R_HOLDING: ListeningMode.CONTINUOUS,
    RobotTaskState.R_WAITING_HANDOVER: ListeningMode.SINGLE,
    RobotTaskState.R_HOLDING_HANDOVER: ListeningMode.CONTINUOUS,
    RobotTaskState.R_WAITING_HOME_PERMISSION: ListeningMode.SINGLE,
    RobotTaskState.R_MANUAL_RECOVERY: ListeningMode.CONTINUOUS,
}


# Idle, after the human took a task over ("I'll do it"): the robot waits for their done.
HUMAN_TURN_REPLIES = {EventType.H_DONE, EventType.H_TASK_DONE}


def listening_mode(context: VoiceContext, wake_word: bool = False) -> ListeningMode:
    """How to listen in this context. With no robot task it listens in reactive mode --
    the human commands the robot from idle there -- and with the wake word, since only
    "hey UR, ..." counts then; otherwise the robot asks first."""
    if context.state is None and (context.reactive or wake_word):
        return ListeningMode.CONTINUOUS
    return STATE_MODES.get(context.state, ListeningMode.OFF)


class CommunicationManager:
    def __init__(self, cli, parser, voice, tts, event_sink, state_provider,
                 guard_seconds=0.25, max_attempts=2, retry_seconds=5.0, logger=None,
                 context_provider=None, wake_word=False, wake_window_seconds=5.0):
        self.cli = cli
        # wake_word: the robot's name ("hey UR, ...") is needed only to start something
        # else -- while it is idle, or to ask for another task while it works. What it
        # is busy with needs none: an answer to its question, "faster" or "stop" while it
        # moves, the "done" it waits for (_needs_name). The name alone opens a window of
        # wake_window_seconds for the command.
        self.wake_word = wake_word
        self.wake_window_seconds = wake_window_seconds
        self._woken_until = 0.0
        self.parser = parser
        self.voice = voice
        self.tts = tts
        self.event_sink = event_sink
        self.state_provider = state_provider
        self.context_provider = context_provider
        self.guard_seconds = guard_seconds
        self.max_attempts = max_attempts
        self.retry_seconds = retry_seconds
        self.logger = logger
        self.mode = ListeningMode.OFF
        self._state = None
        self._attempts = 0
        self._generation = 0
        self._results = SimpleQueue()
        self._retry_at = None
        self._errors = 0
        self._error_notified = False
        self._closed = False
        self._context = VoiceContext(None)
        self._synced = False  # the first sync starts listening even in the default context
        self._question_context = None
        self._question = None
        self._question_speech = None
        # The last line said, for "repeat" when no question is open: (message, speech).
        self._last_said = None

    def _addressed(self, text: str) -> str | None:
        """The command in a recognized utterance; "" for the name alone, None for speech
        not meant for the robot."""
        if not self.wake_word:
            return text
        command = self.parser.addressed(text)
        if command == "":
            self._woken_until = time.monotonic() + self.wake_window_seconds
            return ""
        if command is None and time.monotonic() < self._woken_until:
            command = text  # the name came just before, on its own
        if command is None and not self._needs_name(text):
            command = text
        if command is not None:
            self._woken_until = 0.0
        return command

    def _needs_name(self, text: str) -> bool:
        """Said without the robot's name, is this not meant for it? While it is idle,
        anything but the done it waits for. While it works on a task, only a request for
        another task: answers and controls of the task under way ("yes", "faster",
        "stop", "done") need no name, and neither does what is not understood (so the
        robot can say it did not catch an answer)."""
        event = self.parser.parse(text, source="human_voice", state=self._state)
        if self._state is None:
            return not (self._context.human_turn and event is not None
                        and event.event_type in HUMAN_TURN_REPLIES)
        return event is not None and event.event_type == EventType.H_REQUEST_ROBOT_TASK

    def _repeat(self) -> None:
        """Say the open question again -- or, with none open, the last line."""
        if self._question is not None and self._question_context == self._context:
            self._attempts = 0
            self._announce(self._question, permission=True, speech=self._question_speech)
        elif self._last_said is not None:
            message, speech = self._last_said
            self._announce(message, speech=speech)
        else:
            self._retry_at = time.monotonic()

    def _current_context(self, state) -> VoiceContext:
        return self.context_provider() if self.context_provider else VoiceContext(state)

    def queue_message(self, message: str) -> None:
        """Let CLI workers request output without touching the audio worker."""
        self._results.put((None, "message", message))

    def show_message(self, message: str, *, speech: str | None = None) -> None:
        self._announce(message, speech=speech)

    def show_permission_request(self, message: str, *, speech: str | None = None) -> None:
        self._attempts = 0
        self._question_context = self._current_context(self.state_provider())
        self._question, self._question_speech = message, speech
        self._announce(message, permission=True, speech=speech)

    def _announce(self, message: str, permission=False, *, speech: str | None = None) -> None:
        if self._closed:
            return
        self._generation += 1
        self._retry_at = None
        self.voice.stop_listening()
        output = self.cli.show_permission_request if permission else self.cli.show_message
        output(message)
        self._last_said = (message, speech)
        self.tts.speak(message if speech is None else speech)
        time.sleep(self.guard_seconds)
        state = self.state_provider()
        context = self._current_context(state)
        new_question = context != self._context and STATE_MODES.get(state) is ListeningMode.SINGLE
        if new_question:
            self._question_context, self._question = context, message
        beep = permission or new_question
        self.sync_state(state, force=True, beep=beep)

    def sync_state(self, state, force=False, beep=False) -> None:
        if self._closed:
            return
        context = self._current_context(state)
        if context == self._context and not force and self._synced:
            return
        self._synced = True
        if context != self._context:
            self._attempts = 0
            if self.logger is not None:
                self.logger.log_message("Voice context changed.", {
                    "task_instance_id": context.task_instance_id,
                    "task_id": context.task_id,
                    "state": context.state.name if context.state else None,
                })
        self._context = context
        self._state = state
        self.mode = listening_mode(context, self.wake_word)
        self._generation += 1
        self._retry_at = None
        self.voice.stop_listening()
        if self.mode is not ListeningMode.OFF:
            if self._errors:
                self._retry_at = time.monotonic() + self.retry_seconds
            else:
                self._start_listening(beep=beep)

    def _start_listening(self, beep=False) -> None:
        self._generation += 1
        generation = self._generation
        self.voice.start_listening(
            lambda text: self._on_voice_text(text, generation),
            lambda outcome, detail="": self._on_voice_failure(generation, outcome, detail),
            beep=beep,
            instructions=build_instructions(
                self._context, self._question if self._question_context == self._context else None,
            ),
        )

    def _on_voice_text(self, text: str, generation: int) -> None:
        self._results.put((generation, "text", text))

    def _on_voice_failure(self, generation: int, outcome: VoiceOutcome, detail="") -> None:
        self._results.put((generation, outcome, detail))

    def poll(self) -> None:
        """Process worker results and restart listening on the runtime thread."""
        if self._closed:
            return
        self.sync_state(self.state_provider())
        while True:
            try:
                generation, outcome, detail = self._results.get_nowait()
            except Empty:
                break
            if outcome == "message":
                self._announce(detail)
                continue
            if generation != self._generation:
                continue
            self._generation += 1
            if outcome == "text":
                command = self._addressed(detail)
                if not command:
                    # Not meant for the robot, or its name alone with the command to
                    # follow: keep listening, without a word.
                    if self.logger is not None:
                        self.logger.log_message("Voice input not addressed to the robot." if command is None
                                                else "Voice wake word heard.", {"text": detail})
                    self._retry_at = time.monotonic()
                    continue
                if self.parser.is_repeat(command):
                    self._repeat()
                    continue
                event = self.parser.parse(command, source="human_voice", state=self._state)
                if event is not None:
                    event.task_instance_id = self._context.task_instance_id
                    self._errors = 0
                    self._error_notified = False
                    self.event_sink(event)
                    # Let TaskManager consume the command before listening again: a new
                    # state restarts listening itself; if nothing changed, the next poll does.
                    self._retry_at = time.monotonic()
                    return
                outcome = VoiceOutcome.UNRECOGNIZED
                detail = "Recognized text did not match a command"
            if self.logger is not None:
                self.logger.log_message("Voice input result.", {
                    "outcome": outcome.name, "detail": detail,
                    "state": self._state.name if self._state else None,
                })
            if outcome is VoiceOutcome.ERROR:
                self._errors += 1
                if self._errors >= 2 and not self._error_notified:
                    self._error_notified = True
                    self._announce(
                        "Voice input is temporarily unavailable. Please type your commands.",
                        speech="Voice temporarily unavailable. Please type.",
                    )
                self._retry_at = time.monotonic() + self.retry_seconds
                continue
            self._errors = 0
            self._error_notified = False
            if outcome is VoiceOutcome.UNRECOGNIZED and self.mode is ListeningMode.SINGLE:
                self._attempts += 1
                if self._attempts == 1 and self.max_attempts > 1:
                    choices = "yes, no, or later" if self._state == RobotTaskState.R_WAITING_RESPONSE else "yes or no"
                    self._announce(
                        f"Sorry, I did not catch that. Please say {choices}, or type your reply.",
                        permission=True, speech=f"Sorry, I did not catch that. Please say {choices}.",
                    )
                    continue
            self._retry_at = time.monotonic()
        if self._retry_at is not None and time.monotonic() >= self._retry_at:
            self._retry_at = None
            if self.mode is not ListeningMode.OFF:
                self._start_listening()

    def close(self) -> None:
        self._closed = True
        self._generation += 1
        self._retry_at = None
        self.voice.close()
        self.tts.close()
