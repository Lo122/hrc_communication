"""Coordinates CLI output, TTS playback, and state-driven voice input."""

from enum import Enum, auto
from queue import Empty, SimpleQueue
import time

from events import RobotTaskState
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
    RobotTaskState.R_WAITING_HOME_PERMISSION: ListeningMode.SINGLE,
    RobotTaskState.R_MANUAL_RECOVERY: ListeningMode.CONTINUOUS,
}


class CommunicationManager:
    def __init__(self, cli, parser, voice, tts, event_sink, state_provider,
                 guard_seconds=0.25, max_attempts=2, retry_seconds=5.0, logger=None,
                 context_provider=None):
        self.cli = cli
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
        self._question_context = None
        self._question = None

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
        self._question = message
        self._announce(message, permission=True, speech=speech)

    def _announce(self, message: str, permission=False, *, speech: str | None = None) -> None:
        if self._closed:
            return
        self._generation += 1
        self._retry_at = None
        self.voice.stop_listening()
        output = self.cli.show_permission_request if permission else self.cli.show_message
        output(message)
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
        if context == self._context and not force:
            return
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
        self.mode = STATE_MODES.get(state, ListeningMode.OFF)
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
                event = self.parser.parse(detail, source="human_voice")
                if event is not None:
                    event.task_instance_id = self._context.task_instance_id
                    self._errors = 0
                    self._error_notified = False
                    self.event_sink(event)
                    # Let TaskManager consume the command before listening again.
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
                        f"Please say {choices}, or type your reply.", permission=True,
                        speech=f"Please say {choices}.",
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
