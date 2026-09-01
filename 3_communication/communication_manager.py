"""Coordinates CLI output, TTS playback, and state-driven voice input."""

from enum import Enum, auto
import time

from events import RobotTaskState


class ListeningMode(Enum):
    OFF = auto()
    SINGLE = auto()
    CONTINUOUS = auto()


STATE_MODES = {
    RobotTaskState.R_WAITING_RESPONSE: ListeningMode.SINGLE,
    # Not optional: this is the veto window for an announced task. With
    # the microphone shut here the human cannot stop it in time.
    RobotTaskState.R_ANNOUNCED: ListeningMode.CONTINUOUS,
    RobotTaskState.R_EXECUTING: ListeningMode.CONTINUOUS,
    RobotTaskState.R_PAUSED: ListeningMode.CONTINUOUS,
    # The robot is under load with the human's hands full. Nothing is more
    # important than hearing "secured" here.
    RobotTaskState.R_HOLDING: ListeningMode.CONTINUOUS,
    RobotTaskState.R_WAITING_FREE_DRIVE: ListeningMode.SINGLE,
    RobotTaskState.R_FREE_DRIVE: ListeningMode.CONTINUOUS,
    RobotTaskState.R_WAITING_HOME_PERMISSION: ListeningMode.SINGLE,
    RobotTaskState.R_MANUAL_RECOVERY: ListeningMode.CONTINUOUS,
}


class CommunicationManager:
    def __init__(self, cli, parser, voice, tts, event_sink, state_provider,
                 guard_seconds=0.25, max_attempts=2):
        self.cli = cli
        self.parser = parser
        self.voice = voice
        self.tts = tts
        self.event_sink = event_sink
        self.state_provider = state_provider
        self.guard_seconds = guard_seconds
        self.max_attempts = max_attempts
        self.mode = ListeningMode.OFF
        self._state = None
        #: The last thing spoken, so the human can ask for it again.
        #: A worker looking at the ceiling misses prompts constantly.
        self.last_message = None
        self._attempts = 0
        self._generation = 0

    def show_message(self, message: str) -> None:
        self._announce(message)

    def show_permission_request(self, message: str) -> None:
        self._announce(message, permission=True)

    def _announce(self, message: str, permission=False) -> None:
        self.last_message = message
        self._generation += 1
        self.voice.stop_listening()
        output = self.cli.show_permission_request if permission else self.cli.show_message
        output(message)
        self.tts.speak(message)
        time.sleep(self.guard_seconds)
        self.sync_state(self.state_provider(), force=True)

    def sync_state(self, state, force=False) -> None:
        if state == self._state and not force:
            return
        if state != self._state:
            self._attempts = 0
        self._state = state
        self.mode = STATE_MODES.get(state, ListeningMode.OFF)
        self._generation += 1
        self.voice.stop_listening()
        if self.mode is not ListeningMode.OFF:
            generation = self._generation
            self.voice.start_listening(
                lambda text: self._on_voice_text(text, generation),
                lambda *args: self._on_voice_failure(generation),
            )

    def _on_voice_text(self, text: str, generation: int) -> None:
        if generation != self._generation:
            return
        event = self.parser.parse(text, source="human_voice")
        if event is None:
            self._on_voice_failure(generation)
            return
        self._attempts = 0
        self.event_sink(event)

    def _on_voice_failure(self, generation: int) -> None:
        if generation != self._generation:
            return
        self._attempts += 1
        if self._attempts < self.max_attempts:
            self._announce("Sorry, I did not understand. Please try again.")
        else:
            self._attempts = 0
            if self.mode is ListeningMode.CONTINUOUS:
                self.sync_state(self._state, force=True)

    def close(self) -> None:
        self.voice.close()
        self.tts.close()
