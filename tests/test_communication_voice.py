import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "0_core"), str(ROOT / "3_communication")]

from cmd_parser import CommandParser
from communication_manager import CommunicationManager, ListeningMode
from events import EventType, RobotTaskState
from message_manager import MessageManager
from tts_manager import TTSManager


class FakeCLI:
    def __init__(self):
        self.messages = []

    def show_message(self, text):
        self.messages.append(text)

    def show_permission_request(self, text):
        self.messages.append(text)


class FakeVoice:
    def __init__(self):
        self.callback = None
        self.failure = None
        self.calls = []

    def start_listening(self, callback, failure):
        self.calls.append("start")
        self.callback, self.failure = callback, failure

    def stop_listening(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, text):
        self.messages.append(text)

    def close(self):
        pass


class CommunicationVoiceTests(unittest.TestCase):
    def build_manager(self):
        self.events, self.state = [], RobotTaskState.R_WAITING_RESPONSE
        self.cli, self.voice, self.tts = FakeCLI(), FakeVoice(), FakeTTS()
        return CommunicationManager(
            self.cli, CommandParser(), self.voice, self.tts,
            self.events.append, lambda: self.state, guard_seconds=0,
        )

    def test_cli_and_voice_use_same_event_type(self):
        manager = self.build_manager()
        manager.sync_state(self.state)
        manager._on_voice_text("yes", manager._generation)
        cli_event = CommandParser().parse("yes")
        self.assertEqual(self.events[0].event_type, cli_event.event_type)
        self.assertEqual(self.events[0].source, "human_voice")
        self.assertEqual(cli_event.source, "human_cli")

    def test_output_goes_to_terminal_and_tts_then_listening_resumes(self):
        manager = self.build_manager()
        manager.show_permission_request("May I continue?")
        self.assertEqual(self.cli.messages, ["May I continue?"])
        self.assertEqual(self.tts.messages, ["May I continue?"])
        self.assertEqual(manager.mode, ListeningMode.SINGLE)
        self.assertEqual(self.voice.calls[-1], "start")

    def test_second_single_mode_failure_stops_current_round(self):
        manager = self.build_manager()
        manager.sync_state(self.state)
        manager._on_voice_failure(manager._generation)
        manager._on_voice_failure(manager._generation)
        self.assertEqual(self.tts.messages, ["Sorry, I did not understand. Please try again."])
        self.assertEqual(len(self.events), 0)

    def test_speed_commands_have_acknowledgements(self):
        messages = MessageManager()
        self.assertEqual(
            messages.get_acknowledgement(EventType.H_SPEEDUP),
            "The robot speed has been increased.",
        )
        self.assertEqual(
            messages.get_acknowledgement(EventType.H_SLOWDOWN),
            "The robot speed has been decreased.",
        )

    def test_tts_creates_a_fresh_engine_for_each_message(self):
        engines = []

        class FakeEngine:
            def setProperty(self, name, value):
                pass

            def say(self, text):
                pass

            def runAndWait(self):
                pass

            def stop(self):
                pass

        def create_engine(driver):
            engine = FakeEngine()
            engines.append(engine)
            return engine

        with patch("tts_manager.pyttsx3.init", side_effect=create_engine):
            tts = TTSManager()
            tts.speak("First message")
            tts.speak("Second message")
            tts.close()

        self.assertEqual(len(engines), 2)
        self.assertIsNot(engines[0], engines[1])


if __name__ == "__main__":
    unittest.main()
