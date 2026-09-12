"""Speech policy tests with simulated callbacks, audio, and network messages."""

import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

from cmd_parser import CommandParser
from communication_manager import CommunicationManager
from events import Event, EventType, RobotTaskState as S
from voice_result import VoiceOutcome as O
import voice_interface as backend


class VoicePolicyTests(unittest.TestCase):
    def setUp(self):
        self.state = S.R_WAITING_RESPONSE
        self.voice, self.cli, self.tts, self.sink, self.logger = (Mock() for _ in range(5))
        self.comm = CommunicationManager(
            self.cli, CommandParser(), self.voice, self.tts, self.sink,
            lambda: self.state, guard_seconds=0, logger=self.logger,
        )
        self.comm.show_permission_request("Would you like help?")
        self.tts.reset_mock()

    def outcome(self, outcome, detail="test"):
        self.voice.start_listening.call_args.args[1](outcome, detail)
        self.comm.poll()

    def test_silence_restarts_without_speech_beep_or_attempts(self):
        self.assertTrue(self.voice.start_listening.call_args.kwargs["beep"])
        for _ in range(4):
            self.outcome(O.NO_SPEECH)
            self.assertFalse(self.voice.start_listening.call_args.kwargs["beep"])
        self.assertEqual(self.comm._attempts, 0)
        self.tts.speak.assert_not_called()
        self.sink.assert_not_called()

    def test_question_clarifies_once_then_resumes_silently(self):
        self.outcome(O.UNRECOGNIZED)
        self.assertIn("yes, no, or later", self.tts.speak.call_args.args[0])
        self.assertTrue(self.voice.start_listening.call_args.kwargs["beep"])
        self.outcome(O.NO_SPEECH)
        self.outcome(O.UNRECOGNIZED)
        self.outcome(O.UNRECOGNIZED)
        self.assertEqual(self.tts.speak.call_count, 1)
        self.assertFalse(self.voice.start_listening.call_args.kwargs["beep"])

    def test_free_drive_question_uses_yes_no(self):
        self.state = S.R_WAITING_FREE_DRIVE
        self.comm.show_message("Would you like free drive?")
        self.assertTrue(self.voice.start_listening.call_args.kwargs["beep"])
        self.outcome(O.UNRECOGNIZED)
        self.assertIn("yes or no", self.tts.speak.call_args.args[0])
        self.assertNotIn("later", self.tts.speak.call_args.args[0])

    def test_continuous_noise_is_silent_and_command_still_works(self):
        for state in (S.R_EXECUTING, S.R_FREE_DRIVE, S.R_HOLDING):
            self.state = state
            self.comm.sync_state(state)
            self.outcome(O.UNRECOGNIZED)
            self.assertFalse(self.voice.start_listening.call_args.kwargs["beep"])
        self.tts.speak.assert_not_called()
        callback = self.voice.start_listening.call_args.args[0]
        callback("screw done")
        self.sink.assert_not_called()  # Worker only enqueues a result.
        self.comm.poll()
        self.assertEqual(self.sink.call_args.args[0].event_type, EventType.H_SCREW_DONE)

    def test_error_retry_is_delayed_and_notification_not_repeated(self):
        with patch("communication_manager.time.monotonic", return_value=100):
            self.outcome(O.ERROR, "connection lost")
            count = self.voice.start_listening.call_count
            self.comm.poll()
            self.assertEqual(self.voice.start_listening.call_count, count)
            self.tts.speak.assert_not_called()
        with patch("communication_manager.time.monotonic", return_value=105):
            self.comm.poll()
            self.outcome(O.ERROR)
            self.assertIn("temporarily unavailable", self.tts.speak.call_args.args[0])
        with patch("communication_manager.time.monotonic", return_value=110):
            self.comm.poll()
            self.outcome(O.ERROR)
        self.assertEqual(self.tts.speak.call_count, 1)
        self.assertEqual(self.logger.log_message.call_args.args[1]["outcome"], "ERROR")

    def test_old_results_and_pending_retries_cannot_restart_in_off_state(self):
        text, failure = self.voice.start_listening.call_args.args
        with patch("communication_manager.time.monotonic", return_value=100):
            self.outcome(O.ERROR)
        self.state = None
        self.comm.sync_state(None)
        count = self.voice.start_listening.call_count
        text("yes")
        failure(O.NO_SPEECH, "late result")
        with patch("communication_manager.time.monotonic", return_value=200):
            self.comm.poll()
        self.assertEqual(self.voice.start_listening.call_count, count)
        self.sink.assert_not_called()

    def test_close_ignores_late_callback(self):
        callback = self.voice.start_listening.call_args.args[1]
        count = self.voice.start_listening.call_count
        self.comm.close()
        callback(O.NO_SPEECH)
        self.comm.poll()
        self.assertEqual(self.voice.start_listening.call_count, count)

    def test_silence_does_not_restart_task_response_timer(self):
        from communication_runtime import HRCSystem
        from event_queue import EventQueue
        from message_manager import MessageManager
        from pending_task import PendingTaskPool
        from state_machine import StateMachine
        from task_manager import TaskManager

        system = HRCSystem.__new__(HRCSystem)
        system.event_queue = EventQueue()
        system.communication = self.comm
        timer = Mock()
        system.task_manager = TaskManager(
            StateMachine(), PendingTaskPool(), timer, MessageManager(),
            self.comm, Mock(), Mock(), Mock(),
        )
        self.comm.state_provider = system._current_state
        self.comm.event_sink = system.event_queue.put
        system.event_queue.put(Event(EventType.RECOGNITION_TRIGGER, "test", payload={
            "step_id": 0, "round_id": 0, "piece_id": 0,
        }))
        system.process_events()
        task_id = system.task_manager.active_task.task_instance_id
        for _ in range(3):
            self.voice.start_listening.call_args.args[1](O.NO_SPEECH)
            system.process_events()
        timer.start_response_timer.assert_called_once_with(task_id, 20.0)
        system.event_queue.put(Event(EventType.RESPONSE_TIMEOUT, "test", task_instance_id=task_id))
        system.process_events()
        self.assertIsNone(system.task_manager.active_task)
        self.assertTrue(system.task_manager.pending_pool.contains(task_id))


class VoiceBackendTests(unittest.TestCase):
    def setUp(self):
        self.voice = backend.VoiceInterface.__new__(backend.VoiceInterface)
        self.voice._stop = threading.Event()
        self.voice._thread = None
        self.voice._device = None
        self.voice._model = Mock()
        self.voice._grammar = '["yes"]'
        self.voice._phrases = {"yes"}
        self.voice._timeout = 3
        self.voice._ws = Mock()
        self.voice._connect_gpt = Mock(return_value=True)
        self.voice._play_ready_beep = Mock()
        self.on_text, self.on_failure = Mock(), Mock()

    def gpt(self, events, beep=False):
        now = [0]
        stream = Mock()
        def read(_):
            now[0] += 1
            return b"\x00\x00", False
        stream.read.side_effect = read
        messages = iter(events)
        def recv():
            event = next(messages, None)
            if event is None:
                raise backend.websocket.WebSocketTimeoutException()
            return json.dumps(event)
        self.voice._ws.recv.side_effect = recv
        with patch.object(backend.sd, "RawInputStream") as audio, patch.object(
            backend.time, "monotonic", side_effect=lambda: now[0]
        ):
            audio.return_value.__enter__.return_value = stream
            self.voice._listen_once_gpt(self.on_text, self.on_failure, beep)

    def test_gpt_silence_is_not_unrecognized(self):
        self.gpt([])
        self.assertEqual(self.on_failure.call_args.args[0], O.NO_SPEECH)
        self.voice._play_ready_beep.assert_not_called()

    def test_gpt_speech_timeout_is_distinct_and_discards_late_response(self):
        ws = self.voice._ws
        self.gpt([{"type": "input_audio_buffer.speech_started"}])
        self.assertEqual(self.on_failure.call_args.args[0], O.UNRECOGNIZED)
        ws.close.assert_called_once()
        self.assertIsNone(self.voice._ws)

    def test_gpt_unknown_after_speech_is_unrecognized(self):
        self.gpt([{"type": "input_audio_buffer.speech_started"},
                  {"type": "response.output_text.done", "text": "unknown"}])
        self.assertEqual(self.on_failure.call_args.args[0], O.UNRECOGNIZED)

    def test_gpt_unknown_without_speech_does_not_request_clarification(self):
        self.gpt([{"type": "response.output_text.done", "text": "unknown"}])
        self.assertEqual(self.on_failure.call_args.args[0], O.NO_SPEECH)

    def test_interrupted_gpt_window_discards_socket_without_result(self):
        ws = self.voice._ws
        def connected():
            self.voice._stop.set()
            return True
        self.voice._connect_gpt.side_effect = connected
        self.gpt([])
        self.on_failure.assert_not_called()
        self.on_text.assert_not_called()
        ws.close.assert_called_once()
        self.assertIsNone(self.voice._ws)

    def test_gpt_valid_command_and_explicit_beep(self):
        self.gpt([{"type": "input_audio_buffer.speech_started"},
                  {"type": "response.output_text.done", "text": "yes"}], beep=True)
        self.on_text.assert_called_once_with("yes")
        self.on_failure.assert_not_called()
        self.voice._play_ready_beep.assert_called_once()

    def test_connection_error_is_not_recognition_failure(self):
        self.voice._connect_gpt.side_effect = RuntimeError("offline")
        self.voice._listen_once_gpt(self.on_text, self.on_failure)
        self.on_failure.assert_called_once_with(O.ERROR, "offline")

    def test_real_session_builder_reaches_beep_after_session_updated(self):
        # Exercise session construction too; mocking _connect_gpt would hide
        # undefined constants and invalid top-level configuration fields.
        del self.voice._connect_gpt
        self.voice._ws = None
        ws = Mock()
        ws.recv.side_effect = [
            json.dumps({"type": "session.updated"}),
            json.dumps({"type": "response.output_text.done", "text": "yes"}),
        ]
        with patch.dict(backend.os.environ, {"OPENAI_API_KEY": "test-only", "VOICE_MODEL": "gpt-realtime-2"}), \
                patch.object(backend.websocket, "create_connection", return_value=ws), \
                patch.object(backend.sd, "RawInputStream") as audio:
            audio.return_value.__enter__.return_value.read.return_value = (b"\x00\x00", False)
            self.voice._listen_once_gpt(self.on_text, self.on_failure, beep=True)
        self.on_failure.assert_not_called()
        self.voice._play_ready_beep.assert_called_once()
        self.on_text.assert_called_once_with("yes")
        session = json.loads(ws.send.call_args_list[0].args[0])["session"]
        self.assertEqual(session["audio"]["input"]["format"]["rate"], 24000)
        self.assertEqual(audio.call_args.kwargs["samplerate"], 24000)
        self.assertEqual(session["reasoning"], {"effort": "low"})
        self.assertEqual(set(session), {"type", "instructions", "output_modalities", "max_output_tokens", "audio", "reasoning"})

    def test_vosk_empty_result_is_silent_and_nonempty_goes_to_parser(self):
        self.voice._timeout = 0
        with patch.object(backend.sd, "query_devices", return_value={"default_samplerate": 16000}), \
                patch.object(backend.sd, "RawInputStream"), patch.object(backend, "KaldiRecognizer") as model:
            model.return_value.FinalResult.return_value = '{"text": ""}'
            self.voice._listen_once(self.on_text, self.on_failure)
            self.assertEqual(self.on_failure.call_args.args[0], O.NO_SPEECH)
            model.return_value.FinalResult.return_value = '{"text": "unrecognized request"}'
            self.voice._listen_once(self.on_text, self.on_failure)
            self.on_text.assert_called_once_with("unrecognized request")

    def test_vosk_device_error_is_reported(self):
        with patch.object(backend.sd, "query_devices", side_effect=RuntimeError("device unavailable")):
            self.voice._listen_once(self.on_text, self.on_failure)
        self.on_failure.assert_called_once_with(O.ERROR, "device unavailable")

    def test_stuck_worker_is_not_replaced_or_reactivated(self):
        old_worker = Mock()
        old_worker.is_alive.return_value = True
        self.voice._thread = old_worker
        with patch.object(backend.threading, "Thread") as thread:
            self.voice.start_listening(self.on_text, self.on_failure)
        thread.assert_not_called()
        self.assertIs(self.voice._thread, old_worker)
        self.assertTrue(self.voice._stop.is_set())
        self.assertEqual(self.on_failure.call_args.args[0], O.ERROR)


if __name__ == "__main__":
    unittest.main()
