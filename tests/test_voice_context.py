"""Context routing and WebSocket instruction updates, without live services."""

import json
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "3_communication"):
    sys.path.insert(0, str(ROOT / layer))

from cmd_parser import CommandParser
from communication_manager import CommunicationManager, STATE_MODES
from events import EventType, RobotTaskState as S
from voice_context import VoiceContext, STATE_CONTEXTS, build_instructions
from voice_result import VoiceOutcome as O
import voice_interface as backend


class ContextRoutingTests(unittest.TestCase):
    def setUp(self):
        self.context = VoiceContext(S.R_WAITING_RESPONSE, 1, "lift_1")
        self.voice, self.sink = Mock(), Mock()
        self.comm = CommunicationManager(
            Mock(), CommandParser(), self.voice, Mock(), self.sink,
            lambda: self.context.state, guard_seconds=0,
            context_provider=lambda: self.context,
        )

    def instructions(self):
        return self.voice.start_listening.call_args.kwargs["instructions"]

    def test_short_speech_preserves_full_question_through_retries(self):
        from message_manager import MessageManager

        messages = MessageManager()
        full = messages.get_permission_message(1)
        short = messages.get_permission_message(1, spoken=True)
        self.comm.show_permission_request(full, speech=short)
        self.comm.cli.show_permission_request.assert_called_once_with(full)
        self.comm.tts.speak.assert_called_once_with(short)
        self.assertNotEqual(full, short)
        self.assertIn(full, self.instructions())
        self.assertTrue(self.voice.start_listening.call_args.kwargs["beep"])
        original = self.instructions()
        for outcome in (O.NO_SPEECH, O.UNRECOGNIZED):
            self.voice.start_listening.call_args.args[1](outcome, "test")
            self.comm.poll()
            self.assertEqual(self.instructions(), original)

    def test_short_free_drive_question_preserves_context_and_beep(self):
        from message_manager import MessageManager

        messages = MessageManager()
        self.context = VoiceContext(S.R_WAITING_FREE_DRIVE, 1, "lift_1")
        full = messages.ask_permission_for_free_drive()
        short = messages.ask_permission_for_free_drive(spoken=True)
        self.comm.show_message(full, speech=short)
        self.comm.cli.show_message.assert_called_once_with(full)
        self.comm.tts.speak.assert_called_once_with(short)
        self.assertIn(full, self.instructions())
        self.assertTrue(self.voice.start_listening.call_args.kwargs["beep"])

    def test_every_listening_stage_has_supported_command_template(self):
        parser = CommandParser()
        self.assertEqual(set(STATE_CONTEXTS), set(STATE_MODES))
        for state, (_, commands) in STATE_CONTEXTS.items():
            for command in commands:
                self.assertIsNotNone(parser.parse(command), (state, command))

    def test_adjustment_and_screwing_have_distinct_completion_instructions(self):
        self.context = VoiceContext(S.R_FREE_DRIVE, 1, "lift_1")
        self.comm.sync_state(self.context.state)
        adjustment = self.instructions()
        self.assertIn("Current state: R_FREE_DRIVE", adjustment)
        self.assertIn("Permitted commands: done, cancel, unknown", adjustment)
        self.context = VoiceContext(S.R_HOLDING, 1, "lift_1")
        self.comm.sync_state(self.context.state)
        holding = self.instructions()
        self.assertIn("Permitted commands: screw done, cancel, unknown", holding)
        self.assertNotEqual(holding, adjustment)
        self.assertIn("Explicit statements and negation take precedence", holding)
        # GPT output is still parsed normally; CLI semantics are unchanged.
        self.assertEqual(CommandParser().parse("done").event_type, EventType.H_DONE)
        self.voice.start_listening.call_args.args[0]("screw done")
        self.comm.poll()
        event = self.sink.call_args.args[0]
        self.assertEqual(event.event_type, EventType.H_SCREW_DONE)
        self.assertEqual(event.task_instance_id, "lift_1")

    def test_new_task_same_state_invalidates_old_results(self):
        self.context = VoiceContext(S.R_WAITING_RESPONSE, 2, "leave_1")
        self.comm.show_permission_request("May I leave?")
        old_callback = self.voice.start_listening.call_args.args[0]
        self.context = VoiceContext(S.R_WAITING_RESPONSE, 3, "connector_1")
        self.comm.sync_state(self.context.state)
        self.assertIn("bring the pipe connector", self.instructions())
        self.assertIn("connector_1", self.instructions())
        self.assertNotIn("May I leave?", self.instructions())
        old_callback("yes")
        self.comm.poll()
        self.sink.assert_not_called()

    def test_new_instance_same_task_and_state_also_replaces_context(self):
        self.comm.sync_state(self.context.state)
        count = self.voice.start_listening.call_count
        self.context = VoiceContext(S.R_WAITING_RESPONSE, 1, "lift_2")
        self.comm.sync_state(self.context.state)
        self.assertEqual(self.voice.start_listening.call_count, count + 1)
        self.assertIn("lift_2", self.instructions())

    def test_silent_retry_and_clarification_preserve_original_question(self):
        self.comm.show_permission_request("Would you like me to lift the panel?")
        original = self.instructions()
        for outcome in (O.NO_SPEECH, O.UNRECOGNIZED, O.NO_SPEECH):
            self.voice.start_listening.call_args.args[1](outcome, "test")
            self.comm.poll()
            self.assertEqual(self.instructions(), original)

    def test_error_message_does_not_replace_question(self):
        self.comm.show_permission_request("Would you like me to lift the panel?")
        original = self.instructions()
        with patch("communication_manager.time.monotonic", return_value=0):
            self.voice.start_listening.call_args.args[1](O.ERROR, "offline")
            self.comm.poll()
        with patch("communication_manager.time.monotonic", return_value=5):
            self.comm.poll()
            self.voice.start_listening.call_args.args[1](O.ERROR, "offline")
            self.comm.poll()
        with patch("communication_manager.time.monotonic", return_value=10):
            self.comm.poll()
        self.assertEqual(self.instructions(), original)

    def test_free_drive_question_passed_via_show_message_is_captured(self):
        self.context = VoiceContext(S.R_WAITING_FREE_DRIVE, 1, "lift_1")
        self.comm.show_message("Would you like free drive for adjustment?")
        self.assertIn("Would you like free drive for adjustment?", self.instructions())
        self.assertIn("Permitted commands: free drive, no, cancel, unknown", self.instructions())

    def test_runtime_returns_snapshot_instead_of_active_task(self):
        from communication_runtime import HRCSystem
        system = HRCSystem.__new__(HRCSystem)
        task = Mock(state=S.R_FREE_DRIVE, task_id=1, task_instance_id="lift_1")
        system.task_manager = Mock(active_task=task)
        snapshot = system._current_voice_context()
        task.state = S.R_HOLDING
        self.assertEqual(snapshot, VoiceContext(S.R_FREE_DRIVE, 1, "lift_1"))
        self.assertEqual(system._current_voice_context().state, S.R_HOLDING)


class SessionInstructionTests(unittest.TestCase):
    def setUp(self):
        self.voice = backend.VoiceInterface.__new__(backend.VoiceInterface)
        self.voice._stop = threading.Event()
        self.voice._ws = Mock(connected=True)
        self.voice._applied_instructions = "old context"

    def ack(self, instructions):
        return json.dumps({"type": "session.updated", "session": {"instructions": instructions}})

    def test_changed_instructions_update_same_connection(self):
        ws = self.voice._ws
        ws.recv.return_value = self.ack("new context")
        with patch.object(backend.websocket, "create_connection") as connect:
            self.assertTrue(self.voice._connect_gpt("new context"))
        connect.assert_not_called()
        ws.close.assert_not_called()
        self.assertEqual(json.loads(ws.send.call_args.args[0]), {
            "type": "session.update", "session": {"type": "realtime", "instructions": "new context"},
        })
        self.assertEqual(self.voice._applied_instructions, "new context")

    def test_unchanged_instructions_do_not_send_update(self):
        self.assertTrue(self.voice._connect_gpt("old context"))
        self.voice._ws.send.assert_not_called()
        self.voice._ws.recv.assert_not_called()

    def test_stale_ack_and_reply_are_discarded_until_new_context_confirmed(self):
        ws = self.voice._ws
        ws.recv.side_effect = [self.ack("old context"),
                               json.dumps({"type": "response.output_text.done", "text": "yes"}),
                               self.ack("new context")]
        self.assertTrue(self.voice._connect_gpt("new context"))
        self.assertEqual(ws.recv.call_count, 3)
        self.assertEqual(self.voice._applied_instructions, "new context")

    def test_reconnect_sends_current_context_with_full_configuration(self):
        old_ws = self.voice._ws
        old_ws.connected = False
        ws = Mock(connected=True)
        instructions = build_instructions(VoiceContext(S.R_HOLDING, 1, "lift_1"))
        ws.recv.return_value = self.ack(instructions)
        with patch.dict(backend.os.environ, {"OPENAI_API_KEY": "test-only", "VOICE_MODEL": "gpt-realtime-2"}), \
                patch.object(backend.websocket, "create_connection", return_value=ws):
            self.assertTrue(self.voice._connect_gpt(instructions))
        old_ws.close.assert_called_once()
        session = json.loads(ws.send.call_args.args[0])["session"]
        self.assertEqual(session["instructions"], instructions)
        self.assertEqual(session["audio"]["input"]["format"]["rate"], 24000)

    def test_update_failure_never_beeps_or_opens_microphone(self):
        self.voice._play_ready_beep = Mock()
        self.voice._ws.recv.return_value = json.dumps({"type": "error", "error": {"message": "update rejected"}})
        on_text, on_failure = Mock(), Mock()
        with patch.object(backend.sd, "RawInputStream") as audio:
            self.voice._listen_once_gpt(on_text, on_failure, beep=True, instructions="new context")
        audio.assert_not_called()
        self.voice._play_ready_beep.assert_not_called()
        on_failure.assert_called_once_with(O.ERROR, "update rejected")
        self.assertIsNone(self.voice._ws)

    def test_missing_ack_times_out_and_discards_connection(self):
        ws = self.voice._ws
        ws.recv.side_effect = backend.websocket.WebSocketTimeoutException()
        with patch.object(backend.time, "monotonic", side_effect=[0, 0, 6]):
            with self.assertRaises(TimeoutError):
                self.voice._connect_gpt("new context")
        ws.close.assert_called_once()
        self.assertIsNone(self.voice._ws)
        self.assertEqual(self.voice._applied_instructions, "old context")

    def test_standalone_uses_selected_template_before_recording(self):
        spec = importlib.util.spec_from_file_location("context_stt_test", ROOT / "3_communication/gpt_live/gpt_stt.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        instructions = build_instructions(VoiceContext(S.R_HOLDING, 1, "voice_test"))
        ws = Mock()
        ws.recv.side_effect = [self.ack(instructions),
                               json.dumps({"type": "response.output_text.done", "text": "screw done"})]
        with patch.object(sys, "argv", ["gpt_stt.py", "--state", "R_HOLDING", "--task-id", "1"]), \
                patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-only", "VOICE_MODEL": "gpt-realtime-2"}), \
                patch.object(module.websocket, "create_connection", return_value=ws), \
                patch.object(module.sd, "RawInputStream") as audio, patch("builtins.print"):
            def open_audio(**kwargs):
                self.assertEqual(ws.recv.call_count, 1)  # Session acknowledged first.
                stream = Mock()
                stream.read.return_value = (b"\x00\x00", False)
                manager = Mock()
                manager.__enter__ = Mock(return_value=stream)
                manager.__exit__ = Mock(return_value=False)
                return manager
            audio.side_effect = open_audio
            module.main()
        self.assertEqual(json.loads(ws.send.call_args_list[0].args[0])["session"]["instructions"], instructions)
        ws.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
