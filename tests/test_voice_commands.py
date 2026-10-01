"""Spoken commands: the wake word, the wider vocabulary, and what is ignored."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "3_communication"):
    sys.path.insert(0, str(ROOT / layer))

from cmd_parser import CommandParser
from communication_manager import CommunicationManager
from events import EventType as E, RobotTaskState as S
from voice_context import VoiceContext


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()

    def kind(self, text):
        event = self.parser.parse(text)
        return None if event is None else event.event_type

    def test_synonyms_and_polite_words(self):
        for text, expected in (("Yeah", E.H_ACCEPT), ("go ahead", E.H_ACCEPT), ("yes, please!", E.H_ACCEPT),
                               ("I'll do it", E.H_REFUSE), ("no thanks", E.H_REFUSE),
                               ("not yet", E.H_DEFER), ("carry on", E.H_RESUME), ("start over", E.H_RESTART),
                               ("I'm done", E.H_DONE), ("all screwed", E.H_SCREW_DONE),
                               ("robot, pause please", E.H_PAUSE), ("hand it over", E.H_HANDOVER),
                               ("go home now", E.H_RETURN_HOME)):
            self.assertEqual(self.kind(text), expected, text)
        self.assertIsNone(self.kind("the weather is nice"))
        self.assertEqual(self.parser.parse("execute lift_1_2").task_instance_id, "lift_1_2")

    def test_replies_mean_something_only_where_they_fit(self):
        def kind(text, state):
            event = self.parser.parse(text, "human_voice", state)
            return None if event is None else event.event_type

        # "I'll do it" hands a task over -- never "just hold" when asked to adjust.
        self.assertEqual(kind("I'll do it", S.R_WAITING_RESPONSE), E.H_REFUSE)
        self.assertIsNone(kind("I'll do it", S.R_WAITING_FREE_DRIVE))
        # "Next step" is a yes to a question -- never "go home" or "open the gripper".
        self.assertEqual(kind("next step", S.R_WAITING_RESPONSE), E.H_ACCEPT)
        self.assertIsNone(kind("next step", S.R_WAITING_HOME_PERMISSION))
        self.assertIsNone(kind("next step", S.R_WAITING_HANDOVER))
        # "Just hold" only answers "adjust by hand?".
        self.assertEqual(kind("just hold", S.R_WAITING_FREE_DRIVE), E.H_REFUSE)
        self.assertIsNone(kind("just hold", S.R_WAITING_HOME_PERMISSION))
        # "Not yet": later to a question, not ready for the item held out.
        self.assertEqual(kind("not yet", S.R_WAITING_RESPONSE), E.H_DEFER)
        self.assertEqual(kind("not yet", S.R_WAITING_HANDOVER), E.H_REFUSE)
        # Typed at the CLI, with no state: the first meaning.
        self.assertEqual(self.kind("I'll do it"), E.H_REFUSE)

    def test_the_wake_word_addresses_the_robot(self):
        addressed = self.parser.addressed
        self.assertEqual(addressed("hey you are lift the panel"), "lift the panel")
        self.assertEqual(addressed("Hey UR, yes please"), "yes please")
        self.assertEqual(addressed("hey robot [unk] yes"), "yes")
        self.assertEqual(addressed("you are carry on"), "carry on")  # "hey" lost
        self.assertEqual(addressed("hey you are"), "")          # the name alone
        self.assertIsNone(addressed("yes"))                      # not meant for the robot
        self.assertIsNone(addressed("the [unk]"))
        self.assertEqual(addressed("stop"), "stop")              # emergency words need no name
        self.assertEqual(addressed("cancel"), "cancel")

    def test_voice_grammar(self):
        phrases = CommandParser.voice_phrases(True)
        self.assertIn("hey you are yes", phrases)
        self.assertIn("stop", phrases)
        self.assertEqual(phrases[-1], "[unk]")
        self.assertNotIn("hey you are", CommandParser.voice_phrases(False))


class WakeWordTests(unittest.TestCase):
    """The robot's name is needed except to answer the robot's own question."""

    def setUp(self):
        self.context = VoiceContext(S.R_EXECUTING, 1, "lift_1")
        self.voice, self.sink, self.cli = Mock(), Mock(), Mock()
        self.comm = CommunicationManager(
            self.cli, CommandParser(), self.voice, Mock(), self.sink,
            lambda: self.context.state, guard_seconds=0,
            context_provider=lambda: self.context, wake_word=True,
        )

    def at(self, state, task_id=1):
        self.context = VoiceContext(state, task_id if state else None, f"t{state}")
        self.comm.sync_state(state, force=True)

    def hear(self, text):
        self.comm.poll()  # listening again after the last command, as the runtime's next tick does
        on_text = self.voice.start_listening.call_args.args[0]
        on_text(text)
        self.comm.poll()

    def sent(self):
        return [call.args[0].event_type for call in self.sink.call_args_list]

    def test_an_answer_to_the_robots_question_needs_no_name(self):
        self.at(S.R_WAITING_RESPONSE)
        self.hear("yes")
        self.assertEqual(self.sent(), [E.H_ACCEPT])
        self.at(S.R_WAITING_HANDOVER)
        self.hear("hey you are not yet")  # with the name it still counts: not ready for it
        self.assertEqual(self.sent(), [E.H_ACCEPT, E.H_REFUSE])

    def test_controlling_the_task_under_way_needs_no_name(self):
        self.at(S.R_EXECUTING)
        for said in ("faster", "slower", "pause"):
            self.hear(said)
        self.assertEqual(self.sent(), [E.H_SPEEDUP, E.H_SLOWDOWN, E.H_PAUSE])

    def test_another_task_needs_the_name_while_the_robot_works(self):
        self.at(S.R_EXECUTING)
        self.hear("bring the connector")
        self.assertEqual(self.sent(), [])
        self.cli.show_permission_request.assert_not_called()  # ignored quietly
        self.hear("hey you are bring the connector")
        self.assertEqual(self.sent(), [E.H_REQUEST_ROBOT_TASK])

    def test_the_name_then_the_command(self):
        self.at(S.R_HOLDING)
        self.hear("hey you are")
        self.assertEqual(self.sent(), [])
        self.hear("screw done")
        self.assertEqual(self.sent(), [E.H_SCREW_DONE])

    def test_stop_needs_no_name(self):
        self.at(S.R_EXECUTING)
        self.hear("stop")
        self.assertEqual(self.sent(), [E.H_PAUSE])

    def test_idle_listens_for_its_name(self):
        self.at(None)
        self.voice.start_listening.assert_called()  # listening although nothing is asked
        self.hear("bring the tool")
        self.assertEqual(self.sent(), [])
        self.hear("hey you are bring the tool")
        self.assertEqual(self.sent(), [E.H_REQUEST_ROBOT_TASK])

    def test_repeat_says_the_question_again(self):
        self.at(S.R_WAITING_RESPONSE)
        self.comm.show_permission_request("Would you like me to lift the panel? Say yes or no.",
                                          speech="Would you like me to lift the panel?")
        self.cli.reset_mock()
        self.hear("repeat")  # no name needed while it asks
        self.cli.show_permission_request.assert_called_once_with("Would you like me to lift the panel? Say yes or no.")
        self.assertEqual(self.sent(), [])  # nothing for the decision layer

    def test_repeat_while_idle_needs_the_name(self):
        self.at(None)
        self.comm.show_message("I am out of your way now.")
        self.cli.reset_mock()
        self.hear("say that again")
        self.cli.show_message.assert_not_called()
        self.hear("hey you are say that again")
        self.cli.show_message.assert_called_once_with("I am out of your way now.")

    def test_idle_listens_from_the_start(self):
        self.comm.poll()  # the default context: no state change happened yet
        self.voice.start_listening.assert_called_once()

    def test_a_command_that_changes_nothing_still_listens_again(self):
        self.at(None)
        self.hear("hey you are next piece")  # no state change follows
        self.assertEqual(self.sent(), [E.H_NEXT_PIECE])
        calls = self.voice.start_listening.call_count
        self.comm.poll()
        self.assertEqual(self.voice.start_listening.call_count, calls + 1)

    def test_awaited_replies_need_no_name(self):
        self.at(S.R_HOLDING)
        self.hear("done")
        self.at(S.R_HOLDING_HANDOVER)
        self.hear("give me the connector")
        self.assertEqual(self.sent(), [E.H_DONE, E.H_HANDOVER])

    def test_after_taking_a_task_over_done_needs_no_name(self):
        self.context = VoiceContext(None, task_instance_id="human Pull Cables", human_turn=True)
        self.comm.sync_state(None, force=True)
        self.hear("done")
        self.hear("faster")  # anything else still needs it
        self.assertEqual(self.sent(), [E.H_DONE])

    def test_without_the_wake_word_idle_stays_quiet(self):
        self.comm.wake_word = False
        self.voice.reset_mock()
        self.at(None)
        self.voice.start_listening.assert_not_called()


if __name__ == "__main__":
    unittest.main()
