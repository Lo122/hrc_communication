"""Offline workflow checks; no microphone, network, or robot is started."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

import config
from cli_interface import CLIInterface
from cmd_parser import CommandParser
from communication_manager import CommunicationManager, ListeningMode
from events import Event, EventType as E, RobotTaskState as S
from gh_dispatcher import GHDispatcher
from message_manager import MessageManager
from models import RecognitionResult
from pending_task import PendingTaskPool
from recognition_manager import RecognitionManager
from state_machine import StateMachine
from task_manager import TaskManager
from trigger_manager import TriggerManager


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()
        self.timer, self.ros, self.output, self.udp = Mock(), Mock(), Mock(), Mock()
        self.manager = TaskManager(
            StateMachine(), PendingTaskPool(), self.timer, MessageManager(),
            self.output, GHDispatcher(self.udp), self.ros, Mock(),
        )

    def emit(self, event_type, **kwargs):
        self.manager.handle_event(Event(event_type, "test", **kwargs))

    def reply(self, text):
        self.manager.handle_event(self.parser.parse(text))

    def trigger(self, step=0):
        self.emit(E.RECOGNITION_TRIGGER, payload={
            "step_id": step, "round_id": 7, "piece_id": 12, "progress": 0.8,
        })

    def complete_robot(self):
        self.emit(E.ROBOT_RUNNING)
        self.emit(E.ROBOT_SUCCESS)

    def hold(self, adjust=True):
        self.trigger()
        self.reply("yes")
        self.complete_robot()
        self.reply("yes" if adjust else "no")
        if adjust:
            self.assertEqual(self.manager.active_task.state, S.R_FREE_DRIVE)
            self.reply("adjustment done")
            self.ros.publish_free_drive.assert_called_with(False)
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)

    def test_full_sequence_with_separate_permissions_and_new_tasks(self):
        self.hold()
        self.reply("screw done")
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.step_id, leave.round_id, leave.piece_id), (2, 3, 7, 12))
        self.assertEqual(leave.state, S.R_WAITING_RESPONSE)
        self.assertEqual(self.udp.send.call_count, 1)
        self.reply("screw done")  # Repetition cannot dispatch or ask twice.
        self.assertIs(self.manager.active_task, leave)
        self.reply("yes")
        self.assertEqual(self.manager.active_task.state, S.R_ACCEPTED)
        self.assertEqual(self.output.show_permission_request.call_count, 2)
        self.complete_robot()
        connector = self.manager.active_task
        self.assertEqual((connector.task_id, connector.state), (3, S.R_WAITING_RESPONSE))
        self.assertEqual(self.udp.send.call_count, 2)
        self.emit(E.ROBOT_SUCCESS)
        self.assertIs(self.manager.active_task, connector)
        self.assertEqual(self.output.show_permission_request.call_count, 3)
        self.reply("yes")
        self.complete_robot()
        for human_step, robot_task in ((4, 4), (5, 5)):
            self.trigger(human_step)
            self.assertEqual(self.manager.active_task.task_id, robot_task)
            self.reply("yes")
            self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        messages = [call.args[0] for call in self.udp.send.call_args_list]
        self.assertEqual([message["step_id"] for message in messages], [1, 2, 3, 4, 5])
        self.assertEqual([message["human_step_id"] for message in messages], [0, 3, 3, 4, 5])

    def test_cli_and_voice_share_screw_done_event_and_state_handling(self):
        for alias in ("screw done", "screwing done", "finished screwing"):
            cli_event = CLIInterface(self.parser).parse_command(alias)
            self.assertEqual(cli_event.event_type, E.H_SCREW_DONE)
            self.assertEqual(self.parser.parse(alias, "human_voice").event_type, cli_event.event_type)
        self.hold()
        voice = Mock()
        comm = CommunicationManager(
            Mock(), self.parser, voice, Mock(), self.manager.handle_event,
            lambda: self.manager.active_task.state, guard_seconds=0,
        )
        comm.sync_state(S.R_HOLDING)
        self.assertEqual(comm.mode, ListeningMode.CONTINUOUS)
        on_text = voice.start_listening.call_args.args[0]
        on_text("screw done")
        comm.poll()
        self.assertEqual(self.manager.active_task.task_id, 2)
        self.assertEqual(self.manager.active_task.state, S.R_WAITING_RESPONSE)
        comm.sync_state(S.R_WAITING_RESPONSE)
        on_text("screw done")  # Old voice callback is discarded.
        comm.poll()
        self.assertEqual(self.output.show_permission_request.call_count, 2)

    def test_done_does_not_release_and_screw_done_is_only_valid_while_holding(self):
        self.reply("screw done")
        self.assertIsNone(self.manager.active_task)
        self.trigger()
        self.reply("screw done")
        self.assertEqual(self.manager.active_task.task_id, 1)
        self.udp.send.assert_not_called()
        self.reply("yes")
        self.complete_robot()
        self.reply("screw done")
        self.assertEqual(self.manager.active_task.state, S.R_WAITING_FREE_DRIVE)
        self.reply("no")
        self.reply("done")
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)

    def test_refusal_and_timeout_keep_leave_pending_without_proposing_connector(self):
        for event_type in (E.H_REFUSE, E.RESPONSE_TIMEOUT):
            with self.subTest(event_type=event_type):
                self.setUp()
                self.hold(adjust=False)
                self.reply("screw done")
                leave = self.manager.active_task
                self.emit(event_type, task_instance_id=leave.task_instance_id)
                self.assertIsNone(self.manager.active_task)
                self.assertTrue(self.manager.pending_pool.contains(leave.task_instance_id))
                self.assertEqual(self.udp.send.call_count, 1)
                self.assertEqual(self.output.show_permission_request.call_count, 2)
                self.assertIn("keep holding", self.output.show_message.call_args.args[0])
                self.trigger(4)
                self.assertIsNone(self.manager.active_task)
                self.reply(f"execute {leave.task_instance_id}")
                self.complete_robot()
                self.assertEqual(self.manager.active_task.task_id, 3)
                self.assertEqual(len(self.manager.waiting_triggers), 1)

    def test_defer_uses_task_duration_and_old_timer_cannot_affect_new_prompt(self):
        self.hold()
        self.reply("screw done")
        leave_id = self.manager.active_task.task_instance_id
        with patch.dict(config.TASK_TIMINGS, {2: {"defer_seconds": 9}, 3: {"response_timeout_seconds": 30}}):
            self.output.show_message.side_effect = lambda message: self.timer.start_defer_timer.assert_not_called()
            self.reply("later")
            self.output.show_message.side_effect = None
            self.timer.start_defer_timer.assert_called_with(leave_id, 9)
            self.assertEqual(self.udp.send.call_count, 1)
            self.emit(E.DEFER_TIMEOUT, task_instance_id=leave_id)
            self.complete_robot()
            connector = self.manager.active_task
            self.timer.start_response_timer.assert_called_with(connector.task_instance_id, 30)
            self.emit(E.RESPONSE_TIMEOUT, task_instance_id=leave_id)
            self.emit(E.DEFER_TIMEOUT, task_instance_id=leave_id)
            self.assertIs(self.manager.active_task, connector)
            self.assertEqual(connector.state, S.R_WAITING_RESPONSE)

    def test_cancel_deferred_leave_does_not_start_connector(self):
        self.hold()
        self.reply("screw done")
        self.reply("later")
        leave_id = self.manager.active_task.task_instance_id
        self.reply("cancel")
        self.timer.cancel_defer_timer.assert_called_once()
        self.emit(E.DEFER_TIMEOUT, task_instance_id=leave_id)
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)
        self.ros.publish_cancel.assert_not_called()
        self.assertEqual(self.udp.send.call_count, 1)
        self.assertEqual(self.output.show_permission_request.call_count, 2)

    def test_retry_leave_keeps_context_and_rejects_previous_attempt_timers(self):
        self.hold()
        self.trigger(4)
        self.reply("screw done")
        old_id = self.manager.active_task.task_instance_id
        self.reply("later")
        self.reply("cancel")
        self.assertEqual(len(self.manager.waiting_triggers), 1)
        self.reply("screw done")
        retry = self.manager.active_task
        self.assertNotEqual(retry.task_instance_id, old_id)
        self.assertEqual((retry.task_id, retry.round_id, retry.piece_id), (2, 7, 12))
        self.emit(E.RESPONSE_TIMEOUT, task_instance_id=old_id)
        self.assertEqual(retry.state, S.R_WAITING_RESPONSE)
        self.reply("later")
        self.emit(E.DEFER_TIMEOUT, task_instance_id=old_id)
        self.assertEqual(retry.state, S.R_DEFER)
        self.assertEqual(self.udp.send.call_count, 1)
        self.emit(E.DEFER_TIMEOUT, task_instance_id=retry.task_instance_id)
        self.complete_robot()
        self.assertEqual(self.manager.active_task.task_id, 3)

    def test_holding_cancel_never_offers_home_even_with_open_gripper_sample(self):
        for gripper in (True, False, None):
            with self.subTest(gripper=gripper):
                self.setUp()
                self.hold()
                self.ros.get_latest_joint_positions.return_value = [0.0] * 6
                self.ros.get_latest_gripper_has_object.return_value = gripper
                with patch.object(config, "RECOVERY_STOP_DELAY_SECONDS", 0):
                    self.reply("cancel")
                self.ros.get_latest_gripper_has_object.assert_called_once()
                self.assertEqual(self.manager.active_task.state, S.R_MANUAL_RECOVERY)
                self.ros.publish_return_home.assert_not_called()
                self.reply("done")
                self.assertIsNone(self.manager.active_task)

    def test_execution_recovery_uses_gripper_feedback(self):
        for gripper, expected in ((True, S.R_MANUAL_RECOVERY), (None, S.R_MANUAL_RECOVERY),
                                  (False, S.R_WAITING_HOME_PERMISSION)):
            with self.subTest(gripper=gripper):
                self.setUp()
                self.trigger(4)
                self.reply("yes")
                self.emit(E.ROBOT_RUNNING)
                self.ros.get_latest_joint_positions.return_value = [0.0] * 6
                self.ros.get_latest_gripper_has_object.return_value = gripper
                with patch.object(config, "RECOVERY_STOP_DELAY_SECONDS", 0), patch.object(config, "RETURN_HOME_RECOVERY_ENABLED", True):
                    self.reply("cancel")
                self.assertEqual(self.manager.active_task.state, expected)
                self.ros.publish_return_home.assert_not_called()

    def test_state_table_matches_lift_success_including_paused(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                self.setUp()
                self.trigger()
                self.reply("yes")
                self.emit(E.ROBOT_RUNNING)
                if paused:
                    self.reply("pause")
                self.assertEqual(self.manager.state_machine.get_next_state(
                    self.manager.active_task.state, E.ROBOT_SUCCESS, 1), S.R_WAITING_FREE_DRIVE)
                self.emit(E.ROBOT_SUCCESS)
                self.assertEqual(self.manager.active_task.state, S.R_WAITING_FREE_DRIVE)

    def test_invalid_cancel_sends_no_robot_command(self):
        self.reply("cancel")
        self.trigger()
        self.reply("cancel")
        self.ros.publish_cancel.assert_not_called()

    def test_transition_rejects_state_table_mismatch(self):
        self.trigger()
        with self.assertRaises(ValueError):
            self.manager._transition(self.manager.active_task, S.R_DONE, Event(E.H_ACCEPT, "test"))
        self.assertEqual(self.manager.active_task.state, S.R_WAITING_RESPONSE)

    def test_busy_triggers_are_queued_once_and_r3_has_priority(self):
        self.hold()
        self.trigger(4)
        self.trigger(4)
        self.trigger(5)
        self.assertEqual(len(self.manager.waiting_triggers), 2)
        self.reply("screw done")
        self.reply("yes")
        self.complete_robot()
        self.assertEqual(self.manager.active_task.task_id, 3)
        self.reply("yes")
        self.complete_robot()
        self.assertEqual(self.manager.active_task.task_id, 4)
        self.reply("yes")
        self.complete_robot()
        self.assertEqual(self.manager.active_task.task_id, 5)

    def test_human_steps_one_two_three_do_not_trigger_robot_tasks(self):
        triggers = TriggerManager()
        for step in (1, 2, 3):
            self.assertEqual(triggers.update(RecognitionResult(0, step, 1.0, 0, 1.0, 0)), [])
            self.trigger(step)
            self.assertIsNone(self.manager.active_task)
        for step in (0, 4, 5):
            events = triggers.update(RecognitionResult(0, step, 0.8, 0, 1.0, 0))
            self.assertEqual(len(events), 1)
            self.assertEqual(triggers.update(RecognitionResult(0, step, 0.9, 0, 1.0, 1)), [])

    def test_speed_and_pause_controls_remain_available(self):
        self.trigger(4)
        self.reply("yes")
        self.emit(E.ROBOT_RUNNING)
        self.reply("faster")
        self.assertIn("increased", self.output.show_message.call_args.args[0])
        self.reply("slower")
        self.assertIn("decreased", self.output.show_message.call_args.args[0])
        self.reply("pause")
        self.assertEqual(self.manager.active_task.state, S.R_PAUSED)
        self.reply("resume")
        self.assertEqual(self.manager.active_task.state, S.R_EXECUTING)

    def test_last_step_stays_in_round_until_next_cable_pulling(self):
        recognition = RecognitionManager.__new__(RecognitionManager)
        recognition.step_stabilizer = None
        recognition.round_id = 0
        recognition.piece_id = 0
        recognition.required_steps_per_round = recognition._load_required_steps_per_round()
        recognition.seen_trigger_steps_in_round = set()
        recognition._last_recorded_step_id = None
        for step in (0, 1, 2, 3, 4, 5, 5, 5):
            result = recognition._result_from_passthrough({"step_id": step})
            self.assertEqual((result.round_id, result.piece_id), (0, 0))
        result = recognition._result_from_passthrough({"step_id": 0})
        self.assertEqual((result.round_id, result.piece_id), (1, 1))

    def test_connector_refusal_requires_explicit_pending_execution(self):
        self.hold()
        self.reply("screw done")
        self.reply("yes")
        self.complete_robot()
        connector_id = self.manager.active_task.task_instance_id
        self.reply("no")
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.udp.send.call_count, 2)
        self.reply(f"execute {connector_id}")
        self.assertEqual(self.udp.send.call_args.args[0]["step_id"], 3)


if __name__ == "__main__":
    unittest.main()
