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
from events import Event, EventType as E, RobotTaskState as S, TaskStatus as T
from gh_dispatcher import GHDispatcher
from message_manager import MessageManager
from models import RecognitionResult
from pending_task import PendingTaskPool
from recognition_manager import RecognitionManager
from state_machine import StateMachine
from task_manager import TaskManager
from task_tracker import build_task_tracking
from trigger_manager import TaskUpdatePublisher

PULL, LIFT, PLACE, ALIGN, SCREW, CONNECT, CLAMP = range(7)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()
        self.timer, self.ros, self.output, self.udp = Mock(), Mock(), Mock(), Mock()
        self.tracker, self.policy = build_task_tracking(ROOT, logger=Mock())
        self.manager = TaskManager(
            StateMachine(), PendingTaskPool(), self.timer, MessageManager(),
            self.output, GHDispatcher(self.udp), self.ros, Mock(),
            task_tracker=self.tracker, trigger_policy=self.policy,
        )

    def emit(self, event_type, **kwargs):
        self.manager.handle_event(Event(event_type, "test", **kwargs))

    def reply(self, text):
        self.manager.handle_event(self.parser.parse(text))

    def trigger(self, step=PULL, progress=0.8):
        self.emit(E.HUMAN_TASK_UPDATE, payload={"step_id": step, "round_id": 7, "progress": progress})

    def complete_robot(self):
        self.emit(E.ROBOT_RUNNING)
        self.emit(E.ROBOT_SUCCESS)

    def status(self, name, piece=1):
        return self.tracker.get(name, piece).status

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
        self.assertEqual((leave.task_id, leave.step_id, leave.round_id, leave.piece_id),
                         (2, config.HUMAN_SCREW_DONE, 7, 1))
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
        # Screw done unlocked Bring Tool too; it was queued behind leave + connector.
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CLAMPING_TOOL)
        self.reply("yes")
        self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        self.trigger(CLAMP)
        self.assertEqual(self.manager.active_task.task_id, config.TASK_RETURN_CLAMPING_TOOL)
        self.reply("yes")
        self.complete_robot()
        # Clamp Coupling also lets the robot lift the next piece's panel.
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.piece_id), (config.TASK_LIFT_PANEL, 2))
        messages = [call.args[0] for call in self.udp.send.call_args_list]
        self.assertEqual([message["step_id"] for message in messages], [1, 2, 3, 4, 5])
        self.assertEqual([message["human_step_id"] for message in messages],
                         [PULL, config.HUMAN_SCREW_DONE, config.HUMAN_SCREW_DONE, SCREW, CLAMP])
        for name in ("Lift", "Place", "Screw", "Bring Connector", "Bring Tool", "Bring back Tool"):
            self.assertEqual(self.status(name), T.DONE, name)

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
                self.trigger(CONNECT)
                self.assertIsNone(self.manager.active_task)
                self.reply(f"execute {leave.task_instance_id}")
                self.complete_robot()
                self.assertEqual(self.manager.active_task.task_id, 3)
                # Bring Tool (from screw done) and piece 2's Lift (Connect Cables is a
                # previous task of Lift in the database).
                self.assertEqual([(entry["task_name"], entry["piece_id"]) for entry in self.manager.waiting_triggers],
                                 [("Bring Tool", 1), ("Lift", 2)])

    def test_defer_uses_task_duration_and_old_timer_cannot_affect_new_prompt(self):
        self.hold()
        self.reply("screw done")
        leave_id = self.manager.active_task.task_instance_id
        with patch.dict(config.TASK_TIMINGS, {2: {"defer_seconds": 9}, 3: {"response_timeout_seconds": 30}}):
            self.output.show_message.side_effect = lambda message, **kwargs: self.timer.start_defer_timer.assert_not_called()
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
        self.trigger(SCREW)
        self.reply("screw done")
        old_id = self.manager.active_task.task_instance_id
        self.reply("later")
        self.reply("cancel")
        # Bring Tool and Bring Connector, queued once when Screw passed 50%.
        self.assertEqual(len(self.manager.waiting_triggers), 2)
        self.reply("screw done")
        retry = self.manager.active_task
        self.assertNotEqual(retry.task_instance_id, old_id)
        self.assertEqual((retry.task_id, retry.round_id, retry.piece_id), (2, 7, 1))
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
                self.trigger(SCREW)
                self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CLAMPING_TOOL)
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
        self.trigger(SCREW)
        self.trigger(SCREW, 0.9)
        self.trigger(CONNECT)
        self.assertEqual([(entry["task_name"], entry["piece_id"]) for entry in self.manager.waiting_triggers],
                         [("Bring Tool", 1), ("Bring Connector", 1), ("Lift", 2)])
        self.reply("screw done")
        self.reply("yes")
        self.complete_robot()
        self.assertEqual(self.manager.active_task.task_id, 3)
        self.assertEqual(len(self.manager.waiting_triggers), 2)
        self.reply("yes")
        self.complete_robot()
        self.assertEqual(self.manager.active_task.task_id, 4)
        self.reply("yes")
        self.complete_robot()
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.piece_id), (config.TASK_LIFT_PANEL, 2))

    def test_trigger_rules_gate_robot_offers(self):
        for step in (PLACE, ALIGN):
            self.trigger(step, 1.0)
            self.assertIsNone(self.manager.active_task)
        self.trigger(PULL, 0.4)  # Below the database's 0.5 for Lift.
        self.assertIsNone(self.manager.active_task)
        self.trigger(PULL, 0.6)
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)

    def test_no_robot_offer_for_a_task_the_human_is_doing(self):
        self.trigger(LIFT, 0.2)
        self.trigger(PULL, 0.9)
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.tracker.get("Lift", 1).executor, "Human")

    def test_task_update_publisher_reports_step_changes_and_progress_moves(self):
        publisher = TaskUpdatePublisher(publish_delta=0.05, progress_scale=1.0)
        first = publisher.update(RecognitionResult(0, SCREW, 0.10, 0, 0.9, 0.0))
        self.assertEqual(first[0].event_type, E.HUMAN_TASK_UPDATE)
        self.assertEqual((first[0].payload["step_id"], first[0].payload["task_name"]), (SCREW, "Screw"))
        self.assertEqual(publisher.update(RecognitionResult(0, SCREW, 0.12, 0, 0.9, 0.1)), [])
        self.assertEqual(len(publisher.update(RecognitionResult(0, SCREW, 0.16, 0, 0.9, 0.2))), 1)
        self.assertEqual(len(publisher.update(RecognitionResult(0, CONNECT, 0.16, 0, 0.9, 0.3))), 1)
        scaled = TaskUpdatePublisher(progress_scale=100.0).update(RecognitionResult(0, SCREW, 150.0, 0, 1, 0))
        self.assertEqual(scaled[0].payload["progress"], 1.0)

    def test_speed_and_pause_controls_remain_available(self):
        self.trigger(SCREW)
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
        for step in (PULL, LIFT, PLACE, ALIGN, SCREW, CONNECT, CLAMP, CLAMP):
            result = recognition._result_from_passthrough({"step_id": step})
            self.assertEqual((result.round_id, result.piece_id), (0, 0))
        result = recognition._result_from_passthrough({"step_id": PULL})
        self.assertEqual((result.round_id, result.piece_id), (1, 1))

    def test_connector_refusal_requires_explicit_pending_execution(self):
        self.hold()
        self.reply("screw done")
        self.reply("yes")
        self.complete_robot()
        connector_id = self.manager.active_task.task_instance_id
        self.reply("no")
        # The queued Bring Tool offer comes next; refuse it too.
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CLAMPING_TOOL)
        self.reply("no")
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.udp.send.call_count, 2)
        self.assertEqual(self.status("Bring Connector"), T.PENDING)
        self.reply(f"execute {connector_id}")
        self.assertEqual(self.udp.send.call_args.args[0]["step_id"], 3)
        self.assertEqual(self.status("Bring Connector"), T.WORKING)

    # -- task tracking -----------------------------------------------------------

    def test_lift_chain_updates_tracked_tasks(self):
        self.trigger()
        self.assertEqual(self.status("Pull Cables"), T.WORKING)
        self.assertEqual(self.status("Lift"), T.PENDING)
        self.reply("yes")
        self.assertEqual(self.tracker.get("Lift", 1).executor, "Robot")
        self.assertEqual(self.status("Lift"), T.WORKING)
        self.complete_robot()
        self.assertEqual(self.status("Lift"), T.DONE)
        self.reply("yes")
        self.reply("adjustment done")
        self.assertEqual((self.status("Place"), self.tracker.get("Place", 1).executor), (T.DONE, "Human"))
        self.reply("screw done")
        for name in ("Pull Cables", "Align", "Screw"):
            self.assertEqual(self.status(name), T.DONE, name)
        self.assertTrue(self.tracker.get("Align", 1).inferred)

    def test_human_request_dispatches_without_trigger_or_permission(self):
        self.reply("bring the tool")
        task = self.manager.active_task
        self.assertEqual((task.task_id, task.state, task.piece_id), (config.TASK_BRING_CLAMPING_TOOL, S.R_ACCEPTED, 1))
        self.output.show_permission_request.assert_not_called()
        self.timer.start_response_timer.assert_not_called()
        self.assertEqual(self.udp.send.call_args.args[0]["suggested_action"], "bring_clamping_tool")
        self.assertEqual(self.status("Bring Tool"), T.WORKING)
        self.complete_robot()
        self.assertEqual(self.status("Bring Tool"), T.DONE)
        self.reply("bring the tool")  # Already done for piece 1 -> piece 2's.
        self.assertEqual(self.manager.active_task.piece_id, 2)

    def test_human_request_while_busy_runs_next(self):
        self.trigger()
        self.reply("bring the connector")
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)
        self.assertTrue(self.manager.waiting_triggers[0]["requested"])
        self.reply("no")  # Lift refused -> pooled, so the request runs now.
        task = self.manager.active_task
        self.assertEqual((task.task_id, task.state), (config.TASK_BRING_CONNECTOR, S.R_ACCEPTED))

    def test_robot_cannot_be_asked_for_human_only_or_unconfigured_task(self):
        self.manager.handle_event(Event(E.H_REQUEST_ROBOT_TASK, "test", payload={"task_name": "Screw"}))
        self.assertIsNone(self.manager.active_task)
        self.assertIn("cannot", self.output.show_message.call_args.args[0])

    def test_human_doing_offered_task_withdraws_the_offer(self):
        self.trigger(SCREW)
        offer = self.manager.active_task
        self.assertEqual(offer.task_id, config.TASK_BRING_CLAMPING_TOOL)
        self.reply("tool brought")
        self.assertEqual(offer.state, S.R_CANCELED)
        self.timer.cancel_response_timer.assert_called()
        self.assertEqual((self.status("Bring Tool"), self.tracker.get("Bring Tool", 1).executor), (T.DONE, "Human"))
        # The queued Bring Connector offer takes its place.
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CONNECTOR)
        self.reply("no")
        self.assertEqual(len(self.manager.pending_pool.list_all()), 1)
        self.reply("connector brought")
        self.assertEqual(self.manager.pending_pool.list_all(), [])
        self.assertEqual(self.status("Bring Connector"), T.DONE)

    def test_plain_done_without_robot_task_confirms_human_task(self):
        self.trigger(ALIGN, 0.3)
        self.reply("done")
        self.assertEqual(self.status("Align"), T.DONE)
        self.assertIn("Align is done", self.output.show_message.call_args.args[0])

    def test_status_line_reported_on_change(self):
        lines = []
        self.manager.status_callback = lines.append
        self.trigger(ALIGN, 0.3)
        self.trigger(ALIGN, 0.3)
        self.assertEqual(len(lines), 1)
        self.assertIn("working: Align", lines[0])


class CommandParserTests(unittest.TestCase):
    def test_task_commands_parse_with_task_names(self):
        parser = CommandParser()
        done = parser.parse("Tool Returned")
        self.assertEqual((done.event_type, done.payload), (E.H_TASK_DONE, {"task_name": "Bring back Tool"}))
        request = parser.parse("bring the connector")
        self.assertEqual((request.event_type, request.payload),
                         (E.H_REQUEST_ROBOT_TASK, {"task_name": "Bring Connector"}))
        self.assertEqual(parser.parse("next piece").event_type, E.H_NEXT_PIECE)
        self.assertIn("tool brought", CommandParser.phrases())
        self.assertIn("yes", CommandParser.phrases())


if __name__ == "__main__":
    unittest.main()
