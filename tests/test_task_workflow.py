"""Offline workflow checks; no microphone, network, or robot is started."""

import sys
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import ANY, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

import config
from cli_interface import CLIInterface
from cmd_parser import CommandParser
from communication_manager import CommunicationManager, ListeningMode
from demo_opening import build_demo_opening
from events import Event, EventType as E, RobotTaskState as S, TaskStatus as T
from gh_dispatcher import GHDispatcher
from message_manager import MessageManager
from models import RecognitionResult
from pending_task import PendingTaskPool
from recognition_manager import RecognitionManager
from state_machine import StateMachine
from task_manager import TaskManager
from task_tracker import build_task_tracking
from timer_manager import TimerManager
from trigger_manager import TaskUpdatePublisher

# By name: config.STEP_NAMES is the model head's order, which is not assembly order.
PULL, LIFT, PLACE, ALIGN, SCREW, CONNECT, CLAMP = (config.STEP_NAMES.index(name) for name in (
    "Pull Cables", "Lift", "Place", "Align", "Screw", "Connect Cables", "Clamp Coupling"))


class WorkflowHarness(unittest.TestCase):
    demo = None
    recognition_activation_s = 0.0

    def setUp(self):
        self.parser = CommandParser()
        self.timer, self.ros, self.output, self.udp = Mock(), Mock(), Mock(), Mock()
        self.tracker, self.policy = build_task_tracking(ROOT, logger=Mock())
        self.manager = TaskManager(
            StateMachine(), PendingTaskPool(), self.timer, MessageManager(),
            self.output, GHDispatcher(self.udp), self.ros, Mock(),
            task_tracker=self.tracker, trigger_policy=self.policy,
            demo=self.demo() if self.demo else None,
            recognition_activation_s=self.recognition_activation_s,
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

    def hold(self):
        self.trigger()
        self.reply("yes")
        self.complete_robot()  # arrived: free drive on, no question
        self.assertEqual(self.manager.active_task.state, S.R_FREE_DRIVE)
        self.ros.publish_free_drive.assert_called_with(True)
        self.reply("adjustment done")
        self.ros.publish_free_drive.assert_called_with(False)
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)

    def hand_over(self):
        """A bring task has arrived: take the item, then the robot leaves -- after its
        delay (config.HANDOVER_LEAVE_DELAY_S), or once allowed to."""
        brought = self.manager.active_task
        self.assertEqual(brought.state, S.R_WAITING_HANDOVER)
        self.reply("yes")
        leave = self.manager.active_task
        self.assertEqual(leave.task_id, config.TASK_LEAVE_HANDOVER)
        if brought.task_id in config.HANDOVER_LEAVE_DELAY_S:
            self.assertEqual(leave.state, S.R_DEFER)
            self.emit(E.DEFER_TIMEOUT, task_instance_id=leave.task_instance_id)
        else:
            self.assertEqual(leave.state, S.R_WAITING_RESPONSE)
            self.reply("yes")
        self.complete_robot()


class WorkflowTests(WorkflowHarness):
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
        self.hand_over()
        self.assertIsNone(self.manager.active_task)
        self.trigger(CONNECT, 0.3)  # Clamp Coupling is only believed after Connect Cables.
        self.trigger(CLAMP)
        # Clamp Coupling lets the robot prepare the next piece ("Piece id": "n + 1"):
        # pull its cables first...
        pull = self.manager.active_task
        self.assertEqual((pull.task_id, pull.piece_id), (config.TASK_PULL_CABLES, 2))
        self.reply("yes")
        self.complete_robot()
        # ...then lift its panel, as the "robot task" chain says -- asking first.
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.piece_id, lift.state),
                         (config.TASK_LIFT_PANEL, 2, S.R_WAITING_RESPONSE))
        self.assertEqual(self.manager.waiting_triggers, deque())  # took over the queued Lift offer
        messages = [call.args[0] for call in self.udp.send.call_args_list]
        self.assertEqual([message["step_id"] for message in messages], [1, 2, 3, 7, 6])
        self.assertEqual([message["human_step_id"] for message in messages],
                         [PULL, config.HUMAN_SCREW_DONE, config.HUMAN_SCREW_DONE, config.HUMAN_SCREW_DONE, CLAMP])
        for name in ("Lift", "Place", "Screw", "Bring Connector"):
            self.assertEqual(self.status(name), T.DONE, name)
        self.assertEqual((self.status("Pull Cables", 2), self.tracker.get("Pull Cables", 2).executor),
                         (T.DONE, "Robot"))

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

    def test_done_does_not_release_and_screw_done_releases_only_while_holding(self):
        self.trigger()
        self.reply("screw done")
        self.assertEqual(self.manager.active_task.task_id, 1)
        self.udp.send.assert_not_called()
        self.reply("yes")
        self.complete_robot()
        self.reply("screw done")  # rejected: the human is still adjusting
        self.assertEqual(self.manager.active_task.state, S.R_FREE_DRIVE)
        self.reply("done")  # adjustment done: holding
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)
        self.reply("done")  # does not release the panel
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)

    def test_screw_done_without_held_panel_confirms_screw_and_offers_no_leave(self):
        self.reply("screw done")
        self.assertEqual(self.status("Screw"), T.DONE)
        # Screw done unlocks Bring Connector as usual, but there is no panel to leave.
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CONNECTOR)
        self.assertNotIn(config.TASK_LEAVE, [config.TRACKED_TO_ROBOT_TASK[entry["task_name"]]
                                             for entry in self.manager.waiting_triggers])

    def test_leave_goes_first_ahead_of_offers_queued_while_holding(self):
        self.hold()
        # Half the screws counted unlocks Bring Connector while the robot still holds.
        self.emit(E.TASK_SIGNAL, payload={"task_name": "Screw", "piece_id": 1,
                                          "signal": "screw count", "value": 0.5})
        self.assertEqual([entry["task_name"] for entry in self.manager.waiting_triggers],
                         ["Bring Connector"])
        self.reply("screw done")
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.piece_id, leave.state),
                         (config.TASK_LEAVE, 1, S.R_WAITING_RESPONSE))
        self.reply("yes")
        self.complete_robot()
        self.assertTrue(any("moved away" in call.args[0] for call in self.output.show_message.call_args_list))
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CONNECTOR)

    def test_refusal_and_timeout_keep_leave_pending_without_proposing_connector(self):
        for event_type in (E.H_REFUSE, E.RESPONSE_TIMEOUT):
            with self.subTest(event_type=event_type):
                self.setUp()
                self.hold()
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
                self.assertEqual(leave.state, S.R_WAITING_RESPONSE)
                self.assertEqual(self.udp.send.call_count, 1)
                self.reply("yes")
                self.complete_robot()
                self.assertEqual(self.manager.active_task.task_id, 3)
                # Piece 2's Pull Cables and Lift (Connect Cables triggers both, with
                # "Piece id": "n + 1").
                self.assertEqual([(entry["task_name"], entry["piece_id"]) for entry in self.manager.waiting_triggers],
                                 [("Pull Cables", 2), ("Lift", 2)])

    def test_later_makes_the_task_pending_without_a_timer_and_old_timers_cannot_affect_it(self):
        self.hold()
        self.reply("screw done")
        leave = self.manager.active_task
        with patch.dict(config.TASK_TIMINGS, {3: {"response_timeout_seconds": 30}}):
            self.reply("later")
            self.timer.start_defer_timer.assert_not_called()  # no start in 5 s: pending
            self.assertEqual((leave.state, self.manager.pending_pool.list_all()), (S.R_PENDING, [leave]))
            self.assertIn('say or type "leave"', self.output.show_message.call_args.args[0])
            self.reply("leave")  # asked again, dispatched only on a fresh yes
            self.assertEqual(self.udp.send.call_count, 1)
            self.reply("yes")
            self.assertEqual(self.udp.send.call_count, 2)
            self.complete_robot()
            connector = self.manager.active_task
            self.timer.start_response_timer.assert_called_with(connector.task_instance_id, 30)
            self.emit(E.RESPONSE_TIMEOUT, task_instance_id=leave.task_instance_id)
            self.emit(E.DEFER_TIMEOUT, task_instance_id=leave.task_instance_id)
            self.assertIs(self.manager.active_task, connector)
            self.assertEqual(connector.state, S.R_WAITING_RESPONSE)

    def test_later_to_the_leave_keeps_holding_and_starts_nothing_else(self):
        self.hold()
        self.reply("screw done")
        leave = self.manager.active_task
        self.reply("later")
        self.emit(E.DEFER_TIMEOUT, task_instance_id=leave.task_instance_id)  # nothing to time out
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.manager.held_piece_id, 1)  # still holding the panel
        self.ros.publish_cancel.assert_not_called()
        self.assertEqual(self.udp.send.call_count, 1)  # only the lift was dispatched
        self.assertEqual(self.output.show_permission_request.call_count, 2)

    def test_no_to_a_task_the_human_can_do_hands_it_to_them(self):
        self.trigger()  # the lift is offered
        lift = self.manager.active_task
        self.reply("no")
        self.assertEqual((lift.state, self.manager.pending_pool.list_all()), (S.R_REFUSED, []))
        self.assertIsNone(self.manager.active_task)
        self.assertFalse(self.tracker.get("Lift", 1).robot_offered)  # the human's again
        self.assertIn("Okay, you do this one", self.output.show_message.call_args.args[0])
        self.trigger(PULL, 0.9)
        self.assertIsNone(self.manager.active_task)  # and the robot does not ask again

    def test_no_to_leaving_the_panel_keeps_it_pending(self):
        """Only the robot can leave the panel it holds: a no waits pending, as before."""
        self.hold()
        self.reply("screw done")
        leave = self.manager.active_task
        self.reply("no")
        self.assertEqual((leave.state, self.manager.pending_pool.list_all()), (S.R_REFUSED, [leave]))
        self.assertEqual(self.manager.held_piece_id, 1)

    def test_retry_leave_keeps_context_and_rejects_previous_attempt_timers(self):
        self.hold()
        self.trigger(SCREW)
        self.reply("screw done")
        leave = self.manager.active_task
        self.reply("later")  # pending until asked for: the robot keeps holding
        self.assertEqual(leave.state, S.R_PENDING)
        # Bring Connector, queued once when Screw was confirmed done.
        self.assertEqual([entry["task_name"] for entry in self.manager.waiting_triggers], ["Bring Connector"])
        self.emit(E.DEFER_TIMEOUT, task_instance_id=leave.task_instance_id)  # no timer: nothing starts
        self.assertEqual(self.udp.send.call_count, 1)
        self.reply("leave")  # asked again, with its context
        self.assertIs(self.manager.active_task, leave)
        self.assertEqual((leave.state, leave.task_id, leave.round_id, leave.piece_id),
                         (S.R_WAITING_RESPONSE, 2, 7, 1))
        self.reply("yes")
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
                self.reply("screw done")  # No panel held: offers Bring Connector.
                self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CONNECTOR)
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
                    self.manager.active_task.state, E.ROBOT_SUCCESS, 1), S.R_FREE_DRIVE)
                self.emit(E.ROBOT_SUCCESS)
                self.assertEqual(self.manager.active_task.state, S.R_FREE_DRIVE)
                self.ros.publish_free_drive.assert_called_once_with(True)

    def test_lift_can_still_ask_before_free_drive(self):
        with patch.object(config, "LIFT_ASKS_FREE_DRIVE", True):
            self.trigger()
            self.reply("yes")
            self.complete_robot()
            self.assertEqual(self.manager.active_task.state, S.R_WAITING_FREE_DRIVE)
            self.ros.publish_free_drive.assert_not_called()
            self.reply("yes")
            self.assertEqual(self.manager.active_task.state, S.R_FREE_DRIVE)

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
        queued = lambda: [(entry["task_name"], entry["piece_id"]) for entry in self.manager.waiting_triggers]
        self.hold()
        self.reply("screw done")  # Unlocks Bring Connector behind the leave.
        self.assertEqual(queued(), [("Bring Connector", 1)])
        self.reply("yes")
        self.emit(E.ROBOT_RUNNING)
        # While the robot moves away, Connect Cables passes 50%: piece 2's Pull Cables and
        # Lift wait as well -- once each, however often recognition repeats it.
        self.trigger(CONNECT)
        self.trigger(CONNECT, 0.9)
        self.assertEqual(queued(), [("Bring Connector", 1), ("Pull Cables", 2), ("Lift", 2)])
        self.emit(E.ROBOT_SUCCESS)
        # R3, the connector, comes straight after the leave.
        self.assertEqual(self.manager.active_task.task_id, 3)
        self.assertEqual(queued(), [("Pull Cables", 2), ("Lift", 2)])
        self.reply("yes")
        self.complete_robot()
        self.hand_over()
        pull = self.manager.active_task
        self.assertEqual((pull.task_id, pull.piece_id), (config.TASK_PULL_CABLES, 2))
        self.reply("yes")
        self.complete_robot()
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.piece_id), (config.TASK_LIFT_PANEL, 2))
        self.assertEqual(queued(), [])

    def test_trigger_rules_gate_robot_offers(self):
        for step in (PLACE, ALIGN):
            self.trigger(step, 1.0)
            self.assertIsNone(self.manager.active_task)
        self.trigger(PULL, 0.4)  # Below the database's 0.5 for Lift.
        self.assertIsNone(self.manager.active_task)
        self.trigger(PULL, 0.6)
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)

    def test_task_the_human_left_can_go_to_the_robot(self):
        self.trigger(LIFT, 0.2)
        self.assertEqual(self.tracker.get("Lift", 1).executor, "Human")
        self.trigger(PULL, 0.9)
        # The human works on one task at a time: Lift went back to pending, so the
        # rule on Pull Cables may offer it.
        self.assertEqual((self.status("Lift"), self.status("Pull Cables")), (T.PENDING, T.WORKING))
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)

    def test_recognition_the_table_does_not_expect_is_ignored(self):
        self.trigger(PULL, 0.3)
        self.trigger(CLAMP, 0.9)  # would offer Pull Cables and Lift for piece 2
        self.assertIsNone(self.manager.active_task)
        self.assertEqual((self.status("Pull Cables"), self.status("Clamp Coupling")), (T.WORKING, T.NOT_DONE))

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

    def test_task_update_publisher_carries_every_steps_scores_and_publishes_when_they_move(self):
        publisher = TaskUpdatePublisher(publish_delta=0.05, progress_scale=1.0, probability_delta=0.05)
        n = len(config.STEP_NAMES)
        probabilities, lanes = [0.1] * n, [0.2] * n
        first = publisher.update(RecognitionResult(0, SCREW, 0.1, 0, 0.9, 0.0, probabilities, lanes))
        self.assertEqual((first[0].payload["step_probabilities"], first[0].payload["step_progress"]),
                         (probabilities, lanes))
        jitter = [0.12] + [0.1] * (n - 1)
        self.assertEqual(publisher.update(RecognitionResult(0, SCREW, 0.1, 0, 0.9, 0.1, jitter, lanes)), [])
        moved = list(jitter)
        moved[PULL] = 0.3  # only the scores moved: the decision layer may now pick another step
        self.assertEqual(len(publisher.update(RecognitionResult(0, SCREW, 0.1, 0, 0.9, 0.2, moved, lanes))), 1)
        lane_moved = list(lanes)
        lane_moved[CONNECT] = 0.3
        self.assertEqual(len(publisher.update(RecognitionResult(0, SCREW, 0.1, 0, 0.9, 0.3, moved, lane_moved))), 1)
        without = publisher.update(RecognitionResult(0, CONNECT, 0.1, 0, 1.0, 0.4))
        self.assertNotIn("step_probabilities", without[0].payload)

    def test_speed_and_pause_controls_remain_available(self):
        self.reply("screw done")  # No panel held: offers Bring Tool.
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

    def test_later_connector_requires_explicit_pending_execution(self):
        self.hold()
        self.reply("screw done")
        self.reply("yes")
        self.complete_robot()
        connector_id = self.manager.active_task.task_instance_id
        self.reply("later")
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.udp.send.call_count, 2)
        self.assertEqual(self.status("Bring Connector"), T.PENDING)
        self.reply(f"execute {connector_id}")
        # Executing a pending task asks permission again rather than dispatching.
        self.assertEqual(self.manager.active_task.state, S.R_WAITING_RESPONSE)
        self.assertEqual(self.udp.send.call_count, 2)
        self.assertEqual(self.status("Bring Connector"), T.PENDING)
        self.reply("yes")
        self.assertEqual(self.udp.send.call_args.args[0]["step_id"], 3)
        self.assertEqual(self.status("Bring Connector"), T.WORKING)

    # -- task tracking -----------------------------------------------------------

    def who(self, name, piece=1):
        task = self.tracker.get(name, piece)
        return task.status, task.executor

    def test_robot_lift_leads_place_align_and_screw(self):
        self.trigger()
        self.assertEqual(self.status("Pull Cables"), T.WORKING)
        self.assertEqual(self.status("Lift"), T.PENDING)
        self.reply("yes")
        # The robot lifts the panel and carries it into place; the cables are pulled.
        self.assertEqual(self.who("Lift"), (T.WORKING, "Robot"))
        self.assertEqual(self.who("Place"), (T.WORKING, "Robot"))
        self.assertEqual(self.status("Pull Cables"), T.DONE)
        self.complete_robot()  # arrived at the assembly location; free drive: the human aligns
        self.assertEqual(self.who("Lift"), (T.DONE, "Robot"))
        self.assertEqual(self.who("Place"), (T.DONE, "Robot"))
        self.assertEqual(self.who("Align"), (T.WORKING, "Human"))
        self.reply("adjustment done")  # holding: the human screws
        self.assertEqual(self.who("Align"), (T.DONE, "Human"))
        self.assertFalse(self.tracker.get("Align", 1).inferred)
        self.assertEqual(self.who("Screw"), (T.WORKING, "Human"))
        self.assertEqual(self.tracker.reference_task, "Screw")
        self.reply("screw done")
        self.assertEqual(self.who("Screw"), (T.DONE, "Human"))

    def test_declining_free_drive_means_no_alignment_was_needed(self):
        with patch.object(config, "LIFT_ASKS_FREE_DRIVE", True):  # only asked then
            self.trigger()
            self.reply("yes")
            self.complete_robot()
            self.assertEqual(self.status("Align"), T.PENDING)
            self.reply("no")
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)
        self.assertEqual(self.who("Align"), (T.DONE, "Human"))
        self.assertTrue(self.tracker.get("Align", 1).inferred)
        self.assertEqual(self.who("Screw"), (T.WORKING, "Human"))

    def test_recognition_is_ignored_while_the_robot_lift_leads(self):
        self.trigger()
        self.reply("yes")
        self.trigger(ALIGN, 0.9)  # during the lift
        self.assertNotEqual(self.status("Align"), T.WORKING)
        self.complete_robot()
        self.reply("adjustment done")
        self.trigger(CONNECT, 0.9)  # during the holding: a misrecognition must not count
        self.assertNotEqual(self.status("Connect Cables"), T.WORKING)
        self.assertEqual(list(self.manager.waiting_triggers), [])
        self.assertEqual(self.tracker.reference_task, "Screw")
        self.reply("screw done")  # the holding ends: recognition counts again
        self.trigger(CONNECT, 0.2)
        self.assertEqual(self.status("Connect Cables"), T.WORKING)

    def test_lift_stopped_before_arriving_hands_lift_and_place_back(self):
        self.trigger()
        self.reply("yes")
        self.emit(E.ROBOT_RUNNING)
        self.ros.get_latest_joint_positions.return_value = None  # not safe to go home
        with patch.object(config, "RECOVERY_STOP_DELAY_SECONDS", 0):
            self.reply("cancel")
        self.assertEqual(self.manager.active_task.state, S.R_MANUAL_RECOVERY)
        self.assertIsNone(self.manager.lift_piece_id)  # recognition counts during recovery
        self.reply("done")
        self.assertEqual(self.status("Lift"), T.PENDING)
        self.assertEqual(self.status("Place"), T.PENDING)

    def test_human_request_skips_trigger_but_still_asks_permission(self):
        self.reply("bring the connector")
        task = self.manager.active_task
        self.assertEqual((task.task_id, task.state, task.piece_id),
                         (config.TASK_BRING_CONNECTOR, S.R_WAITING_RESPONSE, 1))
        self.output.show_permission_request.assert_called_once()
        self.timer.start_response_timer.assert_called_once_with(task.task_instance_id, ANY)
        self.udp.send.assert_not_called()
        self.assertEqual(self.status("Bring Connector"), T.PENDING)
        self.reply("yes")
        self.assertEqual(task.state, S.R_ACCEPTED)
        self.assertEqual(self.udp.send.call_args.args[0]["suggested_action"], "bring_pipe_connector")
        self.assertEqual(self.status("Bring Connector"), T.WORKING)
        self.complete_robot()
        self.assertEqual(self.status("Bring Connector"), T.WORKING)  # until handed over
        self.hand_over()
        self.assertEqual(self.status("Bring Connector"), T.DONE)
        self.reply("bring the connector")  # Already done for piece 1 -> piece 2's.
        self.assertEqual(self.manager.active_task.piece_id, 2)

    def test_the_robot_cannot_be_asked_for_the_tool_any_more(self):
        self.reply("bring the tool")  # taken out of the task database on purpose
        self.assertIsNone(self.manager.active_task)
        self.assertIn("cannot", self.output.show_message.call_args.args[0])

    def test_deferred_human_request_is_pooled_not_dispatched(self):
        self.reply("bring the connector")
        task = self.manager.active_task
        self.reply("later")
        self.assertIsNone(self.manager.active_task)
        self.assertTrue(self.manager.pending_pool.contains(task.task_instance_id))
        self.udp.send.assert_not_called()
        # Asking again re-offers the pooled task -- it still needs a yes.
        self.reply("bring the connector")
        self.assertIs(self.manager.active_task, task)
        self.assertEqual(task.state, S.R_WAITING_RESPONSE)
        self.assertEqual(self.output.show_permission_request.call_count, 2)
        self.udp.send.assert_not_called()
        self.reply("yes")
        self.assertEqual(task.state, S.R_ACCEPTED)
        self.assertEqual(self.udp.send.call_count, 1)

    def test_human_request_while_busy_is_asked_next(self):
        self.trigger()
        self.reply("bring the connector")
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)
        self.assertTrue(self.manager.waiting_triggers[0]["requested"])
        self.reply("no")  # Lift refused -> pooled, so the request is offered now.
        task = self.manager.active_task
        self.assertEqual((task.task_id, task.state), (config.TASK_BRING_CONNECTOR, S.R_WAITING_RESPONSE))
        self.udp.send.assert_not_called()
        self.reply("yes")
        self.assertEqual(task.state, S.R_ACCEPTED)

    def test_robot_cannot_be_asked_for_human_only_or_unconfigured_task(self):
        self.manager.handle_event(Event(E.H_REQUEST_ROBOT_TASK, "test", payload={"task_name": "Screw"}))
        self.assertIsNone(self.manager.active_task)
        self.assertIn("cannot", self.output.show_message.call_args.args[0])

    def test_human_doing_offered_task_withdraws_the_offer(self):
        self.reply("screw done")  # No panel held: offers Bring Connector.
        offer = self.manager.active_task
        self.assertEqual(offer.task_id, config.TASK_BRING_CONNECTOR)
        self.reply("connector brought")
        self.assertEqual(offer.state, S.R_CANCELED)
        self.timer.cancel_response_timer.assert_called()
        self.assertEqual((self.status("Bring Connector"), self.tracker.get("Bring Connector", 1).executor),
                         (T.DONE, "Human"))
        self.assertIsNone(self.manager.active_task)

    def test_human_doing_a_pending_task_drops_it(self):
        self.reply("screw done")
        self.reply("later")
        self.assertEqual(len(self.manager.pending_pool.list_all()), 1)
        self.reply("connector brought")
        self.assertEqual(self.manager.pending_pool.list_all(), [])
        self.assertEqual(self.status("Bring Connector"), T.DONE)

    def test_plain_done_without_robot_task_confirms_human_task(self):
        self.trigger(SCREW, 0.3)
        self.reply("done")
        self.assertEqual(self.status("Screw"), T.DONE)
        self.assertIn("Screw is done", self.output.show_message.call_args.args[0])

    def test_status_line_reported_on_change(self):
        lines = []
        self.manager.status_callback = lines.append
        self.trigger(SCREW, 0.3)
        self.trigger(SCREW, 0.3)
        self.assertEqual(len(lines), 1)
        self.assertIn("working: Screw", lines[0])


class NamedTimerTests(unittest.TestCase):
    def test_schedule_emits_once_and_cancel_stops_it(self):
        fired = []
        timers = TimerManager(fired.append)
        first, second = Event(E.SCHEDULED_OFFER, "test"), Event(E.SCHEDULED_OFFER, "test")
        timers.schedule("a", 0.05, first)
        timers.schedule("b", 0.05, second)
        timers.cancel("b")
        time.sleep(0.2)
        self.assertEqual(fired, [first])
        self.assertEqual(timers.timers, {})


class DemoOpeningTests(WorkflowHarness):
    demo = staticmethod(build_demo_opening)

    def setUp(self):
        super().setUp()
        self.emit(E.DEMO_START)
        self.assertEqual(self.manager.question, "start")
        self.reply("yes")  # Shall we start the assembly?

    def fire_scheduled_offer(self):
        """The timer the demo scheduled goes off; returns its delay."""
        name, delay, event = self.timer.schedule.call_args.args
        self.assertEqual(event.event_type, E.SCHEDULED_OFFER)
        self.manager.handle_event(event)
        return delay

    def robot_pulls_and_asks_about_the_lift(self):
        self.reply("yes")
        self.emit(E.ROBOT_RUNNING)
        self.assertEqual(self.fire_scheduled_offer(), config.DEMO_LIFT_ASK_AFTER_PULL_START_S)
        pull, lift = self.manager.active_task, self.manager.advance_task
        self.assertEqual((pull.task_id, pull.state), (config.TASK_PULL_CABLES, S.R_EXECUTING))
        self.assertEqual((lift.task_id, lift.piece_id), (config.TASK_LIFT_PANEL, 1))
        self.assertIn("lift the panel", self.output.show_permission_request.call_args.args[0])
        return pull, lift

    def test_no_to_the_pull_asks_to_move_on_instead_of_leaving_it_pending(self):
        self.reply("no")
        self.assertEqual(self.manager.question, "continue")
        self.assertEqual(self.manager.pending_pool.list_all(), [])  # the human does it
        self.assertIn("next step", self.output.show_permission_request.call_args.args[0])
        self.output.show_message.assert_not_called()  # no "task pending" before the question
        self.reply("yes")
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.piece_id, lift.state),
                         (config.TASK_LIFT_PANEL, 1, S.R_WAITING_RESPONSE))
        self.assertIsNone(self.manager.question)
        self.assertTrue(self.manager.demo_opening)

    def test_cables_pulled_moves_on_to_the_lift(self):
        self.reply("no")
        self.reply("no")
        self.reply("cables pulled")
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)
        self.assertEqual(self.status("Pull Cables"), T.DONE)
        self.timer.cancel.assert_any_call("offer Lift piece 1")

    def test_opens_with_pull_cables_and_ignores_recognition(self):
        pull = self.manager.active_task
        self.assertEqual((pull.task_id, pull.piece_id, pull.state),
                         (config.TASK_PULL_CABLES, 1, S.R_WAITING_RESPONSE))
        self.trigger(PULL, 0.9)
        self.trigger(CLAMP, 0.9)
        self.assertIsNone(self.tracker.reference_task)
        self.assertIs(self.manager.active_task, pull)
        self.emit(E.TASK_SIGNAL, payload={"task_name": "Screw", "piece_id": 1, "signal": "Done signal"})
        self.assertNotEqual(self.status("Screw"), T.DONE)

    def test_yes_to_the_lift_while_pulling_lifts_right_after_the_pull(self):
        pull, lift = self.robot_pulls_and_asks_about_the_lift()
        self.reply("pause")  # the pull keeps its controls
        self.assertEqual(pull.state, S.R_PAUSED)
        self.reply("resume")
        self.reply("yes")
        self.assertEqual((self.manager.active_task, pull.state, self.manager.advance_answer),
                         (pull, S.R_EXECUTING, "H_ACCEPT"))
        self.assertTrue(self.manager.demo_opening)
        self.emit(E.ROBOT_SUCCESS)
        self.assertIs(self.manager.active_task, lift)
        self.assertEqual(lift.state, S.R_ACCEPTED)
        self.assertEqual(self.udp.send.call_count, 2)  # the pull, then the lift at once
        self.assertEqual((self.status("Pull Cables"), self.status("Lift")), (T.DONE, T.WORKING))
        self.assertFalse(self.manager.demo_opening)
        # From here on the usual lift: arrived, free drive, holding.
        self.complete_robot()
        self.assertEqual(lift.state, S.R_FREE_DRIVE)

    def test_later_to_the_lift_while_pulling_makes_it_pending_until_asked(self):
        _, lift = self.robot_pulls_and_asks_about_the_lift()
        self.reply("later")
        self.assertIsNone(self.manager.advance_task)
        self.assertEqual((lift.state, self.manager.pending_pool.list_all()), (S.R_PENDING, [lift]))
        self.assertIn('say or type "lift the panel"', self.output.show_message.call_args.args[0])
        self.assertFalse(self.manager.demo_opening)
        self.emit(E.ROBOT_SUCCESS)
        self.assertIsNone(self.manager.active_task)  # nothing starts after the pull by itself
        self.timer.start_defer_timer.assert_not_called()
        self.reply("lift the panel")
        self.assertIs(self.manager.active_task, lift)
        self.assertEqual(lift.state, S.R_WAITING_RESPONSE)

    def test_no_to_the_pull_is_the_humans_task_not_a_pending_one(self):
        pull = self.manager.active_task
        self.reply("no")
        self.assertEqual((pull.state, self.manager.pending_pool.list_all()), (S.R_REFUSED, []))
        cables = self.tracker.get("Pull Cables", 1)
        self.assertEqual((cables.status, cables.executor, cables.robot_offered), (T.WORKING, "Human", False))
        self.assertEqual(self.manager.question, "continue")  # it speaks for the no
        self.assertTrue(self.manager.demo_opening)  # the lift is asked once the human has pulled

    def test_later_to_the_pull_makes_it_pending_and_ends_the_opening(self):
        pull = self.manager.active_task
        self.reply("later")
        self.assertEqual((pull.state, self.manager.pending_pool.list_all()), (S.R_PENDING, [pull]))
        self.timer.start_defer_timer.assert_not_called()
        self.assertFalse(self.manager.demo_opening)  # recognition takes over meanwhile
        self.reply("pull the cables")
        self.assertIs(self.manager.active_task, pull)
        self.assertEqual(pull.state, S.R_WAITING_RESPONSE)

    def test_unanswered_lift_is_asked_again_after_the_pull(self):
        _, lift = self.robot_pulls_and_asks_about_the_lift()
        self.emit(E.ROBOT_SUCCESS)
        self.assertIs(self.manager.active_task, lift)
        self.assertEqual(lift.state, S.R_WAITING_RESPONSE)
        self.timer.start_response_timer.assert_called_with(lift.task_instance_id, config.RESPONSE_TIMEOUT_SECONDS)
        self.reply("yes")
        self.assertEqual(lift.state, S.R_ACCEPTED)

    def test_no_to_the_lift_while_pulling_hands_it_to_the_human(self):
        _, lift = self.robot_pulls_and_asks_about_the_lift()
        self.reply("no")
        self.assertIsNone(self.manager.advance_task)
        self.assertEqual((lift.state, self.manager.pending_pool.list_all()), (S.R_REFUSED, []))
        self.assertEqual((self.status("Lift"), self.tracker.get("Lift", 1).executor), (T.WORKING, "Human"))
        self.assertFalse(self.manager.demo_opening)
        self.emit(E.ROBOT_SUCCESS)
        self.assertIsNone(self.manager.active_task)  # the lift is not asked again
        self.trigger(PLACE, 0.3)  # recognition counts again
        self.assertEqual(self.status("Place"), T.WORKING)

    def test_no_to_the_pull_waits_for_the_human_then_asks_about_the_lift(self):
        self.reply("no")
        self.assertEqual((self.status("Pull Cables"), self.tracker.get("Pull Cables", 1).executor),
                         (T.WORKING, "Human"))
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.manager.question, "continue")
        self.reply("no")  # not yet: finish the cables first
        delay = self.fire_scheduled_offer()
        self.assertEqual(delay, self.tracker.duration_limits["Pull Cables"] + config.DEMO_HUMAN_PULL_BUFFER_S)
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.state), (config.TASK_LIFT_PANEL, S.R_WAITING_RESPONSE))
        self.reply("no")
        self.assertEqual((self.status("Pull Cables"), self.status("Lift")), (T.DONE, T.WORKING))
        self.assertFalse(self.manager.demo_opening)
        # Screw detection and the trigger rules take over.
        self.trigger(PLACE, 0.5)
        self.trigger(ALIGN, 0.5)
        self.trigger(SCREW, 0.6)
        self.emit(E.TASK_SIGNAL, payload={"task_name": "Screw", "piece_id": 1, "signal": "Done signal"})
        self.assertEqual(self.status("Screw"), T.DONE)
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CONNECTOR)
        self.assertEqual(list(self.manager.waiting_triggers), [])

    def test_yes_to_the_lift_after_the_human_pulled(self):
        self.reply("no")
        self.reply("later")
        self.fire_scheduled_offer()
        self.reply("yes")
        self.assertEqual(self.manager.active_task.state, S.R_ACCEPTED)
        self.assertFalse(self.manager.demo_opening)

    def test_stopped_pull_ends_the_opening(self):
        self.reply("yes")
        self.emit(E.ROBOT_RUNNING)
        self.fire_scheduled_offer()
        self.reply("cancel")
        self.assertIsNone(self.manager.advance_task)
        self.assertFalse(self.manager.demo_opening)
        self.timer.cancel.assert_any_call("offer Lift piece 1")

    def test_pull_done_before_the_timer_asks_about_the_lift_as_usual(self):
        self.reply("yes")
        self.complete_robot()
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.state), (config.TASK_LIFT_PANEL, S.R_WAITING_RESPONSE))
        self.fire_scheduled_offer()  # comes late: nothing more
        self.assertIs(self.manager.active_task, lift)
        self.assertIsNone(self.manager.advance_task)


class SequenceSelectionTests(WorkflowHarness):
    """The human's step comes from recognition's scores against the task sequence."""

    def scored(self, step=PULL, progress=0.3, lanes=None, **named):
        payload = {"step_id": step, "round_id": 0, "progress": progress,
                   "step_probabilities": [named.get(name.replace(" ", "_"), 0.02)
                                          for name in config.STEP_NAMES]}
        if lanes is not None:
            payload["step_progress"] = lanes
        self.manager.handle_event(Event(E.HUMAN_TASK_UPDATE, "recognition", payload=payload))

    def test_the_expected_second_option_wins_over_a_ruled_out_first(self):
        self.tracker.start_task("Screw", 1)
        lanes = [0.0] * len(config.STEP_NAMES)
        lanes[SCREW] = 0.35
        self.scored(PULL, lanes=lanes, Pull_Cables=0.6, Screw=0.45)
        self.assertEqual((self.tracker.reference_task, self.tracker.reference_progress), ("Screw", 0.35))
        self.assertEqual(self.manager.last_recognition.model_task, "Pull Cables")
        logged = [c.args[0] for c in self.manager.logger.log_message.call_args_list]
        self.assertIn("Recognition step chosen by the task sequence.", logged)

    def test_without_scores_the_models_step_is_taken_as_before(self):
        self.tracker.start_task("Screw", 1)
        self.trigger(PULL, 0.8)  # the table allows Pull Cables after Screw (0.30): a repeat
        self.assertEqual(self.tracker.reference_task, "Pull Cables")

    def test_no_task_high_enough_changes_nothing(self):
        self.tracker.start_task("Screw", 1)
        self.scored(config.STEP_NAMES.index("Non Related Task"), Non_Related_Task=0.9, Screw=0.1)
        self.assertIsNone(self.manager.last_recognition.task_name)
        self.assertEqual((self.tracker.reference_task, self.status("Screw")), ("Screw", T.WORKING))

    def test_the_choice_is_made_while_the_lift_leads_for_the_detectors(self):
        self.hold()
        self.scored(PLACE, Place=0.7, Screw=0.5)  # Place is done: the robot placed the panel
        self.assertEqual(self.manager.last_recognition.task_name, "Screw")


class RecognitionWarmUpTests(WorkflowHarness):
    recognition_activation_s = 20.0

    def recognized(self, step, progress=0.8, source="recognition"):
        self.manager.handle_event(Event(E.HUMAN_TASK_UPDATE, source,
                                        payload={"step_id": step, "round_id": 0, "progress": progress}))

    def activate(self):
        """The warm-up timer goes off."""
        name, delay, event = self.timer.schedule.call_args.args
        self.assertEqual((name, delay, event.event_type), ("recognition active", 20.0, E.RECOGNITION_ACTIVE))
        self.manager.handle_event(event)

    def test_task_updates_count_only_after_the_warm_up(self):
        self.manager.handle_event(Event(E.HUMAN_LOCATION_UPDATE, "recognition",
                                        payload={"x": 0.0, "y": 0.0, "z": 0.0}))
        self.recognized(PULL, 0.9)  # would offer Lift
        self.assertIsNone(self.tracker.reference_task)
        self.assertIsNone(self.manager.active_task)
        self.timer.schedule.assert_called_once()  # from the first event only
        self.activate()
        self.recognized(PULL, 0.9)
        self.assertEqual(self.status("Pull Cables"), T.WORKING)
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)

    def test_typed_updates_are_not_held_back(self):
        self.recognized(SCREW, 0.3, source="manual_recognition")
        self.assertEqual(self.status("Screw"), T.WORKING)
        self.timer.schedule.assert_not_called()


class DemoStartQuestionTests(WorkflowHarness):
    demo = staticmethod(build_demo_opening)

    def test_asks_to_start_before_anything_else(self):
        self.emit(E.DEMO_START)
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.manager.question, "start")
        self.assertIn("start the assembly", self.output.show_permission_request.call_args.args[0])
        self.trigger(PULL, 0.9)  # recognition is ignored meanwhile
        self.assertIsNone(self.manager.active_task)

    def test_no_asks_again_later(self):
        self.emit(E.DEMO_START)
        self.reply("no")
        self.assertIsNone(self.manager.question)
        name, delay, event = self.timer.schedule.call_args.args
        self.assertEqual((name, delay, event.event_type),
                         ("question start", config.DEMO_START_REASK_S, E.DEMO_QUESTION))
        self.manager.handle_event(event)
        self.assertEqual(self.manager.question, "start")
        self.reply("yes")
        self.assertEqual(self.manager.active_task.task_id, config.TASK_PULL_CABLES)


class DemoWaitsForRecognitionTests(WorkflowHarness):
    demo = staticmethod(build_demo_opening)
    recognition_activation_s = 20.0

    def test_demo_opens_once_recognition_is_active(self):
        self.emit(E.DEMO_START)
        self.assertIsNone(self.manager.active_task)
        self.assertIsNone(self.manager.question)
        self.assertTrue(self.manager.in_opening)
        self.manager.handle_event(Event(E.HUMAN_TASK_UPDATE, "recognition",
                                        payload={"step_id": PULL, "round_id": 0, "progress": 0.1}))
        self.assertIsNone(self.manager.active_task)
        _, _, event = self.timer.schedule.call_args.args
        self.manager.handle_event(event)
        self.assertEqual(self.manager.question, "start")
        self.reply("yes")
        pull = self.manager.active_task
        self.assertEqual((pull.task_id, pull.state), (config.TASK_PULL_CABLES, S.R_WAITING_RESPONSE))
        self.assertTrue(self.manager.demo_opening)


# The connector is the only item the robot brings, and it leaves after it without
# asking (config.HANDOVER_LEAVE_DELAY_S). With no delay it asks first, as for any item.
ASK_BEFORE_LEAVING = patch.dict(config.HANDOVER_LEAVE_DELAY_S, {}, clear=True)


class HandoverTests(WorkflowHarness):
    def bring(self, request="bring the connector"):
        self.reply(request)
        self.reply("yes")
        self.complete_robot()
        return self.manager.active_task

    @ASK_BEFORE_LEAVING
    def test_yes_opens_the_gripper_then_asks_to_leave(self):
        connector = self.bring()
        self.assertEqual((connector.state, self.status("Bring Connector")), (S.R_WAITING_HANDOVER, T.WORKING))
        self.assertIn("Can I hand over the pipe coupling?", self.output.show_permission_request.call_args.args[0])
        self.ros.publish_gripper_open.assert_not_called()
        self.reply("yes")
        self.ros.publish_gripper_open.assert_called_once_with()
        self.assertEqual((connector.state, self.status("Bring Connector")), (S.R_DONE, T.DONE))
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.piece_id, leave.state),
                         (config.TASK_LEAVE_HANDOVER, 1, S.R_WAITING_RESPONSE))
        self.timer.start_response_timer.assert_called_with(leave.task_instance_id, ANY)
        self.reply("yes")
        self.assertEqual(self.udp.send.call_args.args[0]["suggested_action"], "leave_handover")
        self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.output.show_message.call_args.args[0],
                         MessageManager().get_left_handover_message())

    def test_not_ready_holds_the_item_until_the_human_asks_for_it(self):
        connector = self.bring("bring the connector")
        self.reply("no")
        self.assertEqual(connector.state, S.R_HOLDING_HANDOVER)
        self.assertIn('"give me the pipe coupling"', self.output.show_message.call_args.args[0])
        self.reply("yes")  # Not a request for the item.
        self.assertEqual(connector.state, S.R_HOLDING_HANDOVER)
        self.ros.publish_gripper_open.assert_not_called()
        self.reply("give me the pipe coupling")
        self.ros.publish_gripper_open.assert_called_once_with()
        self.assertEqual((connector.state, self.status("Bring Connector")), (S.R_DONE, T.DONE))
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LEAVE_HANDOVER)

    def test_connector_leaves_after_a_short_delay_without_asking(self):
        connector = self.bring("bring the connector")
        asked = self.output.show_permission_request.call_count
        self.reply("yes")
        self.ros.publish_gripper_open.assert_called_once_with()
        self.assertEqual(connector.state, S.R_DONE)
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.state), (config.TASK_LEAVE_HANDOVER, S.R_DEFER))
        self.assertEqual(self.output.show_permission_request.call_count, asked)  # no question
        self.assertIn("I'm moving away.", self.output.show_message.call_args.args[0])
        self.timer.start_defer_timer.assert_called_with(leave.task_instance_id, 1.0)
        sent = self.udp.send.call_count
        self.emit(E.DEFER_TIMEOUT, task_instance_id=leave.task_instance_id)
        self.assertEqual(self.udp.send.call_count, sent + 1)
        self.assertEqual(self.udp.send.call_args.args[0]["suggested_action"], "leave_handover")
        self.assertEqual(leave.state, S.R_ACCEPTED)

    def test_cancel_during_the_connector_delay_keeps_the_robot_there(self):
        self.bring("bring the connector")
        self.reply("yes")
        leave = self.manager.active_task
        sent = self.udp.send.call_count
        self.reply("cancel")
        self.assertEqual(leave.state, S.R_REFUSED)
        self.assertTrue(self.manager.pending_pool.contains(leave.task_instance_id))
        self.assertEqual(self.udp.send.call_count, sent)

    def test_asking_for_the_item_answers_the_hand_over_question(self):
        connector = self.bring()
        self.reply("give me the connector")
        self.ros.publish_gripper_open.assert_called_once_with()
        self.assertEqual(connector.state, S.R_DONE)

    @ASK_BEFORE_LEAVING
    def test_refused_leave_keeps_the_robot_there_and_holds_back_other_tasks(self):
        self.bring()
        self.reply("yes")
        leave = self.manager.active_task
        self.reply("no")
        self.assertIsNone(self.manager.active_task)
        self.assertIn("I will stay at the hand-over position.", self.output.show_message.call_args.args[0])
        self.reply("bring the connector")  # the next panel's: asked about once the robot has left
        self.assertIsNone(self.manager.active_task)
        self.assertEqual([(entry["task_name"], entry["piece_id"]) for entry in self.manager.waiting_triggers],
                         [("Bring Connector", 2)])
        self.emit(E.H_EXECUTE_PENDING_TASK, task_instance_id=leave.task_instance_id)
        self.assertEqual((self.manager.active_task, leave.state), (leave, S.R_WAITING_RESPONSE))
        self.reply("yes")
        self.complete_robot()
        connector = self.manager.active_task
        self.assertEqual((connector.task_id, connector.state), (config.TASK_BRING_CONNECTOR, S.R_WAITING_RESPONSE))

    @ASK_BEFORE_LEAVING
    def test_later_to_leaving_keeps_the_robot_there(self):
        self.bring()
        self.reply("yes")
        leave = self.manager.active_task
        self.reply("later")
        self.assertEqual(leave.state, S.R_PENDING)
        self.assertIn("I will stay here.", self.output.show_message.call_args.kwargs["speech"])
        self.ros.publish_cancel.assert_not_called()
        self.assertTrue(self.manager.pending_pool.contains(leave.task_instance_id))

    def test_cancel_while_holding_the_item_never_offers_return_home(self):
        for answer in ("none yet", "no"):
            with self.subTest(answer=answer):
                self.setUp()
                item = self.bring()
                if answer == "no":
                    self.reply("no")
                self.ros.get_latest_joint_positions.return_value = [0.0] * 6
                self.ros.get_latest_gripper_has_object.return_value = False  # stale open sample
                with patch.object(config, "RECOVERY_STOP_DELAY_SECONDS", 0):
                    self.reply("cancel")
                self.ros.publish_gripper_open.assert_not_called()
                self.assertEqual(item.state, S.R_MANUAL_RECOVERY)


class NamedRequestTests(WorkflowHarness):
    """Naming an action ("leave", "lift") instead of "execute <task instance id>"."""

    def refuse_leave(self):
        self.hold()
        self.reply("screw done")
        leave = self.manager.active_task
        self.assertEqual(leave.task_id, config.TASK_LEAVE)
        self.reply("no")
        self.assertIsNone(self.manager.active_task)
        return leave

    def test_pending_leave_is_asked_again_by_saying_leave(self):
        leave = self.refuse_leave()
        self.assertIn('say or type "leave"', self.output.show_message.call_args.args[0])
        self.reply("leave")
        self.assertIs(self.manager.active_task, leave)
        self.assertEqual((leave.state, leave.piece_id), (S.R_WAITING_RESPONSE, 1))
        self.reply("yes")
        self.assertEqual(self.udp.send.call_args.args[0]["suggested_action"], "leave")

    def test_naming_the_action_being_asked_about_answers_yes(self):
        self.hold()
        self.reply("screw done")
        leave = self.manager.active_task
        self.reply("leave")
        self.assertEqual(leave.state, S.R_ACCEPTED)

    def test_leave_while_holding_asks_to_release_the_panel(self):
        self.hold()
        self.reply("leave")
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.piece_id, leave.state),
                         (config.TASK_LEAVE, 1, S.R_WAITING_RESPONSE))
        self.assertEqual(self.status("Screw"), T.WORKING)  # recognition or "screw done" decides
        self.assertEqual(self.udp.send.call_count, 1)  # only the lift so far

    def test_leave_names_the_pending_hand_over_leave_too(self):
        self.reply("bring the connector")
        self.reply("yes")
        self.complete_robot()
        self.reply("yes")  # handed over: the robot leaves after its delay...
        leave = self.manager.active_task
        self.reply("cancel")  # ...unless told to stay: pending
        self.assertTrue(self.manager.pending_pool.contains(leave.task_instance_id))
        self.reply("leave")
        self.assertIs(self.manager.active_task, leave)
        self.assertEqual((leave.task_id, leave.state), (config.TASK_LEAVE_HANDOVER, S.R_WAITING_RESPONSE))

    def test_pending_lift_is_asked_again_by_name(self):
        self.trigger()
        lift = self.manager.active_task
        self.reply("later")
        self.assertIn('say or type "lift the panel"', self.output.show_message.call_args.args[0])
        self.reply("lift")
        self.assertIs(self.manager.active_task, lift)
        self.assertEqual(lift.state, S.R_WAITING_RESPONSE)

    def test_a_refused_lift_can_still_be_asked_for_by_name(self):
        self.trigger()
        self.reply("no")  # the human lifts it -- until they change their mind
        self.assertEqual(self.manager.pending_pool.list_all(), [])
        self.reply("lift")
        self.assertEqual((self.manager.active_task.task_id, self.manager.active_task.state),
                         (config.TASK_LIFT_PANEL, S.R_WAITING_RESPONSE))

    def test_leave_without_a_held_panel_or_pending_leave(self):
        self.reply("leave")
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.output.show_message.call_args.args[0], "I am not holding a panel.")


class CommandParserTests(unittest.TestCase):
    def test_every_pending_phrase_names_its_task(self):
        from message_manager import REQUEST_PHRASES

        parser = CommandParser()
        for task_id, phrase in REQUEST_PHRASES.items():
            with self.subTest(phrase=phrase):
                event = parser.parse(phrase)
                self.assertEqual(event.event_type, E.H_REQUEST_ROBOT_TASK)
                named = config.TRACKED_TO_ROBOT_TASK[event.payload["task_name"]]
                # "leave" names the leave from the panel; the hand-over leave is found by it.
                self.assertEqual(named, config.TASK_LEAVE if task_id == config.TASK_LEAVE_HANDOVER else task_id)

    def test_task_commands_parse_with_task_names(self):
        parser = CommandParser()
        done = parser.parse("Tool Returned")
        self.assertEqual((done.event_type, done.payload), (E.H_TASK_DONE, {"task_name": "Bring back Tool"}))
        request = parser.parse("bring the connector")
        self.assertEqual((request.event_type, request.payload),
                         (E.H_REQUEST_ROBOT_TASK, {"task_name": "Bring Connector"}))
        self.assertEqual(parser.parse("next piece").event_type, E.H_NEXT_PIECE)
        self.assertEqual(parser.parse("Give me the tool").event_type, E.H_HANDOVER)
        self.assertIn("tool brought", CommandParser.phrases())
        self.assertIn("yes", CommandParser.phrases())


if __name__ == "__main__":
    unittest.main()
