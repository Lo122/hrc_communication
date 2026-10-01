"""Reactive mode: the robot acts only on the human's commands, for the tracked piece.
Offline; no microphone, network, or robot is started."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

import config
from cmd_parser import CommandParser
from communication_manager import CommunicationManager, ListeningMode
from decision_view import build_snapshot
from events import Event, EventType as E, RobotTaskState as S, TaskStatus as T
from gh_dispatcher import GHDispatcher
from message_manager import MessageManager
from pending_task import PendingTaskPool
from reactive_task_manager import ReactiveTaskManager
from state_machine import StateMachine
from task_database import PANEL_SECURED
from task_tracker import build_task_tracking
from voice_context import VoiceContext, build_instructions



class ReactiveHarness(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()
        self.timer, self.ros, self.output, self.udp = Mock(), Mock(), Mock(), Mock()
        self.tracker, _ = build_task_tracking(ROOT, logger=Mock())
        self.manager = ReactiveTaskManager(
            StateMachine(), PendingTaskPool(), self.timer, MessageManager(reactive=True),
            self.output, GHDispatcher(self.udp), self.ros, Mock(), task_tracker=self.tracker)

    def emit(self, event_type, **kwargs):
        self.manager.handle_event(Event(event_type, "test", **kwargs))

    def say(self, text):
        self.manager.handle_event(self.parser.parse(text))

    def complete_robot(self):
        self.emit(E.ROBOT_RUNNING)
        self.emit(E.ROBOT_SUCCESS)

    def active(self):
        task = self.manager.active_task
        return None if task is None else (task.task_id, task.piece_id, task.state)

    def dispatched(self):
        """(robot task id, piece id) of every task sent to the robot, in order."""
        return [(call.args[0]["step_id"], call.args[0]["piece_id"]) for call in self.udp.send.call_args_list]

    def said(self):
        return self.output.show_message.call_args.args[0]

    def queue(self):
        return [(entry["task_name"], entry["piece_id"]) for entry in self.manager.waiting_triggers]

    def hold(self):
        """Lift the next panel and hold it after the adjustment."""
        self.say("lift the panel")
        self.complete_robot()
        self.assertEqual(self.manager.active_task.state, S.R_FREE_DRIVE)
        self.say("done")
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)


class ReactiveCommandTests(ReactiveHarness):
    def test_a_command_starts_at_once_for_the_tracked_piece_without_asking(self):
        self.say("pull the cables")
        self.assertEqual(self.active(), (config.TASK_PULL_CABLES, 1, S.R_ACCEPTED))
        self.assertEqual(self.dispatched(), [(config.TASK_PULL_CABLES, 1)])
        self.output.show_permission_request.assert_not_called()
        self.timer.start_response_timer.assert_not_called()
        self.assertIn("the left panel (piece 1)", self.said())
        self.assertEqual((self.tracker.get("Pull Cables", 1).status, self.tracker.get("Pull Cables", 1).executor),
                         (T.WORKING, "Robot"))

    def test_nothing_is_offered_on_its_own(self):
        # Proactively, the cables being pulled would get the lift offered.
        self.say("cables pulled")
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.tracker.get("Pull Cables", 1).status, T.DONE)  # still tracked
        # ...nor does the lift follow the robot's pull, as its "robot task" chain would.
        self.say("pull the cables")  # the next panel's
        self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.dispatched(), [(config.TASK_PULL_CABLES, 2)])
        self.output.show_permission_request.assert_not_called()

    def test_without_a_camera_the_robots_tasks_and_the_humans_words_find_the_piece(self):
        # Piece 1, with the robot's help.
        self.say("pull the cables")
        self.complete_robot()
        self.hold()  # its lift says the cables are pulled and the panel placed
        self.say("screw done")
        self.say("give me the connector")  # for piece 1, once the robot has let go of it
        self.assertEqual(self.queue(), [("Bring Connector", 1)])
        self.say("leave")
        self.complete_robot()
        self.assertEqual(self.active(), (config.TASK_BRING_CONNECTOR, 1, S.R_ACCEPTED))
        self.complete_robot()
        self.say("give me the connector")
        self.emit(E.DEFER_TIMEOUT, task_instance_id=self.manager.active_task.task_instance_id)
        self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        # The human connects and clamps piece 1: the robot prepares the next panel.
        self.say("pull the cables")
        self.assertEqual(self.active(), (config.TASK_PULL_CABLES, 2, S.R_ACCEPTED))
        self.assertIn("the middle panel (piece 2)", self.said())
        self.complete_robot()
        self.hold()  # piece 2: holding it means the human has moved on from piece 1
        self.say("give me the connector")
        self.assertEqual(self.queue(), [("Bring Connector", 2)])
        self.assertEqual(self.dispatched(), [
            (config.TASK_PULL_CABLES, 1), (config.TASK_LIFT_PANEL, 1), (config.TASK_LEAVE, 1),
            (config.TASK_BRING_CONNECTOR, 1), (config.TASK_LEAVE_HANDOVER, 1),
            (config.TASK_PULL_CABLES, 2), (config.TASK_LIFT_PANEL, 2)])
        self.assertEqual(self.output.show_permission_request.call_count, 1)  # the hand-over only

    def test_queued_commands_run_in_order(self):
        self.say("pull the cables")
        self.say("lift the panel")
        self.say("give me the connector")
        self.assertEqual(self.queue(), [("Lift", 1), ("Bring Connector", 1)])
        self.assertIn("after the current task", self.said())
        self.complete_robot()
        self.assertEqual(self.active(), (config.TASK_LIFT_PANEL, 1, S.R_ACCEPTED))
        self.output.show_permission_request.assert_not_called()

    def test_the_robot_holds_the_panel_until_told_to_leave(self):
        self.hold()
        self.say("screw done")
        self.assertEqual(self.active(), (config.TASK_LIFT_PANEL, 1, S.R_HOLDING))
        self.assertEqual(self.tracker.get("Screw", 1).status, T.DONE)
        self.assertIn('"leave"', self.said())
        # "lift the panel" now means the next one, after this.
        self.say("lift the panel")
        self.assertEqual(self.queue(), [("Lift", 2)])
        self.say("leave")
        self.assertEqual(self.active(), (config.TASK_LEAVE, 1, S.R_ACCEPTED))
        self.complete_robot()
        self.assertIsNone(self.manager.held_piece_id)
        # No connector offered after the leave: the queued lift is next.
        self.assertEqual(self.active(), (config.TASK_LIFT_PANEL, 2, S.R_ACCEPTED))
        self.assertEqual(self.dispatched(), [(config.TASK_LIFT_PANEL, 1), (config.TASK_LEAVE, 1),
                                             (config.TASK_LIFT_PANEL, 2)])
        self.output.show_permission_request.assert_not_called()

    def test_leave_before_screw_done_leaves_screw_open(self):
        self.hold()
        self.say("leave")
        self.assertEqual(self.active(), (config.TASK_LEAVE, 1, S.R_ACCEPTED))
        self.assertEqual(self.tracker.get("Screw", 1).status, T.WORKING)

    def test_a_panel_secured_signal_does_not_release_the_panel(self):
        self.hold()
        self.emit(E.TASK_SIGNAL, payload={"task_name": "Screw", "piece_id": 1,
                                          "signal": PANEL_SECURED, "value": True})
        self.assertEqual(self.active(), (config.TASK_LIFT_PANEL, 1, S.R_HOLDING))
        self.assertEqual(self.tracker.get("Screw", 1).signals[PANEL_SECURED], True)
        self.assertEqual(self.dispatched(), [(config.TASK_LIFT_PANEL, 1)])

    def test_give_me_the_connector_brings_it_then_hands_it_over(self):
        self.say("give me the connector")
        self.assertEqual(self.active(), (config.TASK_BRING_CONNECTOR, 1, S.R_ACCEPTED))
        self.complete_robot()
        self.assertEqual(self.manager.active_task.state, S.R_WAITING_HANDOVER)  # holds it out
        self.say("give me the connector")
        self.ros.publish_gripper_open.assert_called_once()
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.state), (config.TASK_LEAVE_HANDOVER, S.R_DEFER))
        self.emit(E.DEFER_TIMEOUT, task_instance_id=leave.task_instance_id)
        self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        self.assertEqual(self.tracker.get("Bring Connector", 1).status, T.DONE)

    def test_the_robot_is_kept_at_the_hand_over_until_told_to_leave(self):
        self.say("give me the connector")
        self.complete_robot()
        self.say("yes")
        self.say("cancel")  # during the leave's delay
        self.assertIsNone(self.manager.active_task)
        self.say("pull the cables")  # waits: the robot is still at the hand-over position
        self.assertEqual(self.queue(), [("Pull Cables", 1)])
        self.say("leave")
        self.assertEqual(self.active(), (config.TASK_LEAVE_HANDOVER, 1, S.R_ACCEPTED))
        self.complete_robot()
        self.assertEqual(self.active(), (config.TASK_PULL_CABLES, 1, S.R_ACCEPTED))
        self.assertEqual(self.output.show_permission_request.call_count, 1)  # the hand-over only

    def test_a_queued_command_done_meanwhile_is_skipped(self):
        self.say("pull the cables")
        self.say("lift the panel")
        self.assertEqual(self.queue(), [("Lift", 1)])
        self.say("lifted")  # the human lifted it themselves
        self.complete_robot()
        self.assertIsNone(self.manager.active_task)
        self.assertIn("after all", self.said())
        self.assertEqual(self.dispatched(), [(config.TASK_PULL_CABLES, 1)])

    def test_repeated_and_impossible_commands(self):
        self.say("leave")
        self.assertEqual(self.said(), "I am not holding a panel.")
        self.say("bring the tool")  # not in the task database
        self.assertEqual(self.said(), "Sorry, I cannot do Bring Tool.")
        self.say("pull the cables")
        self.say("pull the cables")
        self.assertEqual(self.said(), "I am already on it.")
        self.say("lift the panel")
        self.say("lift the panel")
        self.assertIn("already", self.said())
        self.assertEqual(self.queue(), [("Lift", 1)])
        self.assertEqual(self.dispatched(), [(config.TASK_PULL_CABLES, 1)])

    def test_every_piece_done_leaves_nothing_to_command(self):
        for piece_id in self.tracker.piece_ids:
            self.tracker.set_manually("Pull Cables", piece_id, T.DONE, "Human")
        self.say("pull the cables")
        self.assertIsNone(self.manager.active_task)
        self.assertIn("already done", self.said())

    def test_refuses_a_trigger_policy(self):
        tracker, policy = build_task_tracking(ROOT, logger=Mock())
        with self.assertRaises(ValueError):
            ReactiveTaskManager(StateMachine(), PendingTaskPool(), Mock(), MessageManager(), Mock(),
                                GHDispatcher(Mock()), Mock(), Mock(), task_tracker=tracker, trigger_policy=policy)

    def test_live_view_shows_the_mode(self):
        self.assertEqual(build_snapshot(self.manager)["mode"], "reactive")


class ReactiveInputTests(unittest.TestCase):
    def test_asking_for_an_item_names_its_task(self):
        parser = CommandParser()
        event = parser.parse("Give me the connector")
        self.assertEqual((event.event_type, event.payload), (E.H_HANDOVER, {"task_name": "Bring Connector"}))
        self.assertEqual(parser.parse("hand over").payload, {})
        lift = parser.parse("lift a panel")
        self.assertEqual((lift.event_type, lift.payload), (E.H_REQUEST_ROBOT_TASK, {"task_name": "Lift"}))
        self.assertIn("give me the connector", CommandParser.phrases())

    def test_voice_listens_while_idle_only_in_reactive_mode(self):
        for reactive, mode in ((True, ListeningMode.CONTINUOUS), (False, ListeningMode.OFF)):
            with self.subTest(reactive=reactive):
                voice = Mock()
                comm = CommunicationManager(Mock(), CommandParser(), voice, Mock(), Mock(), lambda: None,
                                            guard_seconds=0,
                                            context_provider=lambda: VoiceContext(None, reactive=reactive))
                comm.poll()
                self.assertEqual(comm.mode, mode)
                self.assertEqual(voice.start_listening.called, reactive)

    def test_reactive_voice_instructions_permit_commands(self):
        idle = build_instructions(VoiceContext(None, reactive=True))
        self.assertIn("lift the panel", idle.split("Permitted commands:")[1])
        holding = build_instructions(VoiceContext(S.R_HOLDING, 1, "lift_1", reactive=True))
        self.assertIn("leave", holding.split("Permitted commands:")[1])
        self.assertNotIn("separate permission question", holding)
        proactive = build_instructions(VoiceContext(None))
        self.assertNotIn("lift the panel", proactive)


class ReactiveLaunchTests(unittest.TestCase):
    def test_the_launcher_starts_communication_alone(self):
        import run_system
        process = Mock(pid=1, returncode=0, poll=Mock(return_value=0))
        with patch.object(sys, "argv", ["run_system.py", "--reactive", "--no-run-log"]), \
                patch.object(run_system, "_open_window", return_value=process) as open_window, \
                contextlib.redirect_stdout(io.StringIO()):
            run_system.main()
        # No recognition (camera), no live TCP reader.
        open_window.assert_called_once()
        script, script_args = open_window.call_args.args
        self.assertEqual(script, run_system.COMMUNICATION_SCRIPT)
        self.assertIn("--reactive", script_args)

    def test_the_launcher_records_the_camera_and_stops_the_recording_properly(self):
        import run_system

        class FakeRecorder:
            """Exits once asked through its stop file, as run_recorder.py does."""
            pid, returncode, terminated, asked = 2, None, False, False

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.asked = stop_file.exists()
                if not self.asked:
                    raise subprocess.TimeoutExpired("run_recorder.py", timeout)
                self.returncode = 0
                return 0

            def terminate(self):
                self.terminated = True

        recorder = FakeRecorder()
        communication = Mock(pid=1, returncode=0, poll=Mock(return_value=0))  # the run ends
        with tempfile.TemporaryDirectory() as log_dir:
            stop_file = Path(log_dir) / "r1" / run_system.RECORDER_STOP_FILE
            argv = ["run_system.py", "--reactive", "--record-camera", "--video-source", "6",
                    "--log-dir", log_dir, "--run-name", "r1"]
            with patch.object(sys, "argv", argv), \
                    patch.object(run_system, "_open_window", side_effect=lambda script, _args: (
                        recorder if script == run_system.RECORDER_SCRIPT else communication)) as open_window, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_system.main()
            self.assertFalse(stop_file.exists())  # cleaned up
        # The recorder first, then communication: no recognition, no live TCP reader.
        self.assertEqual([call.args[0] for call in open_window.call_args_list],
                         [run_system.RECORDER_SCRIPT, run_system.COMMUNICATION_SCRIPT])
        recorder_args = open_window.call_args_list[0].args[1]
        self.assertEqual(recorder_args[-2:], ["--video-source", "6"])
        self.assertIn(str(stop_file), recorder_args)
        self.assertTrue(recorder.asked)  # stopped through its stop file, so the .mp4 is finished...
        self.assertFalse(recorder.terminated)  # ...not killed

    def test_camera_flags_are_refused_in_reactive_mode(self):
        import run_communication
        import run_system
        for argv in (["run_system.py", "--reactive", "--camera"],
                     ["run_system.py", "--reactive", "--demo"],
                     ["run_system.py", "--record-camera", "--camera"],  # recognition records itself
                     ["run_system.py", "--reactive", "--record-camera", "--no-run-log"],
                     ["run_communication.py", "--reactive", "--demo"]):
            parse = run_system._parse_args if argv[0] == "run_system.py" else run_communication._parse_args
            with self.subTest(argv=argv), patch.object(sys, "argv", argv), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse()
        with patch.object(sys, "argv", ["run_communication.py", "--reactive"]):
            self.assertTrue(run_communication._parse_args().reactive)

    def test_the_runtime_runs_no_detectors_and_reads_no_tcp_force(self):
        import communication_runtime as runtime
        with tempfile.TemporaryDirectory() as run_dir, \
                patch.object(config, "VOICE_ENABLED", False), \
                patch.object(config, "DECISION_VIEW_PORT", None), \
                patch.object(config, "TASK_DETECTORS_MODE", "on"), \
                patch.object(runtime, "ROSCommunication") as ros, \
                patch.object(runtime, "UDPSender"), \
                contextlib.redirect_stdout(io.StringIO()):
            system = runtime.build_system(reactive=True, run_dir=run_dir)
            try:
                self.assertIsInstance(system.task_manager, ReactiveTaskManager)
                self.assertIsNone(system.detectors)
                self.assertIsNone(ros.call_args.kwargs["wrench_topic"])
                self.assertEqual(system._current_voice_context(), VoiceContext(None, reactive=True))
            finally:
                system.close()


if __name__ == "__main__":
    unittest.main()
