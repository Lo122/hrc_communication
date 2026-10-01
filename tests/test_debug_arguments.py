"""Validate debug inputs before any communication hardware is initialized."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from run_communication import _parse_args


class DebugArgumentTests(unittest.TestCase):
    def test_every_model_step_is_accepted(self):
        for step in range(7):
            with self.subTest(step=step), patch.object(
                sys, "argv", ["run_communication.py", "--debug-trigger", "--debug-step-id", str(step)]
            ):
                self.assertEqual(_parse_args().debug_step_id, step)

    def test_unknown_step_lists_the_steps(self):
        error = io.StringIO()
        with patch.object(sys, "argv", ["run_communication.py", "--debug-trigger", "--debug-step-id", "9"]):
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                _parse_args()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Human step 9 does not exist", error.getvalue())
        self.assertIn(f"{config.STEP_NAMES.index('Screw')}=Screw", error.getvalue())

    def test_normal_startup_does_not_require_debug_mapping(self):
        with patch.object(sys, "argv", ["run_communication.py"]):
            self.assertFalse(_parse_args().debug_trigger)

    def test_demo_flag_reaches_communication_through_the_launcher(self):
        import run_system
        with patch.object(sys, "argv", ["run_communication.py", "--demo"]):
            self.assertTrue(_parse_args().demo)
        with patch.object(sys, "argv", ["run_system.py", "--demo", "--camera"]):
            args, recognition_args = run_system._parse_args()
        self.assertIn("--demo", run_system._communication_args(args))
        self.assertEqual(recognition_args, ["--camera"])

    def test_the_launcher_starts_the_live_tcp_reader_with_the_run_log(self):
        import run_system
        with patch.object(sys, "argv", ["run_system.py", "--robot-ip", "192.168.1.10",
                                        "--run-name", "r1", "--camera"]):
            args, recognition_args = run_system._parse_args()
        self.assertEqual(recognition_args, ["--camera"])  # the robot flags stay with the launcher
        self.assertFalse(args.no_robot_live)
        live = run_system._robot_live_args(args)
        self.assertEqual(live[:5], ["--ip", "192.168.1.10", "--publish-ros", "--ros-hz",
                                    str(config.ROBOT_LIVE_DATA_ROS_HZ)])
        self.assertEqual(Path(live[live.index("--log-file") + 1]),
                         Path(args.log_dir) / "r1" / run_system.ROBOT_LIVE_LOG)
        self.assertTrue(run_system.ROBOT_LIVE_SCRIPT.exists())

    def test_the_live_reader_can_be_skipped_and_logs_nowhere_without_a_run_log(self):
        import run_system
        with patch.object(sys, "argv", ["run_system.py", "--no-robot-live", "--no-run-log",
                                        "--video-source", "clip.mp4"]):
            args, recognition_args = run_system._parse_args()
        self.assertTrue(args.no_robot_live)
        self.assertEqual(args.robot_ip, config.ROBOT_IP)
        self.assertNotIn("--log-file", run_system._robot_live_args(args))
        self.assertEqual(recognition_args, ["--video-source", "clip.mp4"])


if __name__ == "__main__":
    unittest.main()
