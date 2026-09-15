"""Validate debug inputs before any communication hardware is initialized."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_communication import _parse_args


class DebugArgumentTests(unittest.TestCase):
    def test_configured_human_steps_are_accepted(self):
        for step in (0, 4, 5):
            with self.subTest(step=step), patch.object(
                sys, "argv", ["run_communication.py", "--debug-trigger", "--debug-step-id", str(step)]
            ):
                self.assertEqual(_parse_args().debug_step_id, step)

    def test_human_step_three_explains_command_only_workflow(self):
        error = io.StringIO()
        with patch.object(sys, "argv", ["run_communication.py", "--debug-trigger", "--debug-step-id", "3"]):
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                _parse_args()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Human step 3 has no recognition trigger", error.getvalue())
        self.assertIn("screw done", error.getvalue())

    def test_normal_startup_does_not_require_debug_mapping(self):
        with patch.object(sys, "argv", ["run_communication.py"]):
            self.assertFalse(_parse_args().debug_trigger)


if __name__ == "__main__":
    unittest.main()
