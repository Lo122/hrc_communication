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
        self.assertIn("4=Screw", error.getvalue())

    def test_normal_startup_does_not_require_debug_mapping(self):
        with patch.object(sys, "argv", ["run_communication.py"]):
            self.assertFalse(_parse_args().debug_trigger)


if __name__ == "__main__":
    unittest.main()
