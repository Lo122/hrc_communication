"""Run logs: communication's timestamped event log and timeline, recognition's
per-run CSVs, and the launcher giving both processes one run directory."""

import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition/eval"):
    sys.path.insert(0, str(ROOT / layer))

from events import Event, EventType as E, RobotTaskState as S
from logger import EventLogger
from run_logger import RunLogger


class EventLoggerTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.logger = EventLogger(self.dir / "communication_events.jsonl",
                                  timeline_path=self.dir / "timeline.csv")

    def read(self):
        self.logger.close()
        lines = [json.loads(line) for line in (self.dir / "communication_events.jsonl").open(encoding="utf-8")]
        with (self.dir / "timeline.csv").open(encoding="utf-8", newline="") as handle:
            return lines, list(csv.DictReader(handle))

    def test_records_are_timestamped_and_the_timeline_shows_the_reaction(self):
        seen = time.time() - 0.25
        update = Event(E.HUMAN_TASK_UPDATE, "recognition",
                       payload={"step_id": 0, "task_name": "Pull Cables", "progress": 0.62, "timestamp": seen})
        self.logger.log_event(update)
        self.logger.log_event(Event(E.HUMAN_LOCATION_UPDATE, "recognition", payload={"x": 1.0}))
        task = SimpleNamespace(task_instance_id="round_0_task_1_piece_1")
        self.logger.log_transition(task, Event(E.H_ACCEPT, "human_voice"), S.R_WAITING_RESPONSE,
                                   S.R_ACCEPTED, "Human accepted task.")
        self.logger.log_message("Task status.", {"line": "[tasks] piece 1"})
        lines, rows = self.read()

        self.assertEqual(len(lines), 4)  # the JSON lines keep everything
        for line in lines:
            self.assertAlmostEqual(line["t"], time.time(), delta=5)
            self.assertRegex(line["time"], r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}$")
        self.assertEqual(lines[0]["event_t"], round(update.timestamp, 3))

        self.assertEqual([row["what"] for row in rows],
                         ["HUMAN_TASK_UPDATE", "R_WAITING_RESPONSE -> R_ACCEPTED", "Task status."])
        self.assertEqual(rows[0]["detail"], "Pull Cables 0.62")
        self.assertGreaterEqual(float(rows[0]["delay_ms"]), 250)
        self.assertRegex(rows[0]["recognized_at"], r"^\d\d:\d\d:\d\d\.\d{3}$")
        self.assertEqual((rows[1]["source"], rows[1]["detail"]), ("H_ACCEPT", "Human accepted task."))

    def test_video_time_is_not_taken_for_a_time_of_day(self):
        self.logger.log_event(Event(E.HUMAN_TASK_UPDATE, "recognition",
                                    payload={"step_id": 4, "progress": 0.1, "timestamp": 12.5}))
        _, rows = self.read()
        self.assertEqual((rows[0]["recognized_at"], rows[0]["delay_ms"], rows[0]["detail"]),
                         ("", "", "step 4 0.10"))


class RecognitionRunLogTests(unittest.TestCase):
    def test_frames_and_events_carry_wall_clock_time(self):
        run = RunLogger(tempfile.mkdtemp(), "run_x")
        manager = SimpleNamespace(last_frame_record={"detected": True, "warmup": False},
                                  playback_frame_index=None, playback_dropped_before=0)
        run.log_frame(manager, update_s=0.01)
        run.log_event(Event(E.HUMAN_TASK_UPDATE, "recognition",
                            payload={"step_id": 0, "task_name": "Pull Cables", "progress": 0.5}))
        run.close()
        for name in ("frames.csv", "events.csv"):
            with (run.run_dir / name).open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["epoch_s"]), time.time(), delta=5)
            self.assertRegex(row["time"], r"^\d\d:\d\d:\d\d\.\d{3}$")
        self.assertEqual(row["task_name"], "Pull Cables")


class LauncherRunLogTests(unittest.TestCase):
    def test_both_processes_get_one_run_directory(self):
        import run_system
        with patch.object(sys, "argv", ["run_system.py", "--camera"]):
            args, recognition_args = run_system._parse_args()
        self.assertRegex(args.run_name, r"^run_\d{8}_\d{6}$")
        run_log = ["--log-dir", args.log_dir, "--run-name", args.run_name]
        self.assertEqual(run_system._run_log_args(args), run_log)
        self.assertEqual(run_system._communication_args(args)[:4], run_log)
        with patch.object(sys, "argv", ["run_system.py", "--no-run-log", "--camera"]):
            args, _ = run_system._parse_args()
        self.assertEqual(run_system._run_log_args(args), ["--no-run-log"])

    def test_both_scripts_accept_the_run_log_flags(self):
        import run_communication
        with patch.object(sys, "argv", ["run_communication.py", "--log-dir", "x", "--run-name", "r"]):
            args = run_communication._parse_args()
        self.assertEqual((args.log_dir, args.run_name, args.no_run_log), ("x", "r", False))
        import run_recognition
        with patch.object(sys, "argv", ["run_recognition.py", "--no-run-log", "--run-name", "r"]):
            args = run_recognition._parse_args()
        self.assertEqual((args.no_run_log, args.run_name), (True, "r"))


if __name__ == "__main__":
    unittest.main()
