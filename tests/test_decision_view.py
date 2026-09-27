"""Live view of the decision layer: snapshot, rule explanations, timeline, server."""

import json
import sys
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

import config
from cmd_parser import CommandParser
from decision_view import DecisionView, TimelineLogger, build_snapshot
from events import Event, EventType as E, RobotTaskState as S
from gh_dispatcher import GHDispatcher
from message_manager import MessageManager
from pending_task import PendingTaskPool
from state_machine import StateMachine
from task_manager import TaskManager
from task_tracker import build_task_tracking
from task_transition_detector import build_detectors

PULL, SCREW = config.STEP_NAMES.index("Pull Cables"), config.STEP_NAMES.index("Screw")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()
        self.logger = TimelineLogger(Mock())
        tracker, policy = build_task_tracking(ROOT, logger=self.logger)
        self.manager = TaskManager(
            StateMachine(), PendingTaskPool(), Mock(), MessageManager(), Mock(),
            GHDispatcher(Mock()), Mock(), self.logger, task_tracker=tracker, trigger_policy=policy,
        )

    def emit(self, event_type, **payload):
        self.manager.handle_event(Event(event_type, "test", payload=payload))

    def reply(self, text):
        self.manager.handle_event(self.parser.parse(text))

    def snapshot(self):
        snapshot = build_snapshot(self.manager, timeline=self.logger, detectors=build_detectors())
        return json.loads(json.dumps(snapshot, default=str))  # what the page receives

    def cell(self, snapshot, task_name, piece_id=1):
        piece = next(p for p in snapshot["pieces"] if p["piece_id"] == piece_id)
        return next(t for t in piece["tasks"] if t["name"] == task_name)

    def rule(self, snapshot, task_name):
        return next(r for r in snapshot["rules"] if r["task"] == task_name)

    def test_shows_task_pool_robot_and_why_rules_offer_or_not(self):
        snapshot = self.snapshot()
        self.assertEqual(len(snapshot["pieces"]), 3)
        self.assertEqual(self.rule(snapshot, "Lift")["verdict"], "waiting for its previous task")

        self.emit(E.HUMAN_TASK_UPDATE, step_id=PULL, round_id=7, progress=0.8)
        snapshot = self.snapshot()
        self.assertEqual(self.cell(snapshot, "Pull Cables")["status"], "WORKING")
        self.assertEqual(self.cell(snapshot, "Pull Cables")["progress"], 0.8)
        self.assertEqual(snapshot["human"]["reference_task"], "Pull Cables")
        self.assertEqual(snapshot["robot"]["active"]["task"], "Lift")
        self.assertEqual(snapshot["robot"]["active"]["state"], "R_WAITING_RESPONSE")
        lift = self.rule(snapshot, "Lift")
        self.assertEqual((lift["triggering"], lift["verdict"], lift["offered"]),
                         (["Pull Cables"], "already offered", [1]))
        self.assertEqual(self.rule(snapshot, "Bring Tool")["verdict"], "waiting for its previous task")

    def test_condition_and_held_panel_explain_what_waits(self):
        self.emit(E.HUMAN_TASK_UPDATE, step_id=PULL, round_id=7, progress=0.8)
        self.reply("yes")
        self.emit(E.ROBOT_RUNNING)
        self.emit(E.ROBOT_SUCCESS)
        self.reply("adjustment done")  # free drive came on at arrival: holding, and the human screws
        self.emit(E.HUMAN_TASK_UPDATE, step_id=SCREW, round_id=7, progress=0.7)  # ignored now
        snapshot = self.snapshot()
        self.assertEqual(snapshot["robot"]["held_piece_id"], 1)
        self.assertEqual(self.cell(snapshot, "Screw")["status"], "WORKING")
        connector = self.rule(snapshot, "Bring Connector")
        self.assertEqual((connector["triggering"], connector["conditions_met"], connector["verdict"]),
                         (["Screw"], False, "waiting for its condition"))
        self.assertEqual(connector["condition_state"]["Screw"]["status"], "WORKING")
        self.assertIn("Recognition ignored while the robot's lift leads.",
                      [entry["text"] for entry in snapshot["timeline"]])
        self.assertEqual([d["name"] for d in snapshot["detectors"]], ["screw count", "force screw"])

    def test_leave_rule_says_when_no_panel_is_held(self):
        self.reply("screw done")  # Screw done, but the robot never held the panel
        snapshot = self.snapshot()
        leave = self.rule(snapshot, "Leave from the panel")
        self.assertEqual((leave["offers"], leave["verdict"]), ([], "the robot holds no panel of that piece"))
        self.assertEqual(snapshot["robot"]["active"]["task"], "Bring Tool")
        self.assertEqual([entry["task_name"] for entry in snapshot["robot"]["queue"]], ["Bring Connector"])

    def test_timeline_keeps_events_transitions_and_status_changes(self):
        self.emit(E.HUMAN_LOCATION_UPDATE, x=0, y=0, z=0)
        self.emit(E.HUMAN_TASK_UPDATE, step_id=PULL, round_id=7, progress=0.8)
        self.reply("yes")
        timeline = self.snapshot()["timeline"]
        texts = [entry["text"] for entry in timeline]
        self.assertIn("recognized Pull Cables at 0.80", texts)
        self.assertIn("Pull Cables (piece 1): PENDING -> WORKING", texts)
        self.assertIn("Lift (piece 1): R_WAITING_RESPONSE -> R_ACCEPTED", texts)
        self.assertFalse(any("LOCATION" in text for text in texts))
        self.assertEqual([e["seq"] for e in timeline], sorted(e["seq"] for e in timeline))
        self.assertTrue(self.logger.inner.log_event.called)  # still written to the log file


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.view = DecisionView("127.0.0.1", 0)
        self.addCleanup(self.view.close)

    def get(self, path):
        with urllib.request.urlopen(self.view.url.rstrip("/") + path, timeout=5) as response:
            return response.read().decode("utf-8")

    def test_serves_page_and_builds_snapshots_only_while_watched(self):
        self.assertIn("<title>Decision Layer</title>", self.get("/"))
        build = Mock(return_value={"time": 1.0, "pieces": []})
        self.view.update(build)
        build.assert_not_called()  # nobody has asked yet
        self.assertEqual(json.loads(self.get("/state.json")), {"waiting": True})
        self.view.update(build)
        self.assertEqual(json.loads(self.get("/state.json")), {"time": 1.0, "pieces": []})
        self.view.update(build)  # again straight away: throttled
        self.assertEqual(build.call_count, 1)

    def test_unknown_path_is_not_found(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.get("/nothing")
        self.assertEqual(raised.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
