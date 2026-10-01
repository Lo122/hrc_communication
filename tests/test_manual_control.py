"""The operator's manual control from the live view (2_decision_making/manual_control.py),
through TaskManager as the runtime applies it: a MANUAL_CONTROL event per command."""

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

import config
from cmd_parser import CommandParser
from decision_view import CONTROL_HEADER, DecisionView, build_snapshot
from events import Event, EventType as E, RobotTaskState as S, TaskStatus as T
from gh_dispatcher import GHDispatcher
from message_manager import MessageManager
from pending_task import PendingTaskPool
from state_machine import StateMachine
from task_manager import TaskManager
from task_tracker import build_task_tracking

PULL = config.STEP_NAMES.index("Pull Cables")


class ManualControlTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()
        self.tracker, self.policy = build_task_tracking(ROOT, logger=Mock())
        self.ros, self.udp, self.timer = Mock(), Mock(), Mock()
        self.manager = TaskManager(StateMachine(), PendingTaskPool(), self.timer, MessageManager(), Mock(),
                                   GHDispatcher(self.udp), self.ros, Mock(),
                                   task_tracker=self.tracker, trigger_policy=self.policy)

    def command(self, **payload):
        self.manager.handle_event(Event(E.MANUAL_CONTROL, "decision_view", payload=payload))
        return self.manager.manual_results[0]

    def status(self, name, piece=1):
        task = self.tracker.get(name, piece)
        return task.status, task.executor

    def test_a_task_is_set_exactly_without_closing_the_ones_before_it(self):
        result = self.command(op="task_status", task_name="Screw", piece_id=1, status="DONE")
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.status("Screw"), (T.DONE, "Human"))  # the human, unsaid
        self.assertNotEqual(self.status("Align")[0], T.DONE)  # not inferred
        self.assertEqual(self.tracker.reference_task, "Screw")
        self.command(op="task_status", task_name="Screw", piece_id=1, status="NOT_DONE")
        self.assertEqual(self.status("Screw"), (T.NOT_DONE, None))
        self.assertEqual(self.tracker.get("Screw", 1).progress, 0.0)

    def test_working_by_the_human_makes_it_their_task_and_the_rules_follow(self):
        self.command(op="task_status", task_name="Pull Cables", piece_id=1, status="WORKING",
                     executor="Human", progress=0.8)
        self.assertEqual((self.tracker.reference_task, self.tracker.reference_progress), ("Pull Cables", 0.8))
        # Pull Cables at 0.8 triggers the Lift rule: the robot asks, as after recognition.
        self.assertEqual(self.manager.active_task.task_id, config.TASK_LIFT_PANEL)
        self.command(op="task_status", task_name="Screw", piece_id=1, status="WORKING", executor="Human")
        self.assertEqual(self.status("Pull Cables")[0], T.PENDING)  # one task at a time

    def test_bad_task_commands_are_refused_with_the_reason(self):
        for payload, reason in (
                ({"task_name": "Screw", "piece_id": 1, "status": "DONE", "executor": "Nobody"}, "Unknown executor"),
                ({"task_name": "Screw", "piece_id": 1, "status": "DONE", "executor": "Robot"}, "may not do"),
                ({"task_name": "Nope", "piece_id": 1, "status": "DONE", "executor": "Human"}, "not a task"),
                ({"task_name": "Screw", "piece_id": 1, "status": "SOON"}, "Unknown status"),
                ({"task_name": "Screw", "piece_id": 1, "status": "WORKING", "executor": "Human",
                  "progress": 2}, "0 to 1")):
            result = self.command(op="task_status", **payload)
            self.assertFalse(result["ok"])
            self.assertIn(reason, result["text"])
        self.assertFalse(self.command(op="launch")["ok"])

    def test_signals_are_set_and_cleared(self):
        self.command(op="signal", task_name="Screw", piece_id=1, name="screw count", value=0.5)
        self.assertEqual(self.tracker.get("Screw", 1).signals, {"screw count": 0.5})
        self.command(op="signal", task_name="Screw", piece_id=1, name="screw count", value=None)
        self.assertEqual(self.tracker.get("Screw", 1).signals, {})

    def test_answers_and_robot_reports_go_through_the_usual_handlers(self):
        self.command(op="offer", task_name="Lift", piece_id=1)
        lift = self.manager.active_task
        self.assertEqual((lift.task_id, lift.state), (config.TASK_LIFT_PANEL, S.R_WAITING_RESPONSE))
        result = self.command(op="event", event_type="H_ACCEPT")
        self.assertIn("robot now R_ACCEPTED", result["text"])
        self.assertEqual(self.udp.send.call_count, 1)  # dispatched, as after a spoken yes
        self.command(op="event", event_type="ROBOT_RUNNING")
        self.assertEqual(lift.state, S.R_EXECUTING)
        self.assertFalse(self.command(op="event", event_type="DEMO_START")["ok"])  # not injectable

    def test_the_pending_pool_can_be_added_to_asked_again_and_emptied(self):
        self.command(op="pool_add", task_name="Pull Cables", piece_id=2)
        pooled = self.manager.pending_pool.list_all()
        self.assertEqual([(t.task_id, t.piece_id, t.state) for t in pooled],
                         [(config.TASK_PULL_CABLES, 2, S.R_PENDING)])
        self.assertTrue(self.tracker.get("Pull Cables", 2).robot_offered)
        self.command(op="pool_offer", task_instance_id=pooled[0].task_instance_id)
        self.assertIs(self.manager.active_task, pooled[0])
        self.assertEqual(pooled[0].state, S.R_WAITING_RESPONSE)
        self.manager.handle_event(self.parser.parse("later"))  # back to the pool
        result = self.command(op="pool_remove", task_instance_id=pooled[0].task_instance_id)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.manager.pending_pool.list_all(), [])
        self.assertFalse(self.tracker.get("Pull Cables", 2).robot_offered)  # the human's again

    def test_a_queued_offer_is_dropped_only_if_it_is_still_the_one_meant(self):
        self.command(op="offer", task_name="Lift", piece_id=1)
        self.command(op="offer", task_name="Bring Connector", piece_id=1)
        self.assertEqual([e["task_name"] for e in self.manager.waiting_triggers], ["Bring Connector"])
        self.assertFalse(self.command(op="queue_remove", index=0, task_name="Lift", piece_id=1)["ok"])
        self.assertTrue(self.command(op="queue_remove", index=0, task_name="Bring Connector", piece_id=1)["ok"])
        self.assertEqual(list(self.manager.waiting_triggers), [])

    def test_a_stuck_active_task_and_held_panel_can_be_forgotten(self):
        self.command(op="offer", task_name="Lift", piece_id=1)
        self.command(op="event", event_type="H_ACCEPT")
        result = self.command(op="clear_active")
        self.assertIn("the robot was not told", result["text"])
        self.assertIsNone(self.manager.active_task)
        self.ros.publish_cancel.assert_not_called()
        self.assertFalse(self.command(op="clear_active")["ok"])
        self.assertFalse(self.command(op="clear_held")["ok"])
        self.manager._held_piece_id = 1
        self.assertTrue(self.command(op="clear_held")["ok"])
        self.assertIsNone(self.manager.held_piece_id)

    def test_the_snapshot_shows_what_the_robot_waits_for_and_the_commands(self):
        self.command(op="offer", task_name="Lift", piece_id=1)
        self.command(op="task_status", task_name="Screw", piece_id=1, status="SOON")
        self.ros.latest_wrench = (0.0, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        snapshot = json.loads(json.dumps(build_snapshot(self.manager), default=str))
        self.assertEqual(snapshot["robot"]["ros"]["wrench"]["force_n"], [1.0, 2.0, 3.0])
        active = snapshot["robot"]["active"]
        self.assertEqual(active["waiting_for"], "the human's yes / no / later")
        self.assertEqual(active["accepts"]["H_ACCEPT"], "R_ACCEPTED")
        self.assertEqual(active["accepts"]["H_DEFER"], "R_PENDING")
        self.assertIn("ros", snapshot["robot"])
        self.assertEqual([r["ok"] for r in snapshot["manual"]["results"]], [False, True])  # newest first
        self.assertIn("H_ACCEPT", snapshot["manual"]["events"])
        self.assertIn("Lift", snapshot["manual"]["robot_tasks"])


class StateMachineEventsTests(unittest.TestCase):
    def test_lists_what_each_state_accepts(self):
        machine = StateMachine()
        self.assertEqual(machine.events_from(S.R_WAITING_RESPONSE)["H_REFUSE"], "R_REFUSED")
        self.assertEqual(machine.events_from(S.R_EXECUTING, config.TASK_LIFT_PANEL)["ROBOT_SUCCESS"],
                         machine.get_next_state(S.R_EXECUTING, E.ROBOT_SUCCESS, config.TASK_LIFT_PANEL).name)
        self.assertNotIn("H_ACCEPT", machine.events_from(S.R_EXECUTING))


class CommandEndpointTests(unittest.TestCase):
    def setUp(self):
        self.submitted = []
        self.view = DecisionView("127.0.0.1", 0, submit=self.submitted.append)
        self.addCleanup(self.view.close)

    def post(self, body, headers=None):
        request = urllib.request.Request(self.view.url + "command", data=body, method="POST",
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_a_checked_command_is_handed_on(self):
        command = {"op": "event", "event_type": "H_ACCEPT"}
        status, reply = self.post(json.dumps(command).encode(), {CONTROL_HEADER: "1"})
        self.assertEqual((status, reply), (202, {"queued": "event"}))
        self.assertEqual(self.submitted, [command])

    def test_without_the_header_nothing_changes(self):
        """A page on another site cannot add the header without the browser asking first."""
        status, _ = self.post(b'{"op": "clear_active"}')
        self.assertEqual((status, self.submitted), (403, []))

    def test_unknown_or_broken_commands_are_refused(self):
        for body in (b'{"op": "launch"}', b"not json", b"[1]"):
            status, _ = self.post(body, {CONTROL_HEADER: "1"})
            self.assertEqual(status, 400, body)
        self.assertEqual(self.submitted, [])

    def test_read_only_until_connected(self):
        view = DecisionView("127.0.0.1", 0)
        self.addCleanup(view.close)
        request = urllib.request.Request(view.url + "command", data=b'{"op": "clear_held"}', method="POST",
                                         headers={"Content-Type": "application/json", CONTROL_HEADER: "1"})
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 503)


if __name__ == "__main__":
    unittest.main()
