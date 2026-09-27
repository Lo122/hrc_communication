"""Task tracker, robot trigger policy, transition table and recognition filter."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making"):
    sys.path.insert(0, str(ROOT / layer))
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))
sys.path.insert(0, str(ROOT / "2_decision_making" / "src"))

import config
from events import TaskStatus as T
from robot_trigger_policy import RobotTriggerPolicy
from step_stabilizer import StepIdStabilizer
from task_database import TaskDatabase, TriggerRule, _parse_piece_ids
from task_sequence_model import TransitionModel, load_duration_stat
from task_tracker import TaskTracker

DATABASE = TaskDatabase.from_json(ROOT / config.TASK_DATABASE_PATH)
TRANSITIONS = TransitionModel.from_csv(ROOT / config.TASK_TRANSITION_TABLE_PATH)
RECOGNIZED = config.STEP_NAMES
SUPPORT = ("Bring Tool", "Bring Connector", "Bring back Tool")


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_tracker(transitions=TRANSITIONS, duration_limits=None, clock=None):
    logger = Mock()
    return TaskTracker(DATABASE, transitions, RECOGNIZED, logger=logger,
                       pending_min_probability=config.PENDING_MIN_PROBABILITY,
                       duration_limits=duration_limits, clock=clock or FakeClock()), logger


def logged(logger, text):
    return any(text in call.args[0] for call in logger.log_message.call_args_list)


class TaskDatabaseTests(unittest.TestCase):
    def test_loads_pieces_executors_and_trigger_rules(self):
        self.assertEqual([piece.piece_id for piece in DATABASE.pieces], [1, 2, 3])
        self.assertEqual(len(DATABASE.pieces[0].task_list), 10)
        self.assertTrue(DATABASE.can_execute("Lift", "Robot"))
        self.assertFalse(DATABASE.can_execute("Screw", "Robot"))
        rule = DATABASE.trigger_rules["Bring Tool"]
        self.assertEqual((rule.previous_tasks, rule.progress, rule.done_signal),
                         (("Screw", "Connect Cables"), 0.5, False))
        leave = DATABASE.trigger_rules["Leave from the panel"]
        self.assertEqual((leave.previous_tasks, leave.done_signal), (("Screw",), True))
        # A robot action outside the pieces' assembly: never blocks a piece's completion.
        self.assertFalse(DATABASE.in_task_lists("Leave from the panel"))

    def test_loads_piece_ids_chains_and_conditions(self):
        lift = DATABASE.trigger_rules["Lift"]
        self.assertEqual([lift.piece_offset(name) for name in lift.previous_tasks], [0, 1, 1])
        self.assertEqual(DATABASE.trigger_rules["Pull Cables"].robot_tasks, ("Pull Cables", "Lift"))
        self.assertEqual(DATABASE.next_robot_task("Pull Cables"), "Lift")
        self.assertIsNone(DATABASE.next_robot_task("Lift"))
        self.assertEqual(DATABASE.trigger_rules["Bring Connector"].conditions["Screw"],
                         ({"Done signal": True}, {"screw count": 0.5}, {"TCP weight change": True}))

    def test_rejects_unknown_task_names(self):
        with self.assertRaises(ValueError):
            TaskDatabase(DATABASE.pieces, {"Lift": ("Robot",)}, {})

    def test_rejects_bad_piece_ids_and_chains(self):
        with self.assertRaisesRegex(ValueError, "Piece id"):
            _parse_piece_ids("m + 1", ("Screw",), "Bring Tool")
        rule = TriggerRule("Pull Cables", ("Clamp Coupling",), 0.5, robot_tasks=("Lift", "Pull Cables"))
        with self.assertRaisesRegex(ValueError, "must start with"):
            TaskDatabase(DATABASE.pieces, DATABASE.executors, {"Pull Cables": rule})


class TaskTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker, self.logger = make_tracker()

    def status(self, name, piece=1):
        return self.tracker.get(name, piece).status

    def test_start_row_decides_first_pending_tasks(self):
        pending = {task.task_name for task, _ in self.tracker.pending()}
        likely = {name for name in RECOGNIZED
                  if TRANSITIONS.probability(None, name) >= config.PENDING_MIN_PROBABILITY}
        self.assertEqual(pending, likely)
        self.assertIn("Pull Cables", pending)
        for name in SUPPORT:
            self.assertEqual(self.status(name), T.NOT_DONE)

    def test_recognized_task_works_and_pending_follows_its_row(self):
        self.tracker.on_task_recognized("Pull Cables", 0.3)
        self.assertEqual((self.status("Pull Cables"), self.tracker.get("Pull Cables", 1).executor),
                         (T.WORKING, "Human"))
        for name in RECOGNIZED:
            if self.status(name) in (T.WORKING, T.DONE):
                continue
            expected = (T.PENDING if TRANSITIONS.probability("Pull Cables", name) >= config.PENDING_MIN_PROBABILITY
                        else T.NOT_DONE)
            self.assertEqual(self.status(name), expected, name)
        ranked = [probability for _, probability in self.tracker.pending()]
        self.assertEqual(ranked, sorted(ranked, reverse=True))

    def human_working(self):
        return [name for tasks in self.tracker.snapshot()["pieces"].values() for name, task in tasks.items()
                if task["status"] == "WORKING" and task["executor"] == "Human"]

    def test_unexpected_task_is_ignored(self):
        self.tracker.on_task_recognized("Pull Cables", 0.3)
        self.assertIsNone(self.tracker.on_task_recognized("Align", 0.5))  # P(Align | Pull Cables) = 0.02
        self.assertEqual(self.status("Align"), T.NOT_DONE)
        self.assertEqual((self.tracker.reference_task, self.tracker.reference_progress), ("Pull Cables", 0.3))
        self.assertEqual(self.tracker.ignored_task, "Align")
        self.tracker.on_task_recognized("Align", 0.6)
        ignored = [call for call in self.logger.log_message.call_args_list
                   if call.args[0].startswith("Recognized task ignored")]
        self.assertEqual(len(ignored), 1)  # logged once, not per update

    def test_moving_on_pulls_the_previous_task_back(self):
        self.tracker.on_task_recognized("Pull Cables", 0.8)
        self.tracker.on_task_recognized("Place", 0.2)
        self.assertEqual((self.status("Place"), self.tracker.get("Place", 1).executor), (T.WORKING, "Human"))
        # Back to the pool: the table hardly expects Pull Cables after Place.
        self.assertEqual((self.status("Pull Cables"), self.tracker.get("Pull Cables", 1).executor),
                         (T.NOT_DONE, None))
        self.tracker.on_task_recognized("Screw", 0.1)  # P(Screw | Place) = 0.03: ignored
        self.assertEqual(self.human_working(), ["Place"])

    def test_human_works_on_one_task_at_a_time(self):
        for name in ("Pull Cables", "Lift", "Place", "Align", "Screw", "Pull Cables", "Screw", "Connect Cables"):
            self.assertIsNotNone(self.tracker.on_task_recognized(name, 0.2), name)
            expected = [] if self.status(name) == T.DONE else [name]
            self.assertEqual(self.human_working(), expected, name)
        # Screw, left for Connect Cables, is back in the pool: not expected after it.
        self.assertEqual(self.status("Screw"), T.NOT_DONE)

    def test_workflow_start_pulls_back_the_recognized_task(self):
        self.tracker.on_task_recognized("Pull Cables", 0.3)
        self.tracker.on_task_recognized("Screw", 0.3)
        self.tracker.start_task("Align", 1)
        self.assertEqual(self.human_working(), ["Align"])

    def test_overrun_reference_lets_recognition_through(self):
        clock = FakeClock()
        tracker, logger = make_tracker(duration_limits={"Pull Cables": 6.0}, clock=clock)
        tracker.on_task_recognized("Pull Cables", 0.3)
        clock.now += 5.0
        self.assertIsNone(tracker.on_task_recognized("Align", 0.5))
        self.assertFalse(tracker.reference_overran())
        clock.now += 2.0
        self.assertTrue(tracker.reference_overran())
        self.assertEqual(tracker.on_task_recognized("Align", 0.5).status, T.WORKING)
        self.assertEqual(tracker.get("Pull Cables", 1).status, T.PENDING)  # P(Pull Cables | Align) = 0.1
        self.assertTrue(logged(logger, "overran its usual duration"))
        self.assertEqual(tracker.reference_seconds(), 0.0)  # a new reference starts its own clock
        self.assertFalse(tracker.reference_overran())  # Align has no limit here

    def test_duration_limits_cover_every_recognized_task(self):
        limits = load_duration_stat(ROOT / config.TASK_DURATION_STATS_PATH, config.TASK_OVERRUN_STAT)
        recognized = {name for name in DATABASE.pieces[0].task_list if name in RECOGNIZED}
        self.assertEqual(recognized - set(limits), set())
        with self.assertRaisesRegex(ValueError, "no column"):
            load_duration_stat(ROOT / config.TASK_DURATION_STATS_PATH, "p99")

    def test_done_reference_overruns_from_when_it_was_done(self):
        clock = FakeClock()
        tracker, _ = make_tracker(duration_limits={"Screw": 90.0}, clock=clock)
        tracker.on_task_recognized("Screw", 0.3)
        clock.now += 80.0
        tracker.confirm_done("Screw")
        clock.now += 80.0
        self.assertIsNone(tracker.on_task_recognized("Clamp Coupling", 0.5))  # P = 0.02 after Screw
        clock.now += 11.0
        self.assertEqual(tracker.on_task_recognized("Clamp Coupling", 0.5).status, T.WORKING)

    def test_confirmation_closes_earlier_recognized_tasks_only(self):
        self.tracker.on_task_recognized("Pull Cables")
        self.tracker.confirm_done("Screw")
        for name in ("Pull Cables", "Lift", "Place", "Align", "Screw"):
            self.assertEqual(self.status(name), T.DONE, name)
        self.assertTrue(self.tracker.get("Place", 1).inferred)
        self.assertFalse(self.tracker.get("Screw", 1).inferred)
        self.assertEqual(self.status("Connect Cables"), T.PENDING)
        for name in SUPPORT:
            self.assertNotEqual(self.status(name), T.DONE)

    def test_confirm_without_name_uses_working_task(self):
        self.assertIsNone(self.tracker.confirm_done())
        self.tracker.on_task_recognized("Screw", 0.7)
        self.assertEqual(self.tracker.confirm_done().task_name, "Screw")

    def test_robot_states_mirror_and_never_undo_done(self):
        self.tracker.set_robot_status("Bring Tool", 1, T.PENDING)
        self.assertEqual(self.status("Bring Tool"), T.PENDING)
        self.tracker.set_robot_status("Bring Tool", 1, T.WORKING)
        self.assertEqual((self.status("Bring Tool"), self.tracker.get("Bring Tool", 1).executor), (T.WORKING, "Robot"))
        self.tracker.set_robot_status("Bring Tool", 1, T.DONE)
        self.tracker.set_robot_status("Bring Tool", 1, T.PENDING)
        self.assertEqual((self.status("Bring Tool"), self.tracker.get("Bring Tool", 1).executor), (T.DONE, "Robot"))

    def test_robot_completion_of_human_only_task_is_rejected(self):
        self.tracker.set_robot_status("Screw", 1, T.DONE)
        self.assertNotEqual(self.status("Screw"), T.DONE)
        self.assertTrue(logged(self.logger, "Rejected completion"))

    def test_repeated_step_stays_on_current_piece(self):
        self.tracker.confirm_done("Pull Cables")
        self.tracker.on_task_recognized("Screw")
        self.tracker.on_task_recognized("Pull Cables")  # second cable pull of piece 1
        self.assertEqual(self.tracker.reference_piece_id, 1)
        self.assertEqual(self.status("Pull Cables", 2), T.NOT_DONE)

    def test_piece_advances_when_every_task_is_done(self):
        for name in DATABASE.pieces[0].task_list:
            self.tracker.confirm_done(name, "Robot" if name in SUPPORT else "Human", piece_id=1)
        self.assertEqual(self.tracker.current_piece_id, 2)
        self.tracker.on_task_recognized("Pull Cables")
        self.assertEqual(self.status("Pull Cables", 2), T.WORKING)

    def test_human_piece_moves_on_before_support_tasks_finish(self):
        self.tracker.confirm_done("Clamp Coupling")
        self.assertEqual((self.tracker.current_piece_id, self.tracker.human_piece_id), (1, 2))

    def test_workflow_start_moves_the_human_to_that_piece(self):
        self.tracker.confirm_done("Connect Cables")
        self.tracker.on_task_recognized("Clamp Coupling", 0.6)  # still finishing piece 1
        self.tracker.set_robot_status("Lift", 2, T.WORKING)  # the robot works ahead
        self.tracker.start_task("Align", 2)  # the human aligns piece 2 in free drive
        self.assertEqual((self.status("Clamp Coupling"), self.tracker.get("Clamp Coupling", 1).inferred),
                         (T.DONE, True))
        self.assertNotEqual(self.status("Bring back Tool"), T.DONE)  # support tasks stay open
        self.assertEqual((self.status("Align", 2), self.tracker.get("Align", 2).executor), (T.WORKING, "Human"))
        self.assertEqual(self.status("Pull Cables", 2), T.DONE)
        self.assertEqual((self.tracker.reference_task, self.tracker.reference_piece_id), ("Align", 2))
        self.assertEqual(self.tracker.human_piece_id, 2)
        self.tracker.on_task_recognized("Screw", 0.3)
        self.assertEqual(self.status("Screw", 2), T.WORKING)

    def test_manual_next_piece(self):
        self.assertEqual(self.tracker.advance_piece(), 2)
        self.assertEqual(self.tracker.current_piece_id, 2)

    def test_falls_back_to_database_order_when_table_expects_nothing(self):
        tracker, _ = make_tracker(TransitionModel({}))
        self.assertEqual([task.task_name for task, _ in tracker.pending()], ["Pull Cables"])

    def test_status_line(self):
        self.tracker.on_task_recognized("Screw", 0.62)
        line = self.tracker.status_line()
        self.assertIn("piece 1", line)
        self.assertIn("working: Screw (0.62)", line)
        self.assertIn("Connect Cables", line)


class RobotTriggerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tracker, self.logger = make_tracker()
        self.policy = RobotTriggerPolicy(DATABASE, self.tracker, config.TRACKED_TO_ROBOT_TASK, logger=self.logger)

    def names(self):
        return [name for name, _ in self.policy.candidates()]

    def test_support_tasks_need_previous_task_progress_and_condition(self):
        self.tracker.on_task_recognized("Screw", 0.4)
        self.assertEqual(self.names(), [])
        self.tracker.on_task_recognized("Align", 0.9)
        self.assertEqual(self.names(), [])
        self.tracker.on_task_recognized("Screw", 0.5)
        self.assertEqual(self.names(), [])  # Bring Tool's "Condition": Screw done
        self.tracker.confirm_done("Screw")
        self.assertEqual([item for item in self.policy.candidates() if item[0] != "Leave from the panel"],
                         [("Bring Tool", 1), ("Bring Connector", 1)])

    def test_condition_alternatives_accept_detector_signals(self):
        self.tracker.on_task_recognized("Screw", 0.6)
        self.assertNotIn("Bring Connector", self.names())
        self.tracker.set_signal("Screw", 1, "screw count", 0.25)
        self.assertNotIn("Bring Connector", self.names())
        self.tracker.set_signal("Screw", 1, "screw count", 0.5)
        self.assertIn(("Bring Connector", 1), self.policy.candidates())
        self.assertNotIn("Bring Tool", self.names())  # its condition wants Screw done itself
        other, _ = make_tracker()
        other.on_task_recognized("Screw", 0.6)
        other.set_signal("Screw", 1, "TCP weight change", True)
        policy = RobotTriggerPolicy(DATABASE, other, config.TRACKED_TO_ROBOT_TASK)
        self.assertIn(("Bring Connector", 1), policy.candidates())

    def test_confirmed_previous_task_counts_as_full_progress(self):
        self.tracker.on_task_recognized("Screw", 0.1)
        self.tracker.confirm_done("Screw")
        self.assertIn("Bring Tool", self.names())

    def test_done_signal_rule_needs_confirmation_not_recognized_progress(self):
        self.tracker.on_task_recognized("Screw", 1.0)
        self.assertNotIn("Leave from the panel", self.names())
        # Recognition has moved on by the time "screw done" arrives: still counts.
        self.tracker.on_task_recognized("Connect Cables", 0.2)
        self.tracker.confirm_done("Screw")
        self.assertIn(("Leave from the panel", 1), self.policy.candidates())
        self.policy.mark_offered("Leave from the panel", 1)
        self.assertNotIn("Leave from the panel", self.names())

    def test_each_task_is_offered_once_per_piece(self):
        self.tracker.on_task_recognized("Screw", 0.9)
        self.tracker.confirm_done("Screw")
        self.assertIn("Bring Tool", self.names())
        self.policy.mark_offered("Bring Tool", 1)
        self.assertEqual(self.tracker.get("Bring Tool", 1).status, T.PENDING)
        self.assertNotIn("Bring Tool", self.names())

    def test_done_or_working_tasks_are_not_offered(self):
        self.tracker.confirm_done("Bring Tool", "Human")
        self.tracker.set_robot_status("Bring Connector", 1, T.WORKING)
        self.tracker.on_task_recognized("Screw", 0.9)
        self.assertEqual(self.names(), [])
        self.assertFalse(self.policy.allows("Bring Tool", 1))

    def test_piece_id_sends_lift_and_pull_cables_to_the_right_piece(self):
        self.tracker.on_task_recognized("Pull Cables", 0.6)
        self.assertEqual(self.policy.candidates(), [("Lift", 1)])  # "Pull Cables": "n"
        self.tracker.set_robot_status("Lift", 1, T.DONE)
        for name in ("Place", "Align", "Screw", "Connect Cables"):
            self.tracker.on_task_recognized(name, 0.3)
        self.tracker.on_task_recognized("Clamp Coupling", 0.6)
        # "n + 1"; Bring back Tool also waits for Screw done, which never came.
        self.assertEqual(self.policy.candidates(), [("Pull Cables", 2), ("Lift", 2)])

    def test_no_piece_after_the_last_one(self):
        self.tracker.advance_piece()
        self.tracker.advance_piece()
        self.tracker.on_task_recognized("Screw", 0.9)
        self.tracker.on_task_recognized("Connect Cables", 0.9)
        self.assertEqual(self.tracker.reference_piece_id, 3)
        self.assertEqual(self.policy.candidates(), [])

    def test_rule_is_skipped_without_robot_action(self):
        robot_tasks = {name: task_id for name, task_id in config.TRACKED_TO_ROBOT_TASK.items()
                       if name != "Pull Cables"}
        policy = RobotTriggerPolicy(DATABASE, self.tracker, robot_tasks, logger=self.logger)
        self.tracker.on_task_recognized("Screw", 0.9)
        self.tracker.on_task_recognized("Connect Cables", 0.9)
        self.assertEqual(policy.candidates(), [("Lift", 2)])
        self.assertTrue(logged(self.logger, "no robot action configured"))


class TransitionFilterTests(unittest.TestCase):
    def test_allowed_transitions_follow_table(self):
        allowed = TRANSITIONS.allowed_transitions(RECOGNIZED, 0.02)
        place, align = RECOGNIZED.index("Place"), RECOGNIZED.index("Align")
        self.assertIn(place, allowed[place])
        self.assertIn(align, allowed[place])
        for index, targets in allowed.items():
            self.assertIn(index, targets)
            for target in targets:
                if target != index:
                    self.assertGreaterEqual(TRANSITIONS.probability(RECOGNIZED[index], RECOGNIZED[target]), 0.02)

    def test_unobserved_step_is_not_restricted(self):
        model = TransitionModel({"START": {"A": 1.0}, "A": {"B": 1.0}})
        self.assertEqual(model.allowed_transitions(["A", "B"], 0.5), {0: [0, 1], 1: [0, 1]})

    def test_stabilizer_blocks_unlikely_change_until_override(self):
        stabilizer = StepIdStabilizer(num_steps=3, smoothing_window=1, confirmation_count=2,
                                      min_confidence=0.5, min_margin=0.1,
                                      allowed_transitions={0: [0, 1], 1: [1], 2: [2]}, override_factor=3)
        one_hot = lambda step: np.eye(3)[step]
        self.assertEqual(stabilizer.update(one_hot(0)), 0)
        for _ in range(5):
            self.assertEqual(stabilizer.update(one_hot(2)), 0)  # 0 -> 2 is not allowed
        self.assertEqual(stabilizer.update(one_hot(2)), 2)  # held 2 * 3 frames: accepted
        self.assertEqual(stabilizer.override_count, 1)

    def test_stabilizer_without_override_never_takes_disallowed_change(self):
        stabilizer = StepIdStabilizer(num_steps=2, smoothing_window=1, confirmation_count=1,
                                      min_confidence=0.5, min_margin=0.1, allowed_transitions={0: [0], 1: [1]})
        stabilizer.update(np.eye(2)[0])
        for _ in range(20):
            self.assertEqual(stabilizer.update(np.eye(2)[1]), 0)


if __name__ == "__main__":
    unittest.main()
