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
from task_database import TaskDatabase
from task_sequence_model import TransitionModel
from task_tracker import TaskTracker

DATABASE = TaskDatabase.from_json(ROOT / config.TASK_DATABASE_PATH)
TRANSITIONS = TransitionModel.from_csv(ROOT / config.TASK_TRANSITION_TABLE_PATH)
RECOGNIZED = config.STEP_NAMES
SUPPORT = ("Bring Tool", "Bring Connector", "Bring back Tool")


def make_tracker(transitions=TRANSITIONS):
    logger = Mock()
    return TaskTracker(DATABASE, transitions, RECOGNIZED, logger=logger,
                       pending_min_probability=config.PENDING_MIN_PROBABILITY), logger


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

    def test_rejects_unknown_task_names(self):
        with self.assertRaises(ValueError):
            TaskDatabase(DATABASE.pieces, {"Lift": ("Robot",)}, {})


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

    def test_unlikely_task_still_works_but_is_logged(self):
        self.tracker.on_task_recognized("Clamp Coupling")
        self.assertEqual(self.status("Clamp Coupling"), T.WORKING)
        self.assertTrue(logged(self.logger, "Unexpected task transition"))

    def test_moving_on_leaves_previous_task_waiting_for_confirmation(self):
        self.tracker.on_task_recognized("Place")
        self.tracker.on_task_recognized("Align")
        place = self.tracker.get("Place", 1)
        self.assertEqual(place.status, T.WORKING)
        self.assertTrue(place.awaiting_confirmation)

    def test_confirmation_closes_earlier_recognized_tasks_only(self):
        self.tracker.on_task_recognized("Align")
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

    def test_support_tasks_need_previous_task_and_progress(self):
        self.tracker.on_task_recognized("Screw", 0.4)
        self.assertEqual(self.names(), [])
        self.tracker.on_task_recognized("Align", 0.9)
        self.assertEqual(self.names(), [])
        self.tracker.on_task_recognized("Screw", 0.5)
        self.assertEqual(self.policy.candidates(), [("Bring Tool", 1)])

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
        self.assertEqual(self.names(), ["Bring Tool"])
        self.policy.mark_offered("Bring Tool", 1)
        self.assertEqual(self.tracker.get("Bring Tool", 1).status, T.PENDING)
        self.assertEqual(self.names(), [])

    def test_done_or_working_tasks_are_not_offered(self):
        self.tracker.confirm_done("Bring Tool", "Human")
        self.tracker.set_robot_status("Bring Connector", 1, T.WORKING)
        self.tracker.on_task_recognized("Screw", 0.9)
        self.assertEqual(self.names(), [])
        self.assertFalse(self.policy.allows("Bring Tool", 1))

    def test_lift_moves_to_next_piece_once_done_here(self):
        self.tracker.on_task_recognized("Pull Cables", 0.6)
        self.assertEqual(self.policy.candidates(), [("Lift", 1)])
        self.tracker.set_robot_status("Lift", 1, T.DONE)
        self.tracker.on_task_recognized("Clamp Coupling", 0.6)
        self.assertEqual(self.policy.candidates(), [("Bring back Tool", 1), ("Lift", 2)])

    def test_pull_cables_rule_is_skipped_without_robot_action(self):
        self.tracker.on_task_recognized("Connect Cables", 0.9)
        self.assertNotIn("Pull Cables", self.names())
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
