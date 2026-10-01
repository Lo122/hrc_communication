"""Sequence-aware step selection: recognition's scores filtered and weighted by the task
sequence (2_decision_making/src/sequence_step_selector.py, TaskTracker.potential_tasks).
Offline, with the real task database and transition table."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "2_decision_making", "2_decision_making/src"):
    sys.path.insert(0, str(ROOT / layer))

import config
from events import TaskStatus as T
from sequence_step_selector import SequenceStepSelector
from task_tracker import build_task_tracking

IDLE = "Non Related Task"


def scores(default=0.02, **named):
    """Per-step scores over config.STEP_NAMES, by name (spaces as underscores)."""
    values = {name.replace("_", " "): value for name, value in named.items()}
    return [values.get(name, default) for name in config.STEP_NAMES]


class PotentialTasksTests(unittest.TestCase):
    def setUp(self):
        self.tracker, _ = build_task_tracking(ROOT, logger=Mock())

    def names(self):
        return [task.task_name for task in self.tracker.potential_tasks()]

    def test_before_any_task_the_start_row_decides(self):
        start = self.tracker.transitions
        expected = [name for name in ("Pull Cables", "Lift", "Place", "Align", "Screw",
                                      "Connect Cables", "Clamp Coupling")
                    if start.probability(None, name) >= config.PENDING_MIN_PROBABILITY]
        self.assertEqual(self.names(), expected)
        self.assertNotIn(IDLE, self.names())  # on no task list

    def test_the_reference_task_and_what_the_table_expects_after_it(self):
        self.tracker.start_task("Screw", 1)  # e.g. once the robot holds the panel
        self.assertEqual(self.names(), ["Screw", "Connect Cables"])  # earlier ones done
        self.assertEqual(self.tracker.sequence_weight("Screw"), 1.0)
        self.assertEqual(self.tracker.sequence_weight("Connect Cables"),
                         self.tracker.transitions.probability("Screw", "Connect Cables"))

    def test_done_tasks_are_never_potential(self):
        self.tracker.start_task("Screw", 1)
        self.tracker.confirm_done("Screw", piece_id=1)
        self.assertNotIn("Screw", self.names())

    def test_an_overrun_reference_task_opens_the_rest_of_the_piece(self):
        self.tracker.start_task("Screw", 1)
        self.assertNotIn("Clamp Coupling", self.names())  # 0.019 after Screw
        with patch.object(type(self.tracker), "reference_overran", return_value=True):
            self.assertIn("Clamp Coupling", self.names())


class SequenceStepSelectorTests(unittest.TestCase):
    def setUp(self):
        self.tracker, _ = build_task_tracking(ROOT, logger=Mock())
        self.thresholds = {"Pull Cables": 0.5, "Lift": 0.5, "Place": 0.5, "Align": 0.5,
                           "Screw": 0.4, "Connect Cables": 0.25, "Clamp Coupling": 0.25}
        self.selector = SequenceStepSelector(config.STEP_NAMES, self.thresholds,
                                             default_threshold=0.5, strength=0.5, confirm_events=3)
        self.tracker.start_task("Screw", 1)

    def select(self, probabilities, step_progress=None, fallback=0.3, model="Pull Cables"):
        return self.selector.select(probabilities, step_progress, fallback, self.tracker,
                                    model_task=model)

    def test_a_ruled_out_first_option_gives_way_to_an_expected_second(self):
        """The robot-led case: the model's first option is Pull Cables (done on this
        piece), its second the Screw the human is on."""
        choice = self.select(scores(Pull_Cables=0.6, Screw=0.45))
        self.assertEqual((choice.task_name, choice.model_task), ("Screw", "Pull Cables"))
        self.assertTrue(choice.changed)
        self.assertNotIn("Pull Cables", choice.scores)

    def test_candidates_below_their_threshold_are_dropped(self):
        choice = self.select(scores(Pull_Cables=0.6, Screw=0.35))  # Screw's bar: 0.4
        self.assertIsNone(choice.task_name)
        self.assertEqual(choice.reason, "no plausible task scores high enough")

    def test_idle_on_top_with_no_task_high_enough_is_no_task(self):
        choice = self.select(scores(Non_Related_Task=0.9, Screw=0.05), model=IDLE)
        self.assertIsNone(choice.task_name)

    def test_the_sequence_weight_keeps_the_reference_against_a_slightly_higher_switch(self):
        # Screw 0.45 x 1; Connect Cables 0.55 x (0.5 + 0.5 x 0.54) = 0.42: stays.
        choice = self.select(scores(Connect_Cables=0.55, Screw=0.45), model="Connect Cables")
        self.assertEqual(choice.task_name, "Screw")
        weight = 0.5 + 0.5 * self.tracker.transitions.probability("Screw", "Connect Cables")
        self.assertAlmostEqual(choice.scores["Connect Cables"][1], weight)
        self.assertAlmostEqual(choice.scores["Screw"][2], 0.45)

    def test_a_switch_needs_to_win_three_updates_in_a_row(self):
        probabilities = scores(Connect_Cables=0.9, Screw=0.42)
        first, second = self.select(probabilities), self.select(probabilities)
        self.assertEqual((first.task_name, second.task_name), ("Screw", "Screw"))  # held
        self.assertEqual(second.reason, "confirming Connect Cables (2/3)")
        self.assertEqual(self.select(probabilities).task_name, "Connect Cables")

    def test_the_count_starts_over_when_the_reference_wins_again(self):
        switch, stay = scores(Connect_Cables=0.9, Screw=0.42), scores(Screw=0.8)
        self.select(switch), self.select(switch), self.select(stay)
        self.assertEqual(self.select(switch).task_name, "Screw")

    def test_a_switch_away_from_a_reference_that_no_longer_scores_holds_no_task(self):
        probabilities = scores(Connect_Cables=0.9, Screw=0.1)
        self.assertIsNone(self.select(probabilities).task_name)
        self.assertIsNone(self.select(probabilities).task_name)
        self.assertEqual(self.select(probabilities).task_name, "Connect Cables")

    def test_progress_is_the_chosen_steps_own_lane(self):
        lanes = scores(default=0.0, Pull_Cables=0.9, Screw=0.35)
        choice = self.select(scores(Pull_Cables=0.6, Screw=0.45), step_progress=lanes)
        self.assertAlmostEqual(choice.progress, 0.35)
        choice = self.select(scores(Pull_Cables=0.6, Screw=0.45), fallback=0.7)  # one value
        self.assertAlmostEqual(choice.progress, 0.7)

    def test_the_chosen_task_is_marked_working_by_the_tracker_as_usual(self):
        choice = self.select(scores(Pull_Cables=0.6, Screw=0.45))
        self.tracker.on_task_recognized(choice.task_name, choice.progress)
        self.assertEqual(self.tracker.get("Screw", 1).status, T.WORKING)
        self.assertEqual(self.tracker.reference_task, "Screw")


if __name__ == "__main__":
    unittest.main()
