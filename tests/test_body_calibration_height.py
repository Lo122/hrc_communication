"""The two stature fields on BodyCalibration and how run_recognition picks between them.

Depth scales LINEARLY with the chosen height, so which field wins -- and
whether a units mistake gets through -- propagates into every absolute world
position the pipeline reports.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))
sys.path.insert(0, str(ROOT / "1_recognition" / "setup"))

from skeleton_utils.body_calibration import (  # noqa: E402
    NOSE_HEIGHT_RATIO, BodyCalibration, validate_stature_m)


def _calibration(**overrides):
    base = dict(bone_lengths={(0, 1): 0.25, (1, 2): 0.31}, n_pose_frames=90,
                subject_id="uid-01")
    base.update(overrides)
    return BodyCalibration(**base)


class EffectiveStatureTest(unittest.TestCase):

    def test_tape_measurement_wins_over_ground_plane(self):
        c = _calibration(stature_m=1.691, measured_stature_m=1.820)
        self.assertEqual(c.effective_stature_m, 1.820)
        self.assertEqual(c.stature_source, "measured (tape)")

    def test_falls_back_to_ground_plane(self):
        c = _calibration(stature_m=1.691)
        self.assertEqual(c.effective_stature_m, 1.691)
        self.assertEqual(c.stature_source, "ground plane")

    def test_none_when_neither_is_present(self):
        """Leaves VisionConfig.user_height_m's assumption in place rather than
        substituting a zero."""
        c = _calibration()
        self.assertIsNone(c.effective_stature_m)
        self.assertEqual(c.stature_source, "none")

    def test_tape_alone_is_enough(self):
        """The whole point of the feature: a height with no extrinsics, so no
        ground plane and no ground-plane stature."""
        c = _calibration(measured_stature_m=1.820)
        self.assertEqual(c.effective_stature_m, 1.820)


class ImpliedNoseRatioTest(unittest.TestCase):

    def test_computed_when_both_measurements_exist(self):
        c = _calibration(measured_stature_m=1.800, nose_height_m=1.650)
        self.assertAlmostEqual(c.implied_nose_height_ratio, 1.650 / 1.800, places=9)

    def test_none_without_a_tape_measurement(self):
        self.assertIsNone(_calibration(nose_height_m=1.650).implied_nose_height_ratio)

    def test_none_without_a_nose_height(self):
        self.assertIsNone(_calibration(measured_stature_m=1.800).implied_nose_height_ratio)

    def test_recovers_the_assumed_ratio_from_a_consistent_pair(self):
        """If the ground-plane stature was right, the implied ratio must come
        back as NOSE_HEIGHT_RATIO -- the check that the arithmetic is the
        inverse of stature_from_ground_plane's."""
        stature = 1.750
        c = _calibration(measured_stature_m=stature,
                         nose_height_m=stature * NOSE_HEIGHT_RATIO)
        self.assertAlmostEqual(c.implied_nose_height_ratio, NOSE_HEIGHT_RATIO, places=9)


class StatureValidationTest(unittest.TestCase):

    def test_accepts_plausible_heights(self):
        for value in (1.50, 1.82, 2.10):
            self.assertEqual(validate_stature_m(value), value)

    def test_rejects_centimetres(self):
        """182 instead of 1.82 -- a 100x error that would otherwise silently
        scale every world position."""
        with self.assertRaises(ValueError) as caught:
            validate_stature_m(182.0)
        self.assertIn("centimetres", str(caught.exception))

    def test_rejects_zero_and_negatives(self):
        for value in (0.0, -1.8):
            with self.assertRaises(ValueError):
                validate_stature_m(value)


class RoundTripTest(unittest.TestCase):

    def test_measured_height_survives_save_and_load(self):
        c = _calibration(stature_m=1.691, measured_stature_m=1.820, nose_height_m=1.573)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body_uid-01.json"
            c.save(path)
            loaded = BodyCalibration.load(path)

        self.assertEqual(loaded.measured_stature_m, 1.820)
        self.assertEqual(loaded.stature_m, 1.691)
        self.assertEqual(loaded.effective_stature_m, 1.820)
        self.assertEqual(loaded.bone_lengths, c.bone_lengths)

    def test_reads_files_written_before_the_field_existed(self):
        """Older calibrations have no measured_stature_m key at all; they must
        still load and fall back to the ground-plane stature."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps({
                "bone_lengths": {"0-1": 0.25},
                "stature_m": 1.691,
                "nose_height_m": 1.573,
                "n_pose_frames": 90,
            }), encoding="utf-8")
            loaded = BodyCalibration.load(path)

        self.assertIsNone(loaded.measured_stature_m)
        self.assertEqual(loaded.effective_stature_m, 1.691)

    def test_a_hand_edited_units_mistake_is_caught_at_load(self):
        """Hand-editing the JSON is an expected workflow, so the check has to
        live at load, not only at the CLI."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "typo.json"
            path.write_text(json.dumps({"bone_lengths": {},
                                        "measured_stature_m": 182}), encoding="utf-8")
            with self.assertRaises(ValueError):
                BodyCalibration.load(path)


class UpdateCommandTest(unittest.TestCase):

    def test_adds_a_height_to_an_existing_calibration(self):
        from calibrate_body import update_measured_height

        c = _calibration(stature_m=1.691, nose_height_m=1.573)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body_uid-01.json"
            c.save(path)
            update_measured_height(path, 1.820)
            reloaded = BodyCalibration.load(path)

        self.assertEqual(reloaded.measured_stature_m, 1.820)
        # Everything else must survive -- the point is not redoing the T-pose.
        self.assertEqual(reloaded.bone_lengths, c.bone_lengths)
        self.assertEqual(reloaded.stature_m, 1.691)
        self.assertEqual(reloaded.n_pose_frames, 90)

    def test_rejects_a_units_mistake(self):
        from calibrate_body import update_measured_height

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.json"
            _calibration().save(path)
            with self.assertRaises(ValueError):
                update_measured_height(path, 182.0)


if __name__ == "__main__":
    unittest.main()
