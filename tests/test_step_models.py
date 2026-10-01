"""Step model formats (1_recognition/src/step_models.py) and RecognitionManager running
either one; offline, with the real model files and a synthetic skeleton instead of a camera."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for path in ("0_core", "1_recognition", "1_recognition/src", "1_recognition/best_model"):
    sys.path.insert(0, str(ROOT / path))

import config
import torch
from recognition_manager import RecognitionManager
from skeleton3d_pipeline import StreamingH36MFeatureExtractor
from step_models import (LEGACY, MULTI_HEAD, MultiHeadStepModel, is_multi_head_training_run,
                         open_step_model)
from vision_model.vision_config import VisionConfig

MODELS = ROOT / "1_recognition" / "best_model"
MULTI_HEAD_DIR = MODELS / "S3_10fps_8s_bg05"
# The same 4-head network in a training run's layout (best_model.pth, norm_stats.npz).
TRAINING_RUN_DIR = MODELS / "3d_skeleton_05"
LEGACY_DIR = MODELS / "3d_skeleton_02"

# A standing H36M pose, pelvis at the origin (metres): pelvis, r_hip, r_knee, r_ankle,
# l_hip, l_knee, l_ankle, spine, thorax, neck, head, l_shoulder, l_elbow, l_wrist,
# r_shoulder, r_elbow, r_wrist.
POSE = np.array([
    [0, 0, 0], [-0.1, 0, 0], [-0.1, 0, -0.45], [-0.1, 0, -0.9], [0.1, 0, 0], [0.1, 0, -0.45],
    [0.1, 0, -0.9], [0, 0, 0.25], [0, 0, 0.5], [0, 0, 0.6], [0, 0, 0.72], [0.18, 0, 0.5],
    [0.2, 0.05, 0.25], [0.22, 0.2, 0.1], [-0.18, 0, 0.5], [-0.2, 0.05, 0.25], [-0.22, 0.2, 0.1],
], dtype=np.float64)


def skeleton_at(t: float) -> np.ndarray:
    """The pose with both forearms swinging, so velocities and angles move."""
    skeleton = POSE.copy()
    swing = 0.15 * math.sin(2 * math.pi * 0.5 * t)
    skeleton[[13, 16], 1] += swing
    skeleton[[12, 15], 2] += 0.5 * swing
    return skeleton


def extracted_features(frames: int = 40, fps: float = 30.0) -> dict:
    extractor = StreamingH36MFeatureExtractor(VisionConfig())
    features = None
    for i in range(frames):
        features = extractor.update(skeleton_at(i / fps), i / fps)
    return features


class FakePipeline:
    """Stands in for RealtimeSkeleton3DPipeline: the synthetic skeleton, no camera."""

    have_extrinsics = False

    def __init__(self, _config):
        pass

    def process(self, frame, timestamp, K=None):
        return {"root_relative": skeleton_at(timestamp), "world_root_xyz": None}


class FormatDetectionTests(unittest.TestCase):
    def test_model_directories_open_in_their_own_format(self):
        multi = open_step_model(MULTI_HEAD_DIR)
        self.assertEqual((multi.format, multi.window_size, multi.sample_period_s), (MULTI_HEAD, 80, 0.1))
        self.assertEqual(multi.step_labels, list(config.STEP_NAMES))
        self.assertEqual(list(multi.class_steps), list(range(7)))  # same order as STEP_NAMES
        self.assertEqual(multi.step_labels[multi.idle_step], "Non Related Task")
        legacy = open_step_model(LEGACY_DIR)
        self.assertEqual((legacy.format, legacy.window_size, legacy.sample_period_s), (LEGACY, 160, None))

    def test_a_training_run_with_an_idle_head_opens_as_multi_head(self):
        run = open_step_model(TRAINING_RUN_DIR)
        self.assertEqual((run.format, run.window_size), (MULTI_HEAD, 120))
        self.assertAlmostEqual(run.sample_period_s, 1 / 30)  # every frame of the 30 fps data
        self.assertEqual(run.classes, list(config.STEP_NAMES[:7]))
        self.assertEqual(run.step_labels[run.idle_step], "Non Related Task")
        self.assertFalse(is_multi_head_training_run(legacy_config()))


def legacy_config() -> dict:
    import json

    return json.loads((LEGACY_DIR / "config.json").read_text(encoding="utf-8"))


class TrainingRunLayoutTests(unittest.TestCase):
    """3d_skeleton_05: the S3 network in another layout, so the same inputs and heads."""

    def setUp(self):
        self.model = open_step_model(TRAINING_RUN_DIR)
        self.model.load(torch, "cpu")

    def test_builds_the_same_raw_input_vector_as_the_export_layout(self):
        export = open_step_model(MULTI_HEAD_DIR)
        features = extracted_features()
        ours = self.model.feature_vector(features) * self.model.std + self.model.mean
        theirs = export.feature_vector(features) * export.std + export.mean
        np.testing.assert_allclose(ours, theirs, atol=1e-4)

    def test_checkpoint_parameters_land_on_the_matching_heads(self):
        checkpoint = torch.load(TRAINING_RUN_DIR / "best_model.pth", map_location="cpu")
        heads = self.model._model.heads
        for name, parameter in (("idle_head", heads.bg), ("step_head", heads.task),
                                ("progress_head", heads.prog), ("mistake_head", heads.mist)):
            torch.testing.assert_close(parameter.weight, checkpoint[f"{name}.weight"])
        torch.testing.assert_close(self.model._model.rnn.weight_hh_l0, checkpoint["lstm.weight_hh_l0"])

    def test_prediction_is_on_the_step_names_like_the_export(self):
        prediction = self.model.predict(
            np.random.default_rng(3).normal(size=(120, 251)).astype(np.float32))
        self.assertEqual(len(prediction.step_scores), len(config.STEP_NAMES))
        self.assertAlmostEqual(float(prediction.step_scores[7]), prediction.idle_probability, places=5)
        self.assertTrue(np.all((prediction.step_progress >= 0) & (prediction.step_progress <= 1)))


class MultiHeadModelTests(unittest.TestCase):
    def setUp(self):
        self.model = open_step_model(MULTI_HEAD_DIR)
        self.model.load(torch, "cpu")

    def test_feature_vector_follows_the_models_column_order_and_transforms(self):
        features = extracted_features()
        vector = self.model.feature_vector(features)
        self.assertEqual(vector.shape, (251,))
        self.assertTrue(np.all(np.isfinite(vector)))
        raw = vector * self.model.std + self.model.mean  # undo the standardisation
        columns = self.model_columns()
        azimuth = np.deg2rad(features["polar_azimuth"])
        np.testing.assert_allclose(raw[columns.index("r_wrist_azimuth_deg_sin")], np.sin(azimuth[-1]), atol=1e-4)
        np.testing.assert_allclose(raw[columns.index("r_wrist_azimuth_deg_cos")], np.cos(azimuth[-1]), atol=1e-4)
        np.testing.assert_allclose(raw[columns.index("l_elbow_speed")], features["joint_speed"][11], atol=1e-4)
        self.assertLessEqual(raw[columns.index("wrist_over_shoulder_ratio")], 4.0 + 1e-4)  # clipped

    def model_columns(self):
        return [str(name) for name in np.load(MULTI_HEAD_DIR / "standardization.npz")["columns"]]

    def test_prediction_maps_the_four_heads_onto_the_step_names(self):
        window = np.random.default_rng(0).normal(size=(80, 251)).astype(np.float32)
        prediction = self.model.predict(window)
        with torch.no_grad():
            out = self.model._model(torch.tensor(window).unsqueeze(0))
        idle = float(torch.sigmoid(out["bg"]))
        tasks = torch.sigmoid(out["task"]).squeeze(0).numpy()
        self.assertAlmostEqual(prediction.idle_probability, idle, places=5)
        np.testing.assert_allclose(prediction.step_scores[:7], tasks * (1 - idle), rtol=1e-5)
        self.assertAlmostEqual(float(prediction.step_scores[7]), idle, places=5)
        # Each task's own sigmoid, unscaled by idle -- what the probability plot draws.
        np.testing.assert_allclose(prediction.task_probabilities[:7], tasks, rtol=1e-5)
        self.assertAlmostEqual(float(prediction.task_probabilities[7]), idle, places=5)
        self.assertEqual(prediction.progress_of(7), 0.0)  # idle has no lane
        self.assertTrue(np.all((prediction.step_progress >= 0) & (prediction.step_progress <= 1)))
        self.assertAlmostEqual(float(prediction.mistake_probabilities.sum()), 1.0, places=5)

    def test_idle_wins_whenever_the_background_head_says_so(self):
        scores = np.zeros(8, dtype=np.float32)
        for idle in (0.51, 0.7, 0.99):
            tasks = np.ones(7, dtype=np.float32)  # even every task certain
            scores[:7], scores[7] = tasks * (1 - idle), idle
            self.assertEqual(int(np.argmax(scores)), 7)


class LegacyModelTests(unittest.TestCase):
    def test_legacy_model_predicts_a_softmax_and_one_progress(self):
        model = open_step_model(LEGACY_DIR)
        model.load(torch, "cpu")
        window = np.random.default_rng(1).normal(size=(160, 89)).astype(np.float32)
        prediction = model.predict(window)
        self.assertAlmostEqual(float(prediction.step_scores.sum()), 1.0, places=4)
        self.assertIsNone(prediction.idle_probability)
        self.assertIsNone(prediction.task_probabilities)  # the softmax already is that
        self.assertIsNone(prediction.step_progress)
        self.assertEqual(prediction.progress_of(3), prediction.progress_of(0))  # one value


class SamplingTests(unittest.TestCase):
    def fed(self, period, loop_hz, seconds=3.0):
        manager = RecognitionManager.__new__(RecognitionManager)
        manager.step_model = SimpleNamespace(sample_period_s=period)
        manager._next_sample_due = None
        return sum(manager._sample_due(i / loop_hz) for i in range(int(seconds * loop_hz)))

    def test_a_model_is_fed_at_its_own_rate_whatever_the_loop_rate(self):
        self.assertEqual(self.fed(0.1, 30), 30)  # every 3rd frame: 10 Hz
        self.assertEqual(self.fed(0.1, 15), 30)  # 2 of every 3 frames, still 10 Hz
        self.assertEqual(self.fed(0.1, 10), 30)  # every frame
        self.assertEqual(self.fed(None, 30), 90)  # legacy: every frame

    def test_a_jittery_loop_at_the_model_rate_feeds_every_frame(self):
        manager = RecognitionManager.__new__(RecognitionManager)
        manager.step_model = SimpleNamespace(sample_period_s=0.1)
        manager._next_sample_due = None
        times = np.cumsum(np.random.default_rng(2).uniform(0.09, 0.11, size=50))
        self.assertTrue(all(manager._sample_due(t) for t in times))


class ManagerTests(unittest.TestCase):
    """RecognitionManager.update_from_frame end to end, but for the camera/pose stage."""

    def run_manager(self, model_dir, frames, fps=30.0):
        manager = RecognitionManager(model_dir=model_dir, device="cpu")
        predictions, results, records = [], [], []
        with patch("skeleton3d_pipeline.RealtimeSkeleton3DPipeline", FakePipeline):
            for i in range(frames):
                result = manager.update_from_frame(np.zeros((8, 8, 3), np.uint8), timestamp=i / fps)
                records.append(dict(manager.last_frame_record))
                stamp = manager.last_step_probabilities_timestamp
                if stamp is not None and (not predictions or predictions[-1] != stamp):
                    predictions.append(stamp)
                if result is not None:
                    results.append((result, manager))
        return manager, predictions, results, records

    def test_multi_head_model_runs_at_10_hz_and_reports_its_extra_heads(self):
        manager, predictions, results, records = self.run_manager(MULTI_HEAD_DIR, frames=300)
        # 100 samples at 30 fps; the first prediction once 80 are buffered.
        self.assertEqual(len(predictions), 100 - 79)
        np.testing.assert_allclose(np.diff(predictions), 0.1, atol=1e-6)
        self.assertEqual(len(manager.last_step_probabilities), len(config.STEP_NAMES))
        self.assertEqual(len(manager.last_step_progress), len(config.STEP_NAMES))
        self.assertIsNotNone(manager.last_idle_probability)
        predicted = [r for r in records if r.get("raw_step_id") is not None]
        self.assertEqual(len(predicted), len(predictions))
        self.assertTrue(all(r["idle_probability"] is not None for r in predicted))
        skipped = records[-2]  # the frame before a sample: not fed, nothing predicted
        self.assertEqual((skipped["warmup"], skipped["raw_step_id"]), (False, None))
        for result, _ in results:
            self.assertIn(result.step_id, range(len(config.STEP_NAMES)))
            self.assertTrue(0.0 <= result.progress <= 1.0)
            # Every step's smoothed score and own lane, for the decision layer's choice.
            self.assertEqual(len(result.step_probabilities), len(config.STEP_NAMES))
            self.assertEqual(len(result.step_progress), len(config.STEP_NAMES))
            self.assertAlmostEqual(result.progress, result.step_progress[result.step_id], places=5)
        if results:
            last, _ = results[-1]
            np.testing.assert_allclose(
                last.step_probabilities, manager.step_stabilizer.smoothed_probabilities, rtol=1e-6)

    def test_the_readout_is_held_on_frames_the_model_skips(self):
        """S3 is fed at 10 Hz; a 30 fps loop skips 2 frames in 3. Those used to be drawn
        with the "Buffering" line instead of the readout -- the flicker."""
        manager = RecognitionManager(model_dir=MULTI_HEAD_DIR, device="cpu")
        shown = []
        with patch("skeleton3d_pipeline.RealtimeSkeleton3DPipeline", FakePipeline), \
                patch.object(manager._view, "show",
                             side_effect=lambda *_a, **readout: shown.append(readout)):
            for i in range(300):
                manager.update_from_frame(np.zeros((8, 8, 3), np.uint8), timestamp=i / 30.0)
        first = next(i for i, readout in enumerate(shown) if readout.get("raw_step_id") is not None)
        after = shown[first:]
        self.assertGreater(first, 200)  # 80 samples at 10 Hz buffered first
        self.assertTrue(all(readout.get("raw_step_id") is not None for readout in after))
        self.assertTrue(all(readout["idle_probability"] is not None for readout in after))
        np.testing.assert_allclose(manager.last_task_probabilities[:7],
                                   manager.last_step_probabilities[:7]
                                   / (1 - manager.last_idle_probability), rtol=1e-4)

    def test_training_run_model_is_fed_every_frame_at_30_fps(self):
        manager, predictions, _results, _records = self.run_manager(TRAINING_RUN_DIR, frames=130)
        self.assertEqual(len(predictions), 130 - 119)
        self.assertIsNotNone(manager.last_idle_probability)
        self.assertEqual(len(manager.last_step_progress), len(config.STEP_NAMES))

    def test_legacy_model_still_runs_every_frame(self):
        manager, predictions, _results, _records = self.run_manager(LEGACY_DIR, frames=170)
        self.assertEqual(len(predictions), 170 - 159)
        self.assertIsNone(manager.last_idle_probability)
        self.assertIsNone(manager.last_step_progress)


class ProbabilityPlotTests(unittest.TestCase):
    def test_plot_draws_per_step_progress_for_a_multi_head_model(self):
        with patch("cv2.namedWindow"), patch("cv2.moveWindow"):
            from step_probability_plot import StepProbabilityPlot

            plot = StepProbabilityPlot(8, step_labels=list(config.STEP_NAMES))
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=-1):
            for i in range(5):
                lanes = np.zeros(8, dtype=np.float32)
                lanes[4] = 0.2 * i
                canvas = plot.update(i * 0.1, np.full(8, 0.1), 0.2 * i, lanes)
        self.assertTrue(plot._per_step_progress())
        self.assertEqual(canvas.shape[:2], (420, 760))


if __name__ == "__main__":
    unittest.main()
