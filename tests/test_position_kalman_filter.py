"""PositionKalmanFilter: smoothing, velocity from real timestamps, gating, resets.

The velocity is what path planning consumes, so the key property is that it
follows the ACTUAL spacing of the timestamps -- irregular live frames must give
the same m/s as evenly spaced ones.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))

from skeleton_utils.position_kalman_filter import PositionKalmanFilter  # noqa: E402


def _walk(filt, times, velocity, start=(0.0, 0.0, 1.0), noise_std=0.0, seed=0):
    rng = np.random.default_rng(seed)
    start, velocity = np.asarray(start), np.asarray(velocity)
    out = []
    for t in times:
        z = start + velocity * t + rng.normal(0.0, noise_std, 3)
        out.append(filt.update(z, t))
    return np.array(out)


class VelocityFromRealTimeTest(unittest.TestCase):

    def test_irregular_timestamps_give_true_velocity(self):
        # Live frame spacing jitters between 30 ms and 120 ms.
        rng = np.random.default_rng(1)
        times = np.cumsum(rng.uniform(0.03, 0.12, 200))
        filt = PositionKalmanFilter()
        _walk(filt, times, velocity=(1.2, -0.4, 0.0))
        np.testing.assert_allclose(filt.velocity, [1.2, -0.4, 0.0], atol=0.02)

    def test_same_motion_at_different_rates_gives_same_velocity(self):
        fast, slow = PositionKalmanFilter(), PositionKalmanFilter()
        _walk(fast, np.arange(1, 150) / 30.0, velocity=(0.8, 0.0, 0.0))
        _walk(slow, np.arange(1, 50) / 10.0, velocity=(0.8, 0.0, 0.0))
        np.testing.assert_allclose(fast.velocity, slow.velocity, atol=0.02)
        self.assertAlmostEqual(fast.velocity[0], 0.8, delta=0.02)


class SmoothingTest(unittest.TestCase):

    def test_reduces_noise_on_a_standing_person(self):
        times = np.arange(300) / 20.0
        filt = PositionKalmanFilter(measurement_std_m=0.08)
        filtered = _walk(filt, times, velocity=(0, 0, 0), start=(2.0, 1.0, 1.0),
                         noise_std=0.08, seed=3)
        raw = [2.0, 1.0, 1.0] + np.random.default_rng(3).normal(0.0, 0.08, (len(times), 3))
        truth = np.array([2.0, 1.0, 1.0])
        raw_err = np.linalg.norm(raw[50:] - truth, axis=1).mean()
        err = np.linalg.norm(filtered[50:] - truth, axis=1).mean()
        self.assertLess(err, 0.6 * raw_err)

    def test_nan_measurement_is_ignored(self):
        filt = PositionKalmanFilter()
        filt.update([1.0, 2.0, 1.0], 0.0)
        out = filt.update([np.nan, 0.0, 0.0], 0.1)
        np.testing.assert_allclose(out, [1.0, 2.0, 1.0])

    def test_nan_before_first_measurement_returns_none(self):
        self.assertIsNone(PositionKalmanFilter().update([np.nan] * 3, 0.0))


class GatingAndResetTest(unittest.TestCase):

    def test_single_outlier_is_rejected(self):
        filt = PositionKalmanFilter()
        _walk(filt, np.arange(40) / 20.0, velocity=(0, 0, 0))
        out = filt.update([3.0, 0.0, 1.0], 2.0)  # 3 m teleport for one frame
        self.assertTrue(filt.last_rejected)
        self.assertLess(np.linalg.norm(out - [0.0, 0.0, 1.0]), 0.05)

    def test_persistent_jump_reinitialises(self):
        filt = PositionKalmanFilter(max_consecutive_rejects=3)
        _walk(filt, np.arange(40) / 20.0, velocity=(0, 0, 0))
        for i in range(3):
            out = filt.update([3.0, 0.0, 1.0], 2.0 + 0.05 * (i + 1))
        np.testing.assert_allclose(out, [3.0, 0.0, 1.0])
        np.testing.assert_allclose(filt.velocity, [0.0, 0.0, 0.0])

    def test_long_gap_reinitialises(self):
        filt = PositionKalmanFilter(max_gap_s=1.0)
        _walk(filt, np.arange(40) / 20.0, velocity=(1.0, 0, 0))
        out = filt.update([10.0, 5.0, 1.0], 10.0)
        np.testing.assert_allclose(out, [10.0, 5.0, 1.0])
        np.testing.assert_allclose(filt.velocity, [0.0, 0.0, 0.0])

    def test_duplicate_timestamp_does_not_blow_up(self):
        filt = PositionKalmanFilter()
        filt.update([0.0, 0.0, 1.0], 1.0)
        out = filt.update([0.01, 0.0, 1.0], 1.0)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(np.isfinite(filt.velocity)))

    def test_reset_forgets_state(self):
        filt = PositionKalmanFilter()
        filt.update([1.0, 1.0, 1.0], 0.0)
        filt.reset()
        self.assertFalse(filt.initialized)
        self.assertIsNone(filt.velocity)


if __name__ == "__main__":
    unittest.main()
