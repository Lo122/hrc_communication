"""Eye-to-hand calibration math, against synthetic data -- no robot, no camera.

The composition itself cannot be validated on real data (see the module
docstring of robot_camera_calibration.py: T_base_from_camera is *defined* by
the composition, so its residual is zero by construction). Synthetic data is
the only place the recovery can be checked against a known truth.
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))
sys.path.insert(0, str(ROOT / "1_recognition" / "setup"))
sys.path.insert(0, str(ROOT / "1_recognition" / "setup" / "calibration"))

from calibration import robot_camera_calibration as rcc  # noqa: E402
from calibration.extrinsic_calibration import solve_marker_pose  # noqa: E402
from camera_utils import transforms as tf  # noqa: E402

MARKER_LENGTH_M = 0.200


def _transform(roll, pitch, yaw, xyz):
    return tf.make_transform(tf.rpy_deg_to_matrix(roll, pitch, yaw), xyz)


class RigidTransformFromPointsTest(unittest.TestCase):

    def test_recovers_a_known_transform_exactly(self):
        T_true = _transform(12.0, -30.0, 75.0, [0.41, -0.23, 0.09])
        source = rcc.marker_corner_points(MARKER_LENGTH_M)
        target = tf.transform_points(T_true, source)

        T_fit, rms = tf.rigid_transform_from_points(source, target)
        np.testing.assert_allclose(T_fit, T_true, atol=1e-9)
        self.assertAlmostEqual(rms, 0.0, places=12)

    def test_noise_shows_up_in_the_residual(self):
        rng = np.random.default_rng(0)
        T_true = _transform(5.0, 10.0, -40.0, [0.3, 0.2, 0.1])
        source = rcc.marker_corner_points(MARKER_LENGTH_M)
        target = tf.transform_points(T_true, source) + rng.normal(0.0, 0.001, (4, 3))

        T_fit, rms = tf.rigid_transform_from_points(source, target)
        # 1 mm of per-point noise -> a residual of the same order, not zero and
        # not wildly amplified.
        self.assertGreater(rms, 0.0002)
        self.assertLess(rms, 0.003)

        relative = T_fit[:3, :3].T @ T_true[:3, :3]
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertLess(angle_deg, 1.5)

    def test_returns_a_proper_rotation_not_a_reflection(self):
        """Mirrored targets must not be 'fitted' by a reflection: det(R) == -1
        would still give a small residual while flipping the frame."""
        source = rcc.marker_corner_points(MARKER_LENGTH_M)
        target = source * np.array([1.0, 1.0, -1.0])  # mirror through the marker plane

        T_fit, _rms = tf.rigid_transform_from_points(source, target)
        self.assertAlmostEqual(float(np.linalg.det(T_fit[:3, :3])), 1.0, places=9)

    def test_rejects_mismatched_and_too_few_points(self):
        points = rcc.marker_corner_points(MARKER_LENGTH_M)
        with self.assertRaises(ValueError):
            tf.rigid_transform_from_points(points, points[:3])
        with self.assertRaises(ValueError):
            tf.rigid_transform_from_points(points[:2], points[:2])


class MarkerCornerOrderTest(unittest.TestCase):

    def test_matches_the_object_points_solve_marker_pose_uses(self):
        """The optical and touch sides must agree on which corner is which.
        solve_marker_pose builds its own object points inline, so this is the
        regression guard against the two drifting apart.
        """
        ours = rcc.marker_corner_points(MARKER_LENGTH_M)
        half = MARKER_LENGTH_M / 2.0
        theirs = np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ])
        np.testing.assert_allclose(ours, theirs, atol=1e-12)

    def test_is_the_expected_physical_square(self):
        corners = rcc.marker_corner_points(MARKER_LENGTH_M)
        self.assertAlmostEqual(np.linalg.norm(corners[0] - corners[1]), 0.200, places=9)
        self.assertAlmostEqual(np.linalg.norm(corners[0] - corners[2]),
                               0.200 * np.sqrt(2), places=9)
        np.testing.assert_allclose(corners.mean(axis=0), np.zeros(3), atol=1e-12)


class CheckTouchGeometryTest(unittest.TestCase):

    def test_clean_touches_report_nothing(self):
        T = _transform(3.0, -7.0, 20.0, [0.4, 0.0, 0.05])
        marker = rcc.marker_corner_points(MARKER_LENGTH_M)
        touched = tf.transform_points(T, marker)
        self.assertEqual(rcc.check_touch_geometry(touched, marker), [])

    def test_a_displaced_corner_is_named(self):
        T = _transform(3.0, -7.0, 20.0, [0.4, 0.0, 0.05])
        marker = rcc.marker_corner_points(MARKER_LENGTH_M)
        touched = tf.transform_points(T, marker)
        touched[2] += np.array([0.015, 0.0, 0.0])  # 15 mm off on C2

        problems = rcc.check_touch_geometry(touched, marker)
        self.assertTrue(problems)
        named = {label for pair in problems for label in pair[:2]}
        self.assertIn("C2", named)

    def test_swapped_corners_are_caught(self):
        T = _transform(0.0, 0.0, 0.0, [0.4, 0.0, 0.05])
        marker = rcc.marker_corner_points(MARKER_LENGTH_M)
        touched = tf.transform_points(T, marker)
        touched[[1, 2]] = touched[[2, 1]]

        problems = rcc.check_touch_geometry(touched, marker)
        self.assertTrue(problems, "swapping an edge for a diagonal must be detected")


class OrientationSpreadTest(unittest.TestCase):

    def test_identical_orientations_give_zero(self):
        poses = np.tile([0.4, 0.0, 0.1, 0.0, 3.14, 0.0], (4, 1))
        self.assertAlmostEqual(rcc.orientation_spread_deg(poses), 0.0, places=6)

    def test_measures_the_largest_pairwise_angle(self):
        rvec_a = np.zeros(3)
        rvec_b = np.array([0.0, 0.0, np.radians(30.0)])
        poses = np.array([
            [0.0, 0.0, 0.0, *rvec_a],
            [0.0, 0.0, 0.0, *rvec_b],
        ])
        self.assertAlmostEqual(rcc.orientation_spread_deg(poses), 30.0, places=4)


class SolveRobotCameraTest(unittest.TestCase):
    """The full composition, recovered from a known ground truth."""

    def setUp(self):
        # Camera looking down at the workspace from 1.8 m, tilted 30 deg.
        self.T_base_from_camera = _transform(-150.0, 0.0, 25.0, [0.85, -0.40, 1.80])
        # Marker lying flat on the table in front of the robot.
        self.T_base_from_marker = _transform(0.0, 0.0, 15.0, [0.62, 0.05, 0.00])
        self.T_camera_from_marker = tf.compose_transforms(
            tf.invert_transform(self.T_base_from_camera), self.T_base_from_marker)

    def test_recovers_the_camera_pose_from_perfect_touches(self):
        marker = rcc.marker_corner_points(MARKER_LENGTH_M)
        touched = tf.transform_points(self.T_base_from_marker, marker)
        poses = np.hstack([touched, np.zeros((4, 3))])

        result = rcc.solve_robot_camera(poses, self.T_camera_from_marker, MARKER_LENGTH_M)

        np.testing.assert_allclose(
            result["T_base_from_camera"], self.T_base_from_camera, atol=1e-9)
        np.testing.assert_allclose(
            result["T_base_from_marker"], self.T_base_from_marker, atol=1e-9)
        self.assertAlmostEqual(result["touch_rms_m"], 0.0, places=12)

    def test_world_from_robot_base_is_the_inverse_of_base_from_marker(self):
        """The repo-native field: extrinsics.json stores the robot base in the
        world frame, and the marker is the world origin by convention."""
        marker = rcc.marker_corner_points(MARKER_LENGTH_M)
        touched = tf.transform_points(self.T_base_from_marker, marker)
        poses = np.hstack([touched, np.zeros((4, 3))])

        result = rcc.solve_robot_camera(poses, self.T_camera_from_marker, MARKER_LENGTH_M)
        np.testing.assert_allclose(
            tf.compose_transforms(result["T_world_from_robot_base"],
                                  result["T_base_from_marker"]),
            np.eye(4), atol=1e-9)

    def test_survives_the_optical_path_through_solve_marker_pose(self):
        """Project the marker with a real K, run the detection-side solve on
        those pixels, and confirm the end-to-end result still lands on the
        truth. Catches a sign or corner-order error that a pure matrix test
        would not."""
        intrinsics = ROOT / "1_recognition" / "calib_data" / "intrinsics.json"
        if not intrinsics.exists():
            self.skipTest(f"no intrinsics at {intrinsics}")
        from camera_utils.calibration_io import load_intrinsics
        K, dist, _size = load_intrinsics(intrinsics)

        marker = rcc.marker_corner_points(MARKER_LENGTH_M)
        corners_camera = tf.transform_points(self.T_camera_from_marker, marker)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        pixels, _ = cv2.projectPoints(corners_camera, rvec, tvec, K, dist)
        pixels = pixels.reshape(4, 2)

        T_camera_from_marker = solve_marker_pose(pixels, MARKER_LENGTH_M, K, dist)
        self.assertIsNotNone(T_camera_from_marker)

        touched = tf.transform_points(self.T_base_from_marker, marker)
        poses = np.hstack([touched, np.zeros((4, 3))])
        result = rcc.solve_robot_camera(poses, T_camera_from_marker, MARKER_LENGTH_M)

        # Sub-millimetre on translation, well under a degree on rotation.
        np.testing.assert_allclose(
            result["T_base_from_camera"][:3, 3], self.T_base_from_camera[:3, 3], atol=1e-3)
        relative = result["T_base_from_camera"][:3, :3].T @ self.T_base_from_camera[:3, :3]
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertLess(angle_deg, 0.5)


class CameraUncertaintyTest(unittest.TestCase):
    """The lever-arm estimate: touch error over a 200 mm marker becomes a much
    larger error in a camera pose metres away."""

    def setUp(self):
        self.T_base_from_camera = _transform(-150.0, 0.0, 25.0, [0.85, -0.40, 1.80])
        self.T_base_from_marker = _transform(0.0, 0.0, 15.0, [0.62, 0.05, 0.00])

    def test_predicts_the_measured_amplification(self):
        """Pinned against an actual noisy run: 0.8 mm of per-point touch noise
        on this geometry gave a 0.59 mm residual and 7.5 mm of camera-position
        error. The estimate has to land near that, or it is not worth printing.
        """
        _ratio, uncertainty = rcc.estimate_camera_uncertainty(
            0.00059, self.T_base_from_camera, self.T_base_from_marker, MARKER_LENGTH_M)
        self.assertAlmostEqual(uncertainty, 0.0075, delta=0.0015)

    def test_error_scales_inversely_with_marker_size(self):
        """The actionable part: a bigger marker is the fix."""
        _r1, small = rcc.estimate_camera_uncertainty(
            0.0005, self.T_base_from_camera, self.T_base_from_marker, 0.200)
        _r2, large = rcc.estimate_camera_uncertainty(
            0.0005, self.T_base_from_camera, self.T_base_from_marker, 0.400)
        self.assertAlmostEqual(small / large, 2.0, places=9)

    def test_lever_ratio_is_distance_over_corner_radius(self):
        ratio, _u = rcc.estimate_camera_uncertainty(
            0.001, self.T_base_from_camera, self.T_base_from_marker, MARKER_LENGTH_M)
        distance = np.linalg.norm(
            self.T_base_from_camera[:3, 3] - self.T_base_from_marker[:3, 3])
        self.assertAlmostEqual(ratio, distance / (MARKER_LENGTH_M / np.sqrt(2)), places=9)


class ReportingTest(unittest.TestCase):

    def test_grasshopper_block_carries_the_frame(self):
        T = _transform(10.0, 20.0, 30.0, [0.41, -0.23, 0.09])
        block = rcc.format_grasshopper_block(T)
        self.assertIn("origin  +0.410000, -0.230000, +0.090000", block)
        self.assertIn("x_axis", block)
        self.assertIn("T_base_from_camera", block)

    def test_module_imports_without_a_robot(self):
        """ur-rtde is imported inside connect_robot, so the math stays usable
        on a machine with no robot on the network."""
        self.assertTrue(hasattr(rcc, "solve_robot_camera"))
        self.assertNotIn("rtde_receive", dir(rcc))


if __name__ == "__main__":
    unittest.main()
