"""load_extrinsics reads both camera-pose files: an extrinsics file and a robot-camera
calibration (setup/calibration/robot_camera_calibration.py)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))

from camera_utils import transforms as tf
from camera_utils.calibration_io import load_extrinsics, save_extrinsics


def pose(rpy_deg, xyz):
    return tf.make_transform(tf.rpy_deg_to_matrix(*rpy_deg), xyz)


class LoadExtrinsicsTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        # A camera 4 m above a flat marker, looking down at it; the robot base beside it.
        self.T_camera_from_marker = tf.invert_transform(pose((180, 0, 30), (-2.0, -0.8, 4.0)))
        self.T_base_from_marker = pose((90, 0, 180), (0.43, 0.13, -0.14))

    def tearDown(self):
        self._dir.cleanup()

    def robot_calibration(self, **extra):
        """A file as robot_camera_calibration.py writes it."""
        path = self.dir / "robot_camera_calibration_test.json"
        data = {
            "T_base_from_camera": tf.compose_transforms(
                self.T_base_from_marker, tf.invert_transform(self.T_camera_from_marker)).tolist(),
            "T_base_from_marker": self.T_base_from_marker.tolist(),
            "T_camera_from_marker": self.T_camera_from_marker.tolist(),
            "T_world_from_robot_base": tf.invert_transform(self.T_base_from_marker).tolist(),
            "marker_id": 0,
            **extra,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, data

    def test_a_robot_camera_calibration_puts_the_world_at_its_marker(self):
        path, data = self.robot_calibration()
        with self.assertLogs("recognition", level="WARNING") as logs:
            T_world_from_camera, ground_z, T_world_from_robot_base = load_extrinsics(path)
        np.testing.assert_allclose(T_world_from_camera, tf.invert_transform(self.T_camera_from_marker),
                                   atol=1e-12)
        np.testing.assert_allclose(T_world_from_camera[:3, 3], (-2.0, -0.8, 4.0), atol=1e-12)
        # The robot base sits where the calibration measured it, in the same world.
        np.testing.assert_allclose(T_world_from_robot_base @ np.array(data["T_base_from_camera"]),
                                   T_world_from_camera, atol=1e-12)
        self.assertEqual(ground_z, 0.0)
        self.assertIn("ground_z", logs.output[0])  # the floor height is missing: said so

    def test_a_floor_height_added_to_the_file_is_used(self):
        path, _ = self.robot_calibration(ground_z=-0.65)
        self.assertEqual(load_extrinsics(path)[1], -0.65)

    def test_an_extrinsics_file_loads_as_before(self):
        path = self.dir / "extrinsics.json"
        T_world_from_camera = pose((170, 5, 0), (2.5, 0.4, 4.3))
        save_extrinsics(path, T_world_from_camera, ground_z=-0.65, T_world_from_robot_base=np.eye(4))
        loaded, ground_z, base = load_extrinsics(path)
        np.testing.assert_allclose(loaded, T_world_from_camera)
        self.assertEqual(ground_z, -0.65)
        np.testing.assert_allclose(base, np.eye(4))

    def test_any_other_file_is_refused(self):
        path = self.dir / "intrinsics.json"
        path.write_text(json.dumps({"K": np.eye(3).tolist(), "dist": [0] * 5, "image_size": [640, 360]}),
                        encoding="utf-8")
        with self.assertRaises(ValueError):
            load_extrinsics(path)


if __name__ == "__main__":
    unittest.main()
