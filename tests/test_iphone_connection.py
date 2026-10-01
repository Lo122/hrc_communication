"""IPhoneCamera's frame hand-off from Record3D: the receive-thread callback only takes
the frame (anything slow there queues frames up in the USB stream, and the picture lags
further and further), the reader gets the newest one, upright. No phone needed."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))

from camera_utils.iphone_connection import (
    IPhoneCamera, _ReceiveStats, _owned, _rotate_intrinsics_90,
)

RAW_H, RAW_W = 48, 64  # the sensor's own orientation


class FakeStream:
    """record3d.Record3DStream's getters: a fresh copy of the current frame per call."""

    def __init__(self):
        self.frame(0)

    def frame(self, value):
        self.rgb = np.full((RAW_H, RAW_W, 3), value, np.uint8)
        self.rgb[0, 0] = (255, 0, 0)  # a marked corner shows the rotation
        self.depth = np.arange(RAW_H * RAW_W, dtype=np.float32).reshape(RAW_H, RAW_W)
        return self

    def get_rgb_frame(self):
        return self.rgb.copy()

    def get_depth_frame(self):
        return self.depth.copy()

    def get_intrinsic_mat(self):
        return SimpleNamespace(fx=50.0, fy=51.0, tx=30.0, ty=20.0)

    def get_camera_pose(self):
        return SimpleNamespace(qx=0.0, qy=0.0, qz=0.0, qw=1.0, tx=0.0, ty=0.0, tz=0.0)


def arrive(camera, stream):
    camera._on_new_frame(stream, camera._generation)


class FrameHandOffTests(unittest.TestCase):
    def test_the_callback_only_takes_the_frame_and_the_reader_gets_it_upright(self):
        camera, stream = IPhoneCamera(capture_rotate90=270), FakeStream()
        arrive(camera, stream)
        stored_rgb = camera._latest[0]
        self.assertEqual(stored_rgb.shape, (RAW_H, RAW_W, 3))  # not rotated on the receive thread
        rgb, depth, K, pose = camera.get_latest_frame(timeout=0.1)
        flag = cv2.ROTATE_90_COUNTERCLOCKWISE
        np.testing.assert_array_equal(rgb, cv2.rotate(stream.rgb, flag))
        np.testing.assert_array_equal(depth, cv2.rotate(stream.depth, flag))
        expected_K, size = _rotate_intrinsics_90(
            np.array([[50.0, 0, 30.0], [0, 51.0, 20.0], [0, 0, 1]]), (RAW_W, RAW_H), 270)
        np.testing.assert_array_equal(K, expected_K)
        self.assertEqual(rgb.shape[1::-1], size)
        self.assertEqual(pose.qw, 1.0)

    def test_without_rotation_the_frame_is_handed_over_as_taken(self):
        camera, stream = IPhoneCamera(), FakeStream()
        arrive(camera, stream)
        rgb, depth, K, _pose = camera.get_latest_frame(timeout=0.1)
        np.testing.assert_array_equal(rgb, stream.rgb)
        np.testing.assert_array_equal(K, [[50.0, 0, 30.0], [0, 51.0, 20.0], [0, 0, 1]])

    def test_only_the_newest_frame_is_read_and_skipped_ones_are_counted(self):
        camera, stream = IPhoneCamera(capture_rotate90=270), FakeStream()
        for value in (10, 20, 30):  # three arrive while the reader is busy
            arrive(camera, stream.frame(value))
        rgb = camera.get_latest_frame(timeout=0.1)[0]
        self.assertEqual(int(rgb[5, 5, 1]), 30)
        self.assertEqual(camera.last_seq, 3)  # FrameSource sees the 2 dropped from the seq gap
        self.assertIsNone(camera.get_latest_frame(timeout=0.05))  # nothing new: no stale repeat

    def test_a_reader_holding_a_frame_is_unaffected_by_the_next_one(self):
        camera, stream = IPhoneCamera(), FakeStream()
        arrive(camera, stream.frame(10))
        held = camera.get_latest_frame(timeout=0.1)[0]
        arrive(camera, stream.frame(99))
        self.assertEqual(int(held[5, 5, 1]), 10)

    def test_a_view_of_a_native_buffer_is_copied_an_owned_array_is_not(self):
        buffer = np.zeros((4, 4), np.uint8)
        view = buffer[:2]
        self.assertIsNot(_owned(view), view)
        self.assertTrue(_owned(view).flags["OWNDATA"])
        self.assertIs(_owned(buffer), buffer)


class ReceiveStatsTests(unittest.TestCase):
    def test_a_callback_busy_most_of_the_time_is_reported(self):
        stats = _ReceiveStats(window_s=1.0)
        with self.assertLogs("recognition", level="WARNING") as logs:
            for frame in range(31):  # 30 fps, each frame's callback taking 25 ms of 33
                started = frame / 30
                stats.note(started, started + 0.025)
        self.assertIn("queuing", logs.output[0])

    def test_a_callback_that_keeps_up_is_only_logged_for_the_record(self):
        stats = _ReceiveStats(window_s=1.0)
        with self.assertLogs("recognition", level="DEBUG") as logs:
            for frame in range(31):
                started = frame / 30
                stats.note(started, started + 0.003)
        self.assertTrue(all(record.levelname == "DEBUG" for record in logs.records))
        self.assertIn("busy 9%", logs.output[0])

    def test_the_callback_feeds_the_stats(self):
        camera, stream = IPhoneCamera(), FakeStream()
        noted = []
        camera._receive_stats = SimpleNamespace(note=lambda started, finished: noted.append(finished - started))
        arrive(camera, stream)
        self.assertEqual(len(noted), 1)
        self.assertGreaterEqual(noted[0], 0.0)


if __name__ == "__main__":
    unittest.main()
