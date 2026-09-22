"""Capture resolution: the K rescaling, and FrameSource requesting/verifying a mode.

These exist because a resolution mismatch is silent end to end -- the skeleton
still looks right in the overlay, and only the world coordinates are wrong.
"""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))
sys.path.insert(0, str(ROOT / "1_recognition" / "setup"))

from camera_utils.calibration_io import load_intrinsics, scale_intrinsics  # noqa: E402
from camera_utils.frame_source import FrameSource, resolve_backend  # noqa: E402

# A plausible 1080p calibration: ~60 deg horizontal FOV, principal point slightly
# off centre the way a real solve puts it.
K_1080P = np.array([
    [1400.0, 0.0, 962.5],
    [0.0, 1398.0, 536.0],
    [0.0, 0.0, 1.0],
])
SIZE_1080P = (1920, 1080)


class ScaleIntrinsicsTest(unittest.TestCase):

    def test_round_trip_restores_original(self):
        K_small, size_small = scale_intrinsics(K_1080P, SIZE_1080P, (640, 480))
        K_back, size_back = scale_intrinsics(K_small, size_small, SIZE_1080P)
        np.testing.assert_allclose(K_back, K_1080P, rtol=1e-12, atol=1e-9)
        self.assertEqual(size_back, SIZE_1080P)

    def test_matches_a_real_resize(self):
        """The property that actually matters, and the one a round trip cannot
        catch: projecting a 3D point with the scaled K must land where the
        resize puts the pixel the original K projected it to.

        cv2.resize maps x_new = (x_old + 0.5) * s - 0.5, so a wrong principal
        point convention shows up here as a constant half-pixel-ish offset.
        """
        K_half, _ = scale_intrinsics(K_1080P, SIZE_1080P, (960, 540))
        scale = 0.5

        points = np.array([
            [0.0, 0.0, 2.0],      # on the optical axis
            [0.4, -0.3, 2.5],
            [-0.9, 0.6, 4.0],
            [0.2, 0.15, 1.2],
        ])
        for point in points:
            full = K_1080P @ point
            full = full[:2] / full[2]
            halved = K_half @ point
            halved = halved[:2] / halved[2]
            expected = (full + 0.5) * scale - 0.5
            np.testing.assert_allclose(halved, expected, rtol=0, atol=1e-9)

    def test_focal_lengths_scale_with_their_axis(self):
        K_new, size_new = scale_intrinsics(K_1080P, SIZE_1080P, (1280, 720))
        self.assertEqual(size_new, (1280, 720))
        self.assertAlmostEqual(K_new[0, 0], K_1080P[0, 0] * (1280 / 1920), places=9)
        self.assertAlmostEqual(K_new[1, 1], K_1080P[1, 1] * (720 / 1080), places=9)
        self.assertEqual(K_new[2, 2], 1.0)

    def test_aspect_change_warns(self):
        """16:9 -> 4:3 cannot be a pure resize, so the device must be cropping
        or letterboxing and the scaled K would be wrong."""
        with self.assertLogs("recognition.calibration_io", "WARNING") as logs:
            scale_intrinsics(K_1080P, SIZE_1080P, (640, 480))
        self.assertIn("aspect ratio", "\n".join(logs.output))

    def test_aspect_preserving_does_not_warn(self):
        logger_name = "recognition.calibration_io"
        with self.assertLogs(logger_name, "DEBUG") as logs:
            # assertLogs fails outright with no records, so emit a marker of our own
            # and assert nothing else joined it.
            import logging
            logging.getLogger(logger_name).debug("marker")
            scale_intrinsics(K_1080P, SIZE_1080P, (960, 540))
        self.assertEqual(len(logs.records), 1)

    def test_rejects_degenerate_sizes(self):
        with self.assertRaises(ValueError):
            scale_intrinsics(K_1080P, (0, 1080), (640, 480))
        with self.assertRaises(ValueError):
            scale_intrinsics(K_1080P, SIZE_1080P, (640, 0))


class ResolveBackendTest(unittest.TestCase):

    def test_named_backends_resolve(self):
        import cv2
        self.assertEqual(resolve_backend("dshow"), cv2.CAP_DSHOW)
        self.assertEqual(resolve_backend("any"), cv2.CAP_ANY)

    def test_auto_and_none_agree(self):
        self.assertEqual(resolve_backend("auto"), resolve_backend(None))

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            resolve_backend("directshow")


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in that records property writes and
    delivers frames of a fixed size."""

    def __init__(self, delivered_size=(1920, 1080)):
        self.delivered_size = delivered_size
        self.properties = {}
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        width, height = self.delivered_size
        return True, np.zeros((height, width, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeCameraConfig:
    def __init__(self, **overrides):
        self.video_source = 0
        self.live = None
        self.realtime_playback = False
        self.playback_speed = 1.0
        self.dev_idx = 0
        self.capture_rotate90 = 0
        self.capture_width = None
        self.capture_height = None
        self.capture_backend = "any"
        self.calib_dir = None  # skips _load_calibration entirely
        self.intrinsics_file = "intrinsics.json"
        self.extrinsics_file = "extrinsics.json"
        self.__dict__.update(overrides)


def _frame_source(camera, capture, *, image_size=None):
    """A FrameSource wired to a fake cv2 and a fake capture, with the
    calibration preflight already satisfied."""
    source = FrameSource(camera, fallback_fps=30.0)
    source._cv2 = types.SimpleNamespace(
        CAP_PROP_FRAME_WIDTH="width", CAP_PROP_FRAME_HEIGHT="height",
        CAP_PROP_FPS="fps",
        VideoCapture=lambda *args, **kwargs: capture)
    source.image_size = image_size
    return source


class FrameSourceResolutionTest(unittest.TestCase):

    def test_live_source_requests_calibrated_size_by_default(self):
        capture = _FakeCapture((1920, 1080))
        source = _frame_source(_FakeCameraConfig(), capture, image_size=(1920, 1080))
        source.read()
        self.assertEqual(capture.properties, {"width": 1920.0, "height": 1080.0})
        self.assertEqual(source.requested_size, (1920, 1080))

    def test_explicit_size_overrides_the_calibrated_one(self):
        capture = _FakeCapture((1280, 720))
        camera = _FakeCameraConfig(capture_width=1280, capture_height=720)
        source = _frame_source(camera, capture, image_size=(1920, 1080))
        source.read()
        self.assertEqual(capture.properties, {"width": 1280.0, "height": 720.0})

    def test_recorded_file_is_never_asked_for_a_mode(self):
        """A file plays at the size it was written at; setting the property
        would be meaningless and CAP_DSHOW would refuse to open it."""
        capture = _FakeCapture((640, 480))
        camera = _FakeCameraConfig(video_source="clip.mp4",
                                   capture_width=1920, capture_height=1080)
        source = _frame_source(camera, capture, image_size=(1920, 1080))
        source.read()
        self.assertEqual(capture.properties, {})
        self.assertIsNone(source.requested_size)

    def test_mismatch_against_calibration_warns(self):
        capture = _FakeCapture((640, 480))
        camera = _FakeCameraConfig(capture_width=640, capture_height=480)
        source = _frame_source(camera, capture, image_size=(1920, 1080))
        with self.assertLogs("recognition.frame_source", "WARNING") as logs:
            source.read()
        message = "\n".join(logs.output)
        self.assertIn("640x480", message)
        self.assertIn("1920x1080", message)
        self.assertIn("rescale_intrinsics.py", message)

    def test_device_refusing_the_request_warns(self):
        capture = _FakeCapture((640, 480))  # asked for 1080p, gives 480p
        source = _frame_source(_FakeCameraConfig(), capture, image_size=(1920, 1080))
        with self.assertLogs("recognition.frame_source", "WARNING") as logs:
            source.read()
        self.assertIn("Requested 1920x1080", "\n".join(logs.output))

    def test_matching_size_is_verified_once_and_does_not_warn(self):
        capture = _FakeCapture((1920, 1080))
        source = _frame_source(_FakeCameraConfig(), capture, image_size=(1920, 1080))
        with self.assertLogs("recognition.frame_source", "INFO") as logs:
            source.read()
            source.read()
            source.read()
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        self.assertIn("Capturing at 1920x1080", logs.output[0])


class RescaleIntrinsicsCliTest(unittest.TestCase):

    def test_rescales_the_real_calibration_file(self):
        import tempfile

        from rescale_intrinsics import rescale_file

        intrinsics = ROOT / "1_recognition" / "calib_data" / "intrinsics.json"
        if not intrinsics.exists():
            self.skipTest(f"no calibration at {intrinsics}")

        K_old, _dist, size_old = load_intrinsics(intrinsics)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "intrinsics_1280x720.json"
            rescale_file(intrinsics, (1280, 720), out)
            K_new, _dist_new, size_new = load_intrinsics(out)

        self.assertEqual(size_new, (1280, 720))
        self.assertAlmostEqual(K_new[0, 0], K_old[0, 0] * (1280 / size_old[0]), places=6)

    def test_refuses_to_overwrite_the_original(self):
        from rescale_intrinsics import rescale_file

        intrinsics = ROOT / "1_recognition" / "calib_data" / "intrinsics.json"
        if not intrinsics.exists():
            self.skipTest(f"no calibration at {intrinsics}")
        with self.assertRaises(ValueError):
            rescale_file(intrinsics, (1280, 720), intrinsics)


if __name__ == "__main__":
    unittest.main()
