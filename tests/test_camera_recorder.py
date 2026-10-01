"""The reactive system's camera recording (run_recorder.py): real .mp4 files from fake
frames, no camera needed."""

import csv
import json
import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_recorder


class FakeFrames:
    """A recorded source of n frames: read() hands them out, then None (exhausted)."""

    def __init__(self, n, on_read=None):
        self.frames = [np.full((48, 64, 3), i * 4 % 256, np.uint8) for i in range(n)]
        self.exhausted = False
        self.on_read = on_read
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads)
        if not self.frames:
            self.exhausted = True
            return None
        return self.frames.pop(0)


def ticking(fps, start=1_700_000_000.0):
    """A clock that moves 1/fps per call: a camera delivering fps frames per second."""
    times = (start + index / fps for index in range(10_000))
    return lambda: next(times)


def video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        return capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def timestamps(path):
    with open(path.with_suffix(".timestamps.csv"), encoding="utf-8") as file:
        return [float(row["timestamp_s"]) for row in csv.DictReader(file)]


class RecordTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self._dir.name)
        self.output = self.dir / "run" / "camera.mp4"

    def tearDown(self):
        # configure_logging's run log holds a file in the temporary directory.
        logger = logging.getLogger("recognition")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        self._dir.cleanup()

    def test_every_frame_is_written_at_the_rate_the_camera_delivers(self):
        clock = ticking(15.0)
        rate = run_recorder.record(FakeFrames(30), self.output, stop=lambda: False, clock=clock)
        self.assertAlmostEqual(rate, 15.0, places=1)
        fps, count = video(self.output)
        self.assertAlmostEqual(fps, 15.0, places=1)
        self.assertEqual(count, 30)
        stamps = timestamps(self.output)
        self.assertEqual(len(stamps), 30)
        self.assertAlmostEqual(stamps[0], 1_700_000_000.0, places=3)  # epoch, as communication logs
        self.assertAlmostEqual(stamps[-1] - stamps[0], 29 / 15.0, places=3)

    def test_a_run_stopped_before_the_rate_is_measured_keeps_its_frames(self):
        frames = FakeFrames(100)
        run_recorder.record(frames, self.output, stop=lambda: frames.reads >= 5, clock=ticking(10.0))
        self.assertEqual(video(self.output)[1], 5)
        self.assertEqual(len(timestamps(self.output)), 5)

    def test_a_given_rate_is_stamped_instead_of_measured(self):
        run_recorder.record(FakeFrames(3), self.output, stop=lambda: False, fps=25.0, clock=ticking(10.0))
        self.assertAlmostEqual(video(self.output)[0], 25.0, places=1)

    def test_a_rate_no_camera_delivers_is_not_stamped(self):
        rate = run_recorder.record(FakeFrames(30), self.output, stop=lambda: False, clock=ticking(5000.0))
        self.assertEqual(rate, run_recorder.FALLBACK_FPS)
        self.assertEqual(video(self.output)[1], 30)

    def test_q_in_the_preview_stops_and_finishes_the_video(self):
        shown = []
        run_recorder.record(FakeFrames(50), self.output, stop=lambda: False, clock=ticking(10.0),
                            preview=lambda frame: shown.append(frame) or len(shown) < 25)
        self.assertEqual(video(self.output)[1], 25)

    def test_ctrl_c_still_finishes_the_video(self):
        def interrupt(reads):
            if reads == 8:
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_recorder.record(FakeFrames(50, on_read=interrupt), self.output, stop=lambda: False,
                                clock=ticking(10.0))
        self.assertEqual(video(self.output)[1], 7)

    def test_the_stop_file_ends_a_run_started_from_the_command_line(self):
        source = self.dir / "source.mp4"
        run_recorder.record(FakeFrames(40), source, stop=lambda: False, clock=ticking(20.0))
        with redirect_stdout(StringIO()):
            run_recorder.main(["--video-source", str(source), "--log-dir", str(self.dir),
                               "--run-name", "r1", "--no-display"])
        # A recorded source is read as fast as the disk allows: its own timeline gives the rate.
        fps, count = video(self.dir / "r1" / "camera.mp4")
        self.assertEqual(count, 40)
        self.assertAlmostEqual(fps, 20.0, places=1)
        self.assertTrue((self.dir / "r1" / "recorder.log").exists())
        # The stop file is there already: nothing is recorded.
        stop = self.dir / "camera.stop"
        stop.touch()
        with redirect_stdout(StringIO()):
            run_recorder.main(["--video-source", str(source), "--output", str(self.dir / "r2.mp4"),
                               "--stop-file", str(stop), "--no-display"])
        self.assertFalse((self.dir / "r2.mp4").exists())


class CameraTests(unittest.TestCase):
    def args(self, *argv):
        return run_recorder._parse_args(list(argv))

    def test_a_webcam_opens_at_the_calibrated_resolution_without_calibrating(self):
        with tempfile.TemporaryDirectory() as calib:
            Path(calib, "cam.json").write_text(json.dumps(
                {"K": [[500, 0, 320], [0, 500, 180], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                 "image_size": [640, 360]}), encoding="utf-8")
            camera = run_recorder.camera_config(self.args("--video-source", "6", "--calib-dir", calib,
                                                          "--intrinsics-file", "cam.json"))
        self.assertEqual((camera.video_source, camera.capture_width, camera.capture_height), ("6", 640, 360))
        self.assertIsNone(camera.calib_dir)  # nothing to fail on: a recording needs no K

    def test_an_explicit_size_wins_and_no_intrinsics_lets_the_camera_pick(self):
        camera = run_recorder.camera_config(self.args("--capture-width", "1920", "--capture-height", "1080"))
        self.assertEqual((camera.video_source, camera.capture_width, camera.capture_height), (0, 1920, 1080))
        with tempfile.TemporaryDirectory() as empty:
            camera = run_recorder.camera_config(self.args("--calib-dir", empty))
        self.assertEqual((camera.capture_width, camera.capture_height), (None, None))

    def test_iphone(self):
        camera = run_recorder.camera_config(self.args("--iphone", "--iphone-rotate", "270"))
        self.assertEqual((camera.video_source, camera.capture_rotate90), ("iphone", 270))

    def test_output_goes_to_the_run_directory(self):
        args = self.args("--log-dir", "logs/runs", "--run-name", "demo_p01")
        self.assertEqual(run_recorder.output_path(args), Path("logs/runs/demo_p01/camera.mp4"))
        self.assertEqual(run_recorder.output_path(self.args("--output", "take.mp4")), Path("take.mp4"))


if __name__ == "__main__":
    unittest.main()
