"""Frame acquisition for RecognitionManager.

Opens whatever `CameraConfig.video_source` names -- a webcam index, an
iPhone over Record3D, a stream URL or a recorded file -- and hands back one
frame at a time together with its position on the RECORDING's timeline.

Extracted from RecognitionManager because it is a self-contained state
machine that had nothing to do with recognition: ~11 attributes of capture
and playback bookkeeping (the demuxer handle, the playback anchor, the
frame/drop counters) that no other part of the manager reads.

Two read modes, and the difference between them is the point:

  - **frame by frame** (the default): every frame the file contains is
    handed over, however long inference takes. Timestamps still come from
    the recording (`index / fps`), not `time.time()`, so a 30 fps clip
    processed at 10 fps doesn't look like a human moving at a third speed
    to the velocity/acceleration fit downstream.
  - **wall clock** (`camera.realtime_playback`): the frame handed back is
    the one that would have been the most recent to ARRIVE at the moment we
    ask, so a recorded source drops frames exactly the way a live camera
    does. Running the same clip both ways is what measures how much of the
    offline/live accuracy gap comes from processing latency -- see
    eval/run_logger.py and eval/analyse_runs.py.

A live source is timestamped with `None` (the caller falls back to
`time.time()`): its frames really do arrive when they arrive, and it has no
recording timeline to be placed on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)

# Live-stream URL schemes for video_source auto-detection, see classify().
_STREAM_URL_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")

# Accepted CameraConfig.capture_backend values, also used by setup/video_source.py.
BACKEND_NAMES = ("auto", "dshow", "msmf", "any")


def resolve_backend(name="auto"):
    """Map a backend name to a cv2.CAP_* constant.

    'auto' (and None) picks DirectShow on Windows, where it is markedly more
    reliable than Media Foundation for virtual cameras, and leaves the choice
    to OpenCV everywhere else.

    cv2 is imported inside the function on purpose: this module must stay
    importable without it, since RecognitionManager builds a FrameSource in
    its constructor and nothing should pay for OpenCV until a frame is read.
    """
    import cv2

    if name is None or name == "auto":
        return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
    if name not in backends:
        raise ValueError(
            f"Unknown capture backend {name!r}; expected one of {sorted(backends)} or 'auto'.")
    return backends[name]


def classify_video_source(video_source) -> tuple[bool, bool]:
    """Returns (is_live, is_iphone) for a video_source value, so callers
    don't have to pass live= explicitly:
      - an int, or a digit-only string (e.g. "0") -> live webcam index.
      - "iphone" (case-insensitive) -> live iPhone via Record3D.
      - a string starting with a known stream URL scheme (rtsp://,
        http(s)://, udp://, tcp://) -> live network stream.
      - anything else (a path, or an unrecognized string) -> a recorded
        file, the same assumption cv2.VideoCapture makes for a plain path.
    """
    if isinstance(video_source, int):
        return True, False
    if not isinstance(video_source, str):
        return False, False

    normalized = video_source.strip().lower()
    if normalized == "iphone":
        return True, True
    if normalized.isdigit():
        return True, False
    if normalized.startswith(_STREAM_URL_PREFIXES):
        return True, False
    return False, False


class FrameSource:
    """Owns one capture and its playback clock.

    camera: the live CameraConfig (see vision_model/vision_config.py). It is
    read on every frame and `camera.live` is written back once the source has
    been classified, so the caller can introspect it afterwards.
    fallback_fps: used to timestamp a recorded container that reports no
    usable frame rate of its own (VisionConfig.fps).
    """

    def __init__(self, camera, *, fallback_fps: float):
        self.camera = camera
        self.fallback_fps = float(fallback_fps)

        self._cv2 = None
        self._capture = None

        # Playback clock. _last_timestamp carries the current frame's position on the
        # RECORDING's timeline; it stays None for a live source.
        self._anchor: float | None = None
        self._fps: float | None = None
        self._next_index = 0
        self._last_timestamp: float | None = None

        self.frames_read = 0
        self.frames_dropped = 0
        self.frame_index: int | None = None  # source index of the frame just read
        self.dropped_before: int = 0         # frames skipped to reach that one

        # Set once a RECORDED source runs out. A live source returning no frame is a
        # hiccup to ride out; a file that ends is the end of the run.
        self.exhausted = False

        # What _open_capture asked the device for, and whether the first decoded frame
        # has been checked against it yet (see _verify_frame_size).
        self.requested_size: tuple[int, int] | None = None
        self._size_verified = False

        # Calibration preflight, loaded once when the capture opens so a missing or bad
        # calibration fails at camera-open time rather than on the first processed frame.
        self.K = None
        self.dist = None
        self.image_size = None
        self.T_world_from_camera = None
        self.have_extrinsics = False

    # -- reading -----------------------------------------------------------

    @property
    def last_timestamp(self) -> float | None:
        """Position of the frame just read on the recording's timeline, or
        None for a live source (the caller should use time.time())."""
        return self._last_timestamp

    @property
    def last_intrinsics(self):
        """K for the frame just read, when the SOURCE reports its own per frame
        (an iPhone via Record3D does -- see IPhoneVideoCaptureAdapter), else None
        and the caller should use the calibrated file K. Autofocus moves fx/fy, so
        for those sources the file K is only ever right on average."""
        return getattr(self._capture, "last_intrinsics", None)

    def read(self):
        """Next frame, or None if the source is closed or exhausted."""
        if self.camera.video_source is None:
            return None

        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Reading from a video source requires opencv-python.") from exc
            self._cv2 = cv2

        if self._capture is None:
            self._load_calibration()
            self._capture = self._open_capture()

        # camera.live was written back by _open_capture(), so it is a real bool by now.
        if self.camera.realtime_playback and not self.camera.live:
            frame = self._read_on_wall_clock()
        else:
            ok, frame = self._capture.read()
            frame = frame if ok else None
            self._stamp_sequential(frame, live=bool(self.camera.live))

        if frame is not None and not self._size_verified:
            self._verify_frame_size(frame)

        if frame is None and not self.camera.live:
            self.exhausted = True
        return frame

    def _verify_frame_size(self, frame) -> None:
        """Check the first decoded frame against what was requested and what
        the calibration was solved at. Runs once per capture.

        Only a decoded frame can answer this: cap.get(CAP_PROP_FRAME_WIDTH)
        frequently echoes back whatever was just set even when the device
        ignored it.

        It matters more here than a resolution mismatch usually would,
        because K is only valid at the resolution it was solved for. Frames
        at any other size scale fx, fy, cx and cy, so depth comes out wrong
        by that factor and the back-projected ray is tilted -- every world
        position is wrong, and nothing else about the run looks abnormal.

        Warns rather than raises: a recorded clip at another size is still
        worth running for posture, and aborting a live take is worse than
        telling the operator what they are getting.
        """
        self._size_verified = True
        actual = (int(frame.shape[1]), int(frame.shape[0]))

        if self.requested_size is not None and actual != self.requested_size:
            logger.warning(
                "Requested %dx%d but the source delivered %dx%d. For an OBS Virtual "
                "Camera, set Settings > Video > Output (Scaled) Resolution, then stop "
                "and start the virtual camera.",
                *self.requested_size, *actual)

        calibrated = tuple(self.image_size) if self.image_size is not None else None
        if calibrated is not None and actual != calibrated:
            logger.warning(
                "Frames are %dx%d but the intrinsics in %s were calibrated at %dx%d. "
                "K does not transfer between resolutions -- fx, fy, cx and cy are all "
                "out by roughly %.2fx, so every world position will be wrong. Capture "
                "at %dx%d, or convert the calibration: setup/rescale_intrinsics.py "
                "--input %s --size %dx%d --output <new.json>",
                *actual, self.camera.intrinsics_file, *calibrated,
                actual[0] / calibrated[0] if calibrated[0] else float("nan"),
                *calibrated, self.camera.intrinsics_file, *actual)
        else:
            logger.info("Capturing at %dx%d.", *actual)

    def _stamp_sequential(self, frame, *, live: bool) -> None:
        """Put a plain frame-by-frame read on the RECORDING's timeline too.

        A recorded file's frames were captured 1/fps apart, whatever pace we
        replay them at, so that -- not time.time() -- is their real spacing,
        and it is what the streaming velocity/acceleration fit and the depth
        filter's time constants have to be fed. Wall-clock stamps would tell
        them a 30 fps recording processed at 10 fps shows a human moving at a
        third speed. It also puts a frame-by-frame run and a wall-clock run on
        one comparable axis, which is the whole point of running both.

        A LIVE source keeps time.time(): its frames really do arrive when they
        arrive, and it has no recording timeline to be placed on."""
        if live or frame is None:
            self._last_timestamp = None
            return

        index = self._next_index
        self._next_index += 1
        self._last_timestamp = index / self.source_fps()
        self.frame_index = index
        self.dropped_before = 0  # frame-by-frame never skips: that is the point
        self.frames_read += 1

    def source_fps(self) -> float:
        """The recording's own frame rate, cached. Falls back to
        fallback_fps for a container that doesn't report one."""
        if self._fps is None:
            fps = self._capture.get(self._cv2.CAP_PROP_FPS)
            if not fps or not np.isfinite(fps) or fps <= 0.0:
                fps = self.fallback_fps
                logger.warning(
                    "Video source %s reports no usable FPS; timestamping frames with "
                    "fallback fps=%.3f instead.", self.camera.video_source, fps)
            self._fps = float(fps)
        return self._fps

    def _read_on_wall_clock(self):
        """Play a recorded file against the WALL CLOCK instead of frame by
        frame, so a recorded source drops frames exactly the way a live
        camera does.

        Frame-by-frame reading hands the model every frame however long
        inference takes, which is the one thing a live camera never does: at
        30 fps and 100 ms of inference, a real camera silently discards ~2 of
        every 3 frames, so the model sees a ~10 fps, unevenly spaced stream.
        That changes the input to the temporal parts of the pipeline
        (MotionBERT's clip window, the streaming Savitzky-Golay
        velocity/acceleration fit, the step stabilizer's confirmation
        counting), which is why offline accuracy on a file can beat live
        accuracy on the same scene.

        Here, the frame handed back is the one that would have been the most
        recent to arrive at the moment we ask, i.e. index
        round((now - anchor) * playback_speed * fps):
          - Behind (model slower than the recording): everything in between is
            thrown away with grab(), which advances the demuxer WITHOUT
            decoding the pixels -- so skipped frames cost almost nothing and
            the skip itself doesn't distort the timing it is meant to measure.
          - Ahead (model faster): sleep until the frame is actually due,
            mirroring a live capture.read() blocking on the next exposure.

        Timestamps come from the RECORDING's timeline (frame_index / fps), not
        from time.time(), so the feature extractor sees the true spacing of the
        frames it was given (including the gaps left by dropped ones) and a run
        is reproducible. Frames are counted in frames_read / frames_dropped for
        reporting the realized rate afterwards."""
        capture = self._capture
        speed = float(self.camera.playback_speed)
        if speed <= 0.0:
            raise ValueError(f"camera.playback_speed must be > 0, got {self.camera.playback_speed}.")

        fps = self.source_fps()

        # First frame anchors the clock: it is always read in full, and every
        # later frame's due time is measured from the moment it was handed over.
        if self._anchor is None:
            ok, frame = capture.read()
            if not ok:
                return None
            self._next_index = 1
            self._last_timestamp = 0.0
            self.frames_read = 1
            self.frames_dropped = 0
            self.frame_index = 0
            self.dropped_before = 0
            self._anchor = time.perf_counter()
            return frame

        elapsed = time.perf_counter() - self._anchor
        due_index = int(elapsed * speed * fps)

        if due_index < self._next_index:
            wait_s = (self._next_index / fps) / speed - elapsed
            if wait_s > 0:
                time.sleep(wait_s)
            due_index = self._next_index

        dropped = 0
        while self._next_index < due_index:
            if not capture.grab():
                return None
            self._next_index += 1
            dropped += 1

        ok, frame = capture.read()
        if not ok:
            return None

        frame_index = self._next_index
        self._next_index += 1
        self._last_timestamp = frame_index / fps
        self.frames_read += 1
        self.frames_dropped += dropped
        self.frame_index = frame_index
        self.dropped_before = dropped
        if dropped:
            logger.debug(
                "Realtime playback: dropped %d frame(s) before frame %d (t=%.3fs) -- "
                "the model was behind the video clock.", dropped, frame_index,
                self._last_timestamp)
        return frame

    # -- opening and closing -----------------------------------------------

    def _open_capture(self):
        """Open camera.video_source -- a live iPhone (Record3D) connection
        for "iphone" (see camera_utils/iphone_connection.py), otherwise a
        plain cv2.VideoCapture (webcam index, video file path, or stream URL
        -- cv2.VideoCapture already handles all three). camera.live is
        auto-detected via classify_video_source() unless already set
        explicitly, and written back so it's introspectable afterwards."""
        is_live, is_iphone = classify_video_source(self.camera.video_source)
        if self.camera.live is None:
            self.camera.live = is_live

        if is_iphone:
            from camera_utils.iphone_connection import IPhoneVideoCaptureAdapter
            # max_wait_sec short on purpose. The adapter's default (30 s) exists so the
            # one-shot calibration scripts ride out a USB drop rather than aborting, but
            # here it would block the whole recognition loop for half a minute on a dead
            # stream -- the loop then logs a 30 s overrun and looks hung. A live source
            # returning None is already treated as a hiccup (see read() below), so the
            # loop keeps ticking while IPhoneCamera's watchdog reconnects underneath.
            return IPhoneVideoCaptureAdapter(
                dev_idx=self.camera.dev_idx, capture_rotate90=self.camera.capture_rotate90,
                max_wait_sec=self.camera.iphone_read_timeout_sec)

        source = self.camera.video_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        # A backend and a capture mode only mean anything for a live device. A recorded
        # file plays at the size it was written at, and CAP_DSHOW would refuse to open
        # one at all.
        if not self.camera.live:
            capture = self._cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video source: {self.camera.video_source}")
            return capture

        capture = self._cv2.VideoCapture(source, resolve_backend(self.camera.capture_backend))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.camera.video_source}")

        self.requested_size = self._resolve_requested_size()
        if self.requested_size is not None:
            capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_size[0]))
            capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_size[1]))
        return capture

    def _resolve_requested_size(self) -> tuple[int, int] | None:
        """The mode to ask a live device for: CameraConfig's explicit
        capture_width/height, else the resolution the intrinsics were
        calibrated at.

        Defaulting to the calibrated size rather than to "whatever the backend
        picks" is the point. Nothing negotiates this for us -- DirectShow
        settles on 640x480 however capable the device is -- so leaving it unset
        is how a 1080p camera ends up feeding a K solved at 1080p with 640x480
        frames. _load_calibration runs before this, so image_size is known.
        """
        if self.camera.capture_width and self.camera.capture_height:
            return int(self.camera.capture_width), int(self.camera.capture_height)
        if self.image_size is not None:
            return int(self.image_size[0]), int(self.image_size[1])
        return None

    def _load_calibration(self) -> None:
        """Load intrinsics (required for 3D posture) / extrinsics (optional
        -- defines the world origin, see eval/pose_detection_live.py's module
        docstring) once per capture, so a missing/bad calibration fails fast
        at camera-open time instead of on the first processed frame.
        RealtimeSkeleton3DPipeline loads its own copy independently when the
        LSTM pipeline spins up -- this is just an early sanity check."""
        if self.K is not None or self.have_extrinsics:
            return
        if self.camera.calib_dir is None:
            return

        from camera_utils.calibration_io import load_extrinsics, load_intrinsics

        calib_dir = Path(self.camera.calib_dir)
        intrinsics_path = calib_dir / self.camera.intrinsics_file
        extrinsics_path = calib_dir / self.camera.extrinsics_file
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"No intrinsics found at {intrinsics_path}. Run setup/calibrate_camera.py "
                "first (see eval/pose_detection_live.py's module docstring).")
        self.K, self.dist, self.image_size = load_intrinsics(intrinsics_path)

        self.have_extrinsics = extrinsics_path.exists()
        if self.have_extrinsics:
            self.T_world_from_camera, _ground_z, _robot_base = load_extrinsics(extrinsics_path)
            camera_up_world = self.T_world_from_camera[:3, :3] @ np.array([0.0, -1.0, 0.0])
            tilt_deg = np.degrees(np.arccos(np.clip(camera_up_world[2], -1.0, 1.0)))
            logger.info("Loaded extrinsics from %s -- camera tilt ~%.1fdeg from vertical "
                        "per this calibration.", extrinsics_path, tilt_deg)
        else:
            logger.warning("No extrinsics at %s -- posture will stay camera-frame, see "
                           "eval/pose_detection_live.py's module docstring.", extrinsics_path)

    def release(self) -> None:
        """Release the capture and reset the playback clock. A new capture
        starts a new playback timeline (see _read_on_wall_clock)."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.exhausted = False
        self._anchor = None
        self._fps = None
        self._next_index = 0
        self._last_timestamp = None
        self.requested_size = None
        self._size_verified = False
