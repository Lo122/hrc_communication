"""Lazily opened .mp4 writer shared by the recognition recordings.

Used for the rendered debug view (DebugView, --record) and for the raw camera
frames (RecognitionManager, --record-raw).
"""

from __future__ import annotations

import csv
from pathlib import Path

from logging_setup import get_logger

logger = get_logger(__name__)


class VideoRecorder:
    """Appends frames to an .mp4, opening the writer on the first frame.

    The frame size comes from that first frame, so callers need not know it up
    front. A writer that will not open (missing codec, unwritable path) disables
    the recording and logs once, rather than raising: losing a recording should
    not take the recognition run down with it. A mid-run size change would be
    written as garbage by VideoWriter, so those frames are dropped and reported once.

    With write_timestamps, a sibling <name>.timestamps.csv gets one row per written
    frame. An .mp4 carries a single constant fps, while frames arrive at whatever
    rate the loop manages, so the CSV is the record of when each frame was taken.
    """

    def __init__(self, path: str | Path, fps: float, *, label: str,
                 write_timestamps: bool = False):
        self.path = Path(path)
        self.fps = float(fps)
        self.label = label
        self.write_timestamps = write_timestamps
        self._writer = None
        self._size: tuple[int, int] | None = None
        self._size_warned = False
        self._failed = False
        self._frames_written = 0
        self._timestamps_file = None
        self._timestamps_csv = None

    def write(self, frame, timestamp: float | None = None) -> None:
        if self._failed:
            return
        height, width = frame.shape[:2]
        if self._writer is None and not self._open(width, height):
            return
        if (width, height) != self._size:
            if not self._size_warned:
                logger.warning("%s frame size changed from %s to %s; those frames are left "
                               "out of %s.", self.label.capitalize(), self._size, (width, height),
                               self.path)
                self._size_warned = True
            return
        self._writer.write(frame)
        if self._timestamps_csv is not None:
            self._timestamps_csv.writerow(
                [self._frames_written, "" if timestamp is None else f"{timestamp:.6f}"])
        self._frames_written += 1

    def _open(self, width: int, height: int) -> bool:
        import cv2

        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (width, height))
        if not writer.isOpened():
            logger.error("Could not open %s for recording the %s (codec or path problem); "
                         "continuing without it.", self.path, self.label)
            self._failed = True
            return False
        self._writer = writer
        self._size = (width, height)
        if self.write_timestamps:
            self._timestamps_file = open(self.path.with_suffix(".timestamps.csv"), "w",
                                         newline="", encoding="utf-8")
            self._timestamps_csv = csv.writer(self._timestamps_file)
            self._timestamps_csv.writerow(["frame", "timestamp_s"])
        logger.info("Recording the %s to %s (%dx%d @ %.1f fps)",
                    self.label, self.path, width, height, self.fps)
        return True

    def close(self) -> None:
        if self._writer is not None:
            # Without this the container is left without its index and the file is
            # unplayable.
            self._writer.release()
            self._writer = None
            logger.info("%s recording written to %s (%d frames)",
                        self.label.capitalize(), self.path, self._frames_written)
        if self._timestamps_file is not None:
            self._timestamps_file.close()
            self._timestamps_file = None
