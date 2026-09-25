"""Side files of a recorded take, for replaying it without a camera.

A take is a raw .mp4 (run_recognition.py --record-raw, or
eval/pose_detection_live.py --record-raw) plus, beside it:

  <name>.timestamps.csv   frame, timestamp_s -- when each frame was captured
                          (render_utils/video_recorder.py writes it). The .mp4
                          carries one nominal fps, but a live loop runs at
                          whatever rate it manages, so this is the real timing.
  <name>.location.csv     frame, timestamp_s, world_x, world_y, world_z, camera_z
                          -- the human location detected live
                          (pose_detection_live.py --save-location). Empty
                          cells mean nobody was detected on that frame.

Both are indexed by frame number, i.e. row N belongs to frame N of the .mp4.
"""

from __future__ import annotations

import csv
from pathlib import Path

from logging_setup import get_logger

logger = get_logger(__name__)

# Velocity is differenced over at least this span: the recorded positions are
# already smoothed, but frame-to-frame differences would still amplify their noise.
VELOCITY_MIN_SPAN_S = 0.2
# ...and at most this one: across a longer detection gap there is no velocity to report.
VELOCITY_MAX_SPAN_S = 1.0


def timestamps_path_for(video_path) -> Path:
    return Path(video_path).with_suffix(".timestamps.csv")


def load_frame_timestamps(video_path) -> list[float] | None:
    """Per-frame capture times from the take's .timestamps.csv, relative to the
    first frame (the file holds wall-clock times for a live recording). None if
    there is no such file or it has gaps, so the caller falls back to index / fps."""
    csv_path = timestamps_path_for(video_path)
    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        values = [row["timestamp_s"] for row in csv.DictReader(f)]
    if not values or any(value == "" for value in values):
        logger.warning("%s has missing timestamps; using the video's fps instead.", csv_path)
        return None
    times = [float(value) for value in values]
    logger.info("Using recorded frame timestamps from %s (%d frames, %.1f s).",
                csv_path, len(times), times[-1] - times[0])
    return [t - times[0] for t in times]


class RecordedLocations:
    """The human location detected live, looked up by frame index."""

    def __init__(self, rows: list[tuple[float, tuple[float, float, float] | None]]):
        # rows[frame] = (timestamp_s, (x, y, z) or None when nobody was detected)
        self.rows = rows

    @classmethod
    def from_csv(cls, path) -> "RecordedLocations":
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cells = (row["world_x"], row["world_y"], row["world_z"])
                xyz = None if "" in cells else tuple(float(v) for v in cells)
                rows.append((float(row["timestamp_s"]), xyz))
        detected = sum(xyz is not None for _t, xyz in rows)
        logger.info("Loaded recorded locations from %s (%d frames, %d with a detection).",
                    path, len(rows), detected)
        if not detected:
            logger.warning("%s has no world locations -- was it recorded without extrinsics?", path)
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def at(self, frame_index: int | None):
        """(timestamp_s, (x, y, z)) for this frame, or None when it has no location."""
        if frame_index is None or not 0 <= frame_index < len(self.rows):
            return None
        timestamp, xyz = self.rows[frame_index]
        return None if xyz is None else (timestamp, xyz)

    def velocity(self, frame_index: int | None):
        """World-frame m/s, differenced back to the latest detected frame between
        VELOCITY_MIN_SPAN_S and VELOCITY_MAX_SPAN_S earlier; None when there is none."""
        current = self.at(frame_index)
        if current is None:
            return None
        timestamp, xyz = current
        for earlier in range(frame_index - 1, -1, -1):
            earlier_timestamp, earlier_xyz = self.rows[earlier]
            span = timestamp - earlier_timestamp
            if span > VELOCITY_MAX_SPAN_S:
                return None
            if earlier_xyz is None or span < VELOCITY_MIN_SPAN_S:
                continue
            return tuple((a - b) / span for a, b in zip(xyz, earlier_xyz))
        return None
