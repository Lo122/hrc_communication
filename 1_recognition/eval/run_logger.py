"""Per-frame result logging for recognition runs, so two runs of the SAME
recording can be compared afterwards.

Motivating question (see RecognitionManager._read_frame_on_wall_clock): does
processing latency change what the model predicts? Answering it needs more than
the trigger events -- a frame-by-frame run and a wall-clock ("realtime
playback") run of one video have to be laid side by side on the recording's own
timeline, which means recording, per frame, both what the model saw and what it
said.

A run writes one directory:

    <out_dir>/<run_name>/
        frames.csv   one row per frame the model PROCESSED
        events.csv   one row per RECOGNITION_TRIGGER-style event emitted
        run.json     configuration + end-of-run summary

frames.csv columns
    time, epoch_s      wall-clock time the row was written (local clock / epoch
                       seconds) -- the clock the communication layer's
                       communication_events.jsonl and timeline.csv use, so a run
                       of run_system.py lines up across both processes
    frame_index        index in the SOURCE video (NaN for a live camera)
    video_time_s       position on the recording's timeline (NaN for live) --
                       the x-axis every cross-run comparison is aligned on
    wall_time_s        seconds since the first processed frame
    dropped_before     source frames skipped to reach this one (0 without
                       realtime playback, where the model sees every frame)
    update_ms          wall-clock cost of the whole recognition_manager.update()
                       call -- the latency that CAUSES the drops
    detected           1 if the 3D lift produced a skeleton this frame
    warmup             1 while the LSTM window buffer is still filling
    raw_step_id        argmax of the step classifier, before stabilization
    stable_step_id     StepIdStabilizer's output (blank until it commits)
    confidence         max step probability
    progress           regression head's progress output
    world_x/y/z        world-frame pelvis position, blank when unavailable

events.csv columns: time, epoch_s, wall_time_s and video_time_s as above, then
the event sent -- type, task_name, round/step/piece id, confidence, progress.

Rows are flushed as they are written, so a run killed with Ctrl-C still leaves
a usable CSV behind.

Analyse the result with analyse_runs.py.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

FRAME_COLUMNS = [
    "time", "epoch_s", "frame_index", "video_time_s", "wall_time_s", "dropped_before", "update_ms",
    "detected", "warmup", "raw_step_id", "stable_step_id", "confidence", "progress",
    # Blank for a model trained without a mistake head. mistake_score is 1 - P(no
    # mistake), so it stays meaningful if a later model has more than two classes.
    "mistake_id", "mistake_score",
    "world_x", "world_y", "world_z",
]

EVENT_COLUMNS = [
    "time", "epoch_s", "wall_time_s", "video_time_s", "event_type", "task_name", "round_id",
    "step_id", "piece_id", "confidence", "progress",
]


def default_run_name(prefix: str = "run") -> str:
    """Timestamped name, so repeated runs never overwrite each other."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class RunLogger:
    """Writes frames.csv / events.csv / run.json for one recognition run.

    Usage mirrors the run loop itself -- log_frame() right after each
    update(), log_event() for each event sent, close() at the end:

        logger = RunLogger(out_dir, run_name, metadata={...})
        ...
        started = time.perf_counter()
        result = manager.update()
        logger.log_frame(manager, update_s=time.perf_counter() - started)
    """

    def __init__(self, out_dir: str | Path, run_name: str | None = None,
                 metadata: dict[str, Any] | None = None):
        self.run_dir = Path(out_dir) / (run_name or default_run_name())
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = dict(metadata or {})
        self.metadata.setdefault("run_name", self.run_dir.name)
        self.metadata.setdefault("started_at", datetime.now().isoformat(timespec="seconds"))
        self.metadata.setdefault("python", platform.python_version())
        self.metadata.setdefault("platform", platform.platform())

        self._frames_file = (self.run_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        self._frames = csv.DictWriter(self._frames_file, fieldnames=FRAME_COLUMNS)
        self._frames.writeheader()

        self._events_file = (self.run_dir / "events.csv").open("w", newline="", encoding="utf-8")
        self._events = csv.DictWriter(self._events_file, fieldnames=EVENT_COLUMNS)
        self._events.writeheader()

        # run.json is written once up front too, so an interrupted run still describes itself.
        self._write_metadata()

        self._wall_origin: float | None = None
        self._frame_count = 0
        self._update_ms: list[float] = []
        self._last_video_time: float | None = None

    # -- writing -----------------------------------------------------------

    def log_frame(self, manager, *, update_s: float) -> None:
        """Record one processed frame. Pulls everything from the manager's
        public per-frame state (last_frame_record / playback_*), so the run
        loop doesn't have to know what the pipeline produces."""
        now = time.perf_counter()
        if self._wall_origin is None:
            self._wall_origin = now

        record = getattr(manager, "last_frame_record", None) or {}
        world = record.get("world_xyz") or (None, None, None)
        video_time = record.get("timestamp")
        # A live source timestamps with time.time(), which is an epoch, not a position in
        # a recording. playback_frame_index is only ever set for a recorded file (either
        # read path), so it doubles as "this frame has a place on the video timeline".
        if manager.playback_frame_index is None:
            video_time = None
        self._last_video_time = video_time

        self._frames.writerow({
            **_wall_clock(),
            "frame_index": manager.playback_frame_index,
            "video_time_s": _round(video_time, 4),
            "wall_time_s": round(now - self._wall_origin, 4),
            "dropped_before": manager.playback_dropped_before,
            "update_ms": round(update_s * 1000.0, 3),
            "detected": int(bool(record.get("detected"))),
            "warmup": int(bool(record.get("warmup", True))),
            "raw_step_id": record.get("raw_step_id"),
            "stable_step_id": record.get("stable_step_id"),
            "confidence": _round(record.get("confidence"), 5),
            "progress": _round(record.get("progress"), 5),
            "mistake_id": record.get("mistake_id"),
            "mistake_score": _round(record.get("mistake_score"), 5),
            "world_x": _round(world[0], 4),
            "world_y": _round(world[1], 4),
            "world_z": _round(world[2], 4),
        })
        self._frames_file.flush()

        self._frame_count += 1
        self._update_ms.append(update_s * 1000.0)

    def log_event(self, event) -> None:
        """Record one emitted event, stamped with the video time of the frame
        that produced it -- that is what makes trigger timing comparable
        across runs of the same recording."""
        now = time.perf_counter()
        payload = getattr(event, "payload", None) or {}
        self._events.writerow({
            **_wall_clock(),
            "wall_time_s": round(now - (self._wall_origin or now), 4),
            "video_time_s": _round(self._last_video_time, 4),
            "event_type": getattr(getattr(event, "event_type", None), "name", str(event)),
            "task_name": payload.get("task_name"),
            "round_id": payload.get("round_id"),
            "step_id": payload.get("step_id"),
            "piece_id": payload.get("piece_id"),
            "confidence": _round(payload.get("confidence"), 5),
            "progress": _round(payload.get("progress"), 5),
        })
        self._events_file.flush()

    # -- finishing ---------------------------------------------------------

    def summary(self, manager=None) -> dict[str, Any]:
        wall = 0.0 if self._wall_origin is None else time.perf_counter() - self._wall_origin
        latencies = sorted(self._update_ms)
        summary: dict[str, Any] = {
            "frames_processed": self._frame_count,
            "wall_seconds": round(wall, 3),
            "effective_fps": round(self._frame_count / wall, 3) if wall > 0 else None,
            "update_ms_mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "update_ms_p50": round(_percentile(latencies, 50), 2) if latencies else None,
            "update_ms_p95": round(_percentile(latencies, 95), 2) if latencies else None,
            "update_ms_max": round(latencies[-1], 2) if latencies else None,
        }
        if manager is not None:
            read = getattr(manager, "playback_frames_read", 0)
            dropped = getattr(manager, "playback_frames_dropped", 0)
            total = read + dropped
            summary.update(
                source_frames=total or None,
                frames_dropped=dropped,
                # The headline number: what fraction of the recording the model
                # actually got to look at.
                frame_coverage=round(read / total, 4) if total else None,
            )
        return summary

    def close(self, manager=None) -> dict[str, Any]:
        """Flush, write the summary into run.json, and return it."""
        summary = self.summary(manager)
        self.metadata["summary"] = summary
        self.metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_metadata()
        for handle in (self._frames_file, self._events_file):
            if not handle.closed:
                handle.close()
        return summary

    def _write_metadata(self) -> None:
        with (self.run_dir / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, default=str)


def _wall_clock() -> dict[str, Any]:
    now = time.time()
    return {"time": datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3], "epoch_s": round(now, 3)}


def _round(value, digits):
    """None/NaN-tolerant round -- these columns are legitimately empty on
    frames with no detection or before the LSTM window fills, and an empty
    CSV cell reads back as NaN rather than as a fake 0.0."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else round(value, digits)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100.0
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
