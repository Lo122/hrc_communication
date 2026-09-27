"""Event and transition logging.

Every record is one JSON line, stamped with when it was written: "t" (epoch
seconds, the same clock the recognition process stamps its events and frames
with) and "time" (local clock, readable). An event also carries "event_t", when
its sender created it -- for recognition's events, in the recognition process.

With timeline_path, the same records also go to a readable CSV without the
human-location stream: when recognition saw an action ("recognized_at", the frame
its result came from), when the communication layer got it ("time", and
"delay_ms" in between), and what the layer did about it (the rows that follow).
"""

import csv
import json
import threading
import time
from datetime import datetime

from events import Event
from models import RobotTask

# Too frequent (and too bulky) to read in the timeline; the JSON lines keep them.
TIMELINE_SKIPPED_EVENTS = {"HUMAN_LOCATION_UPDATE"}
TIMELINE_COLUMNS = ["time", "t", "kind", "what", "task_instance_id", "source", "detail",
                    "recognized_at", "delay_ms"]
EPOCH_2001 = 1e9


def clock(t: float, date: bool = False) -> str:
    """Local time with milliseconds."""
    text = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f" if date else "%H:%M:%S.%f")
    return text[:-3]


class EventLogger:
    """Records incoming events, transitions, and integration actions."""

    def __init__(self, file_path, timeline_path=None):
        self.file_path = file_path
        self.timeline_path = timeline_path
        # The voice and CLI threads log too.
        self._lock = threading.Lock()
        self._file = open(file_path, "a", encoding="utf-8", buffering=1)
        self._timeline_file = self._timeline = None
        if timeline_path is not None:
            self._timeline_file = open(timeline_path, "w", encoding="utf-8", newline="")
            self._timeline = csv.DictWriter(self._timeline_file, fieldnames=TIMELINE_COLUMNS)
            self._timeline.writeheader()

    def _write(self, record: dict, row: dict | None = None) -> None:
        now = time.time()
        line = json.dumps({"t": round(now, 3), "time": clock(now, date=True), **record},
                          default=str, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            if self._timeline is not None and row is not None:
                self._timeline.writerow({"time": clock(now), "t": round(now, 3), **row})
                self._timeline_file.flush()

    def log_event(self, event: Event) -> None:
        """Record every incoming event before it is handled."""
        name = event.event_type.name
        row = None
        if name not in TIMELINE_SKIPPED_EVENTS:
            row = {"kind": "event", "what": name, "task_instance_id": event.task_instance_id,
                   "source": event.source, "detail": _event_detail(event)}
            seen = event.payload.get("timestamp") if name == "HUMAN_TASK_UPDATE" else None
            # A live source stamps its frames with the epoch; a recorded video with the
            # position in the video, which is no time of day.
            if isinstance(seen, (int, float)) and seen > EPOCH_2001:
                row["recognized_at"] = clock(seen)
                row["delay_ms"] = round((time.time() - seen) * 1000.0, 1)
        self._write(
            {
                "type": "event",
                "event_type": name,
                "event_t": round(event.timestamp, 3),
                "source": event.source,
                "task_instance_id": event.task_instance_id,
                "payload": event.payload,
            },
            row,
        )

    def log_transition(
        self,
        task: RobotTask,
        event: Event,
        old_state,
        new_state,
        message: str | None = None,
    ) -> None:
        """Record a state transition performed by TaskManager."""
        self._write(
            {
                "type": "transition",
                "task_instance_id": task.task_instance_id,
                "old_state": old_state.name,
                "new_state": new_state.name,
                "event_type": event.event_type.name,
                "message": message,
            },
            {"kind": "transition", "what": f"{old_state.name} -> {new_state.name}",
             "task_instance_id": task.task_instance_id, "source": event.event_type.name,
             "detail": message},
        )

    def log_message(self, message: str, context: dict | None = None) -> None:
        """Record non-transition integration messages such as ignored events."""
        context = context or {}
        self._write(
            {"type": "log", "message": message, "context": context},
            {"kind": "log", "what": message,
             "detail": json.dumps(context, default=str, separators=(",", ":")) if context else None},
        )

    def close(self) -> None:
        with self._lock:
            for handle in (self._file, self._timeline_file):
                if handle is not None and not handle.closed:
                    handle.close()


def _event_detail(event: Event) -> str | None:
    payload = event.payload
    if event.event_type.name == "HUMAN_TASK_UPDATE":
        name = payload.get("task_name") or f"step {payload.get('step_id')}"
        progress = payload.get("progress")
        return name if not isinstance(progress, (int, float)) else f"{name} {progress:.2f}"
    return json.dumps(payload, default=str, separators=(",", ":")) if payload else None
