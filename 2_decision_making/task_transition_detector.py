"""Detectors: task transitions read from signals, not from recognition's task guess.

Recognition's step output says which task the human is probably on, and TaskManager
does not ask it at all while a robot lift leads. A detector reports that something
specific has happened, from a signal it can read reliably: the shape of recognition's
Screw progress over one screw, or the TCP force while the robot holds a panel. Each
report is a Signal on one task of one piece:

    Signal("Screw", 1, "progress", 0.5)            how far the task is, in place of
                                                   recognition's progress
    Signal("Screw", 1, "panel secured", True)      the panel is fixed to the frame: the
                                                   robot may release it and leave
    Signal("Screw", 1, "screw count", 0.5)         anything else: a value the task
    Signal("Screw", 1, "TCP weight change", True)  database's trigger rules can test

Trigger rules test signals under "Condition" (src/task_database.py). TaskManager gets
each signal as a TASK_SIGNAL event (_handle_task_signal): it stores it on the tracked
task and offers what the rules now allow. "panel secured" while the robot holds that
panel ends the holding and asks to release it and leave -- without confirming Screw:
recognition, or the human saying "screw done", says when Screw is done. A "Done
signal" from a detector would confirm its task like the human does.

TaskTransitionDetectors runs the detectors in order once per main-loop pass
(HRCSystem.process_events) with a DetectorContext: the robot task and its state, the
held piece, the task tracker, what arrived since the last pass (recognition updates,
wrench samples), and what the detectors have reported so far -- so a later detector
can build on an earlier one's signals whatever TASK_DETECTORS_MODE says.
config.TASK_DETECTORS_MODE decides whether their signals count or are only logged.

Adding a detector:
  1. subclass Detector, set name, and implement update(context): return the signals
     that are new since the last call -- each change once, not on every pass;
  2. add it in build_detectors(), after any detector whose signals it reads;
  3. test what it reports under "Condition" in the task database.
update() runs on the main loop, so it must not block. A sensor with its own thread
hands samples over through DetectorContext, the way ROSCommunication buffers the wrench.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# src/ holds this layer's helpers (task database, transition table, trigger policy).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config
from events import Event, EventType, RobotTaskState, TaskStatus
from force_monitors import ScrewingMonitor, ScrewingThresholds
from task_database import DONE_SIGNAL, PANEL_SECURED, PROGRESS_SIGNAL

SCREW = "Screw"
SCREW_COUNT = "screw count"
TCP_WEIGHT_CHANGE = "TCP weight change"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    task_name: str
    piece_id: int
    name: str                   # PROGRESS_SIGNAL, PANEL_SECURED, or a signal rules can test
    value: float | bool = True
    source: str = ""            # the detector's name
    details: dict = field(default_factory=dict, compare=False)  # for the log only


@dataclass
class DetectorContext:
    """What the detectors see on one pass. Read it; never change it."""

    now: float
    active_task: object | None     # models.RobotTask
    held_piece_id: int | None      # the panel the robot holds, if any
    tracker: object | None         # task_tracker.TaskTracker
    wrench: list = field(default_factory=list)       # (t, [fx, fy, fz, tx, ty, tz]) since the last pass
    recognition: list = field(default_factory=list)  # (t, task name, progress) since the last pass
    # Latest value each detector has reported, by (task, piece, signal); kept by
    # TaskTransitionDetectors, whether or not TaskManager acted on it.
    reported: dict = field(default_factory=dict)

    @property
    def holding(self) -> bool:
        """The robot holds a panel for the human (R_HOLDING)."""
        return (self.active_task is not None and self.held_piece_id is not None
                and self.active_task.state == RobotTaskState.R_HOLDING)

    def task(self, task_name: str, piece_id: int):
        """The tracked task, or None (also without a tracker)."""
        return None if self.tracker is None else self.tracker.get(task_name, piece_id)


class Detector:
    """One detector."""

    name = "detector"

    def update(self, context: DetectorContext) -> list[Signal]:
        """The signals that are new since the last call."""
        raise NotImplementedError

    def status(self) -> dict:
        """What the detector sees right now, for the live view (decision_view.py)."""
        return {}

    def signal(self, task_name: str, piece_id: int, name: str, value=True, **details) -> Signal:
        return Signal(task_name, piece_id, name, value, self.name, details)


class ScrewCountDetector(Detector):
    """Screws counted from the shape of recognition's Screw progress.

    Recognition's progress for Screw runs per screw: it climbs to about 0.7 while one
    screw goes in and drops back near 0 as the next one starts. A climb to high then a
    fall to low (hysteresis, so jitter in between counts nothing) is one screw -- and so
    is recognition moving on from Screw after a climb, since the last screw has no next
    one to drop into. Counts closer together than min_interval_s are taken as one.

    Counts on the panel the robot holds, else the piece the human is on, and starts over
    when that piece changes; stops once that piece's Screw is done. It reports there:
      "screw count"    screws counted / screws_per_panel (at most 1), per screw
      "progress"       the same, as Screw's progress (recognition's own runs per screw)
      "panel secured"  once every screw is counted
    """

    name = "screw count"

    def __init__(self, *, screws_per_panel: int, high: float, low: float, min_interval_s: float):
        self.screws_per_panel = max(1, int(screws_per_panel))
        self.high, self.low, self.min_interval_s = high, low, min_interval_s
        self._piece: int | None = None
        self._start_over(None)

    def _start_over(self, piece_id: int | None) -> None:
        self._piece = piece_id
        self._count = 0
        self._under_way = False
        self._last_counted_at: float | None = None
        self._progress: float | None = None
        self._secured_reported = False

    def status(self) -> dict:
        return {
            "piece": self._piece,
            "screw progress now": None if self._progress is None else round(self._progress, 2),
            "screw under way": self._under_way,
            "screws counted": f"{self._count} of {self.screws_per_panel}",
            "panel secured reported": self._secured_reported,
        }

    def update(self, context: DetectorContext) -> list[Signal]:
        piece = context.held_piece_id
        if piece is None and context.tracker is not None:
            piece = context.tracker.human_piece_id
        if piece != self._piece:
            self._start_over(piece)
        screw = context.task(SCREW, piece) if piece is not None else None
        if piece is None or (screw is not None and screw.status == TaskStatus.DONE):
            self._under_way = False
            return []
        signals = []
        for t, task_name, progress in context.recognition:
            if task_name != SCREW:
                if self._under_way:
                    signals += self._count_one(t, f"recognition moved on to {task_name}")
                self._progress = None
                continue
            self._progress = progress
            if progress >= self.high:
                self._under_way = True
            elif self._under_way and progress <= self.low:
                signals += self._count_one(t, "progress fell back")
        return signals

    def _count_one(self, t: float, why: str) -> list[Signal]:
        self._under_way = False
        if self._last_counted_at is not None and t - self._last_counted_at < self.min_interval_s:
            return []
        if self._count >= self.screws_per_panel:
            return []
        self._count += 1
        self._last_counted_at = t
        counted = self._count / self.screws_per_panel
        signals = [self.signal(SCREW, self._piece, SCREW_COUNT, counted, screws=self._count, why=why),
                   self.signal(SCREW, self._piece, PROGRESS_SIGNAL, counted)]
        if self._count == self.screws_per_panel and not self._secured_reported:
            self._secured_reported = True
            signals.append(self.signal(SCREW, self._piece, PANEL_SECURED, True, screws=self._count))
        return signals


class ForceScrewDetector(Detector):
    """Screwing, from the TCP force while the robot holds a panel.

    Listens only while the robot holds a panel (R_HOLDING, after a lift or a leave
    canceled back to holding): the force on the TCP then comes from the human screwing
    into it. ScrewingMonitor restarts at every new hold, so the panel's own weight is in
    its baseline. On that piece's Screw it reports:
      "TCP weight change"  once, when the load has moved by weight_change_n since the
                           hold began and stayed there (a level shift)
      "panel secured"      once, when either
                           - the monitor says the pushing is over and Screw is at least
                             min_progress along (None: no minimum): the screw count
                             reported by ScrewCountDetector, or the tracker's own
                             progress for Screw if further on; or
                           - whatever recognition counted: the load change since the
                             hold began has stayed within weight_range_n, with no push,
                             for weight_steady_s in a row -- the frame carries the panel
    """

    name = "force screw"

    def __init__(self, thresholds: ScrewingThresholds, *, min_progress: float | None,
                 weight_change_n: float, weight_range_n: tuple[float, float] | None = None,
                 weight_steady_s: float = 10.0):
        """weight_range_n: (low, high) in N, or None for no done signal from the weight."""
        self.monitor = ScrewingMonitor(thresholds)
        self.min_progress = min_progress
        self.weight_change_n = weight_change_n
        self.weight_range_n = weight_range_n
        self.weight_steady_s = weight_steady_s
        self._hold: str | None = None
        self._piece: int | None = None
        self._secured_reported = False
        self._in_range_since: float | None = None

    def status(self) -> dict:
        monitor = self.monitor
        if self._hold is None:
            return {"listening": False}
        return {
            "listening": True,
            "piece": self._piece,
            "baseline": "taken" if monitor.baseline is not None else "taking",
            "force change (N)": None if monitor.delta_force is None else round(monitor.delta_force, 2),
            "pushing now": monitor.active,
            "pushes": monitor.push_count,
            "pushing in total (s)": round(monitor.active_s, 1),
            "quiet for (s)": round(monitor.quiet_for(), 1),
            "level shifts": len(monitor.level_shifts),
            "load change since hold (N)": (None if monitor.load is None
                                           else round(float(np.linalg.norm(monitor.load[:3])), 2)),
            "load in weight range for (s)": round(self._steady_for(), 1),
            "pushing over": monitor.done_at is not None,
            "panel secured reported": self._secured_reported,
        }

    def update(self, context: DetectorContext) -> list[Signal]:
        if not context.holding:
            self._hold = None
            return []
        hold = context.active_task.task_instance_id
        if hold != self._hold:
            self._hold, self._piece, self._since = hold, context.held_piece_id, context.now
            self.monitor.reset(context.now)
            self._weight_reported, self._secured_reported = False, False
            self._in_range_since = None
        for t, wrench in context.wrench:
            if t >= self._since:
                self.monitor.update(t, wrench)
                self._track_weight(t)

        monitor, signals = self.monitor, []
        if (not self._weight_reported and monitor.level_shifts and not monitor.active
                and monitor.load is not None
                and np.linalg.norm(monitor.load[:3]) >= self.weight_change_n):
            self._weight_reported = True
            signals.append(self.signal(SCREW, self._piece, TCP_WEIGHT_CHANGE, True,
                                       load_change_n=[round(float(v), 2) for v in monitor.load[:3]]))
        if not self._secured_reported and self._screw_open(context):
            if monitor.done_at is not None and self._far_enough(context):
                self._secured_reported = True
                signals.append(self.signal(SCREW, self._piece, PANEL_SECURED, True,
                                           because="pushing over",
                                           pushes=monitor.push_count,
                                           pushing_s=round(monitor.active_s, 2),
                                           quiet_s=round(monitor.quiet_for(), 2)))
            elif self.weight_range_n is not None and self._steady_for() >= self.weight_steady_s:
                self._secured_reported = True
                signals.append(self.signal(SCREW, self._piece, PANEL_SECURED, True,
                                           because="panel weight steady on the frame",
                                           load_change_n=round(float(np.linalg.norm(monitor.load[:3])), 2),
                                           steady_s=round(self._steady_for(), 2)))
        return signals

    def _track_weight(self, t: float) -> None:
        """Keep when the load change entered weight_range_n; a push or leaving the
        range starts the wait over."""
        load, monitor = self.monitor.load, self.monitor
        inside = (self.weight_range_n is not None and load is not None and not monitor.active
                  and self.weight_range_n[0] <= np.linalg.norm(load[:3]) <= self.weight_range_n[1])
        if not inside:
            self._in_range_since = None
        elif self._in_range_since is None:
            self._in_range_since = t

    def _steady_for(self) -> float:
        """How long the load change has stayed in weight_range_n, with no push (s)."""
        if self._in_range_since is None or self.monitor.t is None:
            return 0.0
        return self.monitor.t - self._in_range_since

    def _screw_open(self, context: DetectorContext) -> bool:
        """That piece's Screw is not done yet: once it is, the robot has been asked to
        leave already ("screw done" while holding)."""
        screw = context.task(SCREW, self._piece)
        return screw is None or screw.status != TaskStatus.DONE

    def _far_enough(self, context: DetectorContext) -> bool:
        """Screw is at least min_progress along."""
        if self.min_progress is None:
            return True
        screw = context.task(SCREW, self._piece)
        counted = context.reported.get((SCREW, self._piece, SCREW_COUNT), 0.0)
        tracked = screw.progress if screw is not None and screw.status == TaskStatus.WORKING else 0.0
        return max(counted, tracked) >= self.min_progress


class TaskTransitionDetectors:
    """Runs the detectors in order and collects what they report."""

    def __init__(self, detectors: list[Detector]):
        self.detectors = list(detectors)
        self.reported: dict[tuple, object] = {}
        self._failed: set[str] = set()

    def update(self, context: DetectorContext) -> list[Signal]:
        context.reported = self.reported
        signals = []
        for detector in self.detectors:
            try:
                new = detector.update(context)
            except Exception:
                # A broken detector must not take the robot's dialogue down with it.
                if detector.name not in self._failed:
                    self._failed.add(detector.name)
                    _logger.exception("Task detector %r failed; its signals are missing "
                                      "until it recovers.", detector.name)
                continue
            for signal in new:
                self.reported[(signal.task_name, signal.piece_id, signal.name)] = signal.value
            signals += new
        return signals


def build_detectors() -> TaskTransitionDetectors:
    """The detectors config.py sets up, in the order they run."""
    return TaskTransitionDetectors([
        ScrewCountDetector(screws_per_panel=config.SCREWS_PER_PANEL,
                           high=config.SCREW_PROGRESS_HIGH, low=config.SCREW_PROGRESS_LOW,
                           min_interval_s=config.SCREW_MIN_INTERVAL_S),
        ForceScrewDetector(ScrewingThresholds(**config.SCREWING_THRESHOLDS),
                           min_progress=config.SCREW_DONE_MIN_PROGRESS,
                           weight_change_n=config.TCP_WEIGHT_CHANGE_N,
                           weight_range_n=config.TCP_WEIGHT_RANGE_N,
                           weight_steady_s=config.TCP_WEIGHT_STEADY_S),
    ])


def signal_payload(signal: Signal) -> dict:
    payload = {"task_name": signal.task_name, "piece_id": signal.piece_id,
               "signal": signal.name, "value": signal.value}
    if signal.details:
        payload["details"] = signal.details
    return payload


def signal_event(signal: Signal) -> Event:
    return Event(EventType.TASK_SIGNAL, f"detector:{signal.source}", payload=signal_payload(signal))
