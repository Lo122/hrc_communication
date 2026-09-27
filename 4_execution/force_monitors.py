"""Signal-level monitors on the TCP wrench: what the force says, not what it means
for the task.

A task-level detector (2_decision_making) decides when to listen -- only while the
robot holds a panel, say -- and combines this with recognition progress. This module
only turns a stream of wrench samples into pushes, level shifts, quiet time and a
"done" moment.

ScrewingMonitor runs live in force_logger.py and on recorded takes in
force_log_review.py, so thresholds tuned on a take behave the same way online.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ScrewingThresholds:
    """Placeholders until tuned on recorded takes (force_log_review.py)."""

    active_force_n: float = 8.0   # |dF| at or above this: someone is pushing on the panel
    quiet_force_n: float = 3.0    # |dF| below this again: the push is over (hysteresis)
    min_push_s: float = 0.2       # shorter pushes are bumps, not a screw
    max_push_s: float = 8.0       # loud for longer than this: the level moved, not a push
    min_active_s: float = 2.0     # pushing time needed before screwing can be done
    quiet_s: float = 2.0          # quiet this long after the last push: screwing done
    smooth_s: float = 0.1         # trailing mean over this span before comparing
    settle_s: float = 0.5         # after a reset, wait this long before the baseline...
    baseline_s: float = 1.0       # ...then average this long for it


class ScrewingMonitor:
    """Screwing pushes and the quiet after them, from the change in TCP force.

    The baseline is the mean wrench over baseline_s, from the first sample at least
    settle_s after reset().
    Reset when the robot starts holding: the panel's own weight is then part of the
    baseline, and only what the human adds shows up as change.

    - A push starts when |dF| reaches active_force_n and ends when it drops below
      quiet_force_n (hysteresis); pushes shorter than min_push_s are ignored.
    - |dF| at or above quiet_force_n for longer than max_push_s is a level shift, not
      a push: the reference moves to the current level. That is what the panel's
      weight going over to the frame looks like. level_shifts keeps each one, and
      load is the change since the hold's own baseline.
    - done_at is set once: at least min_active_s of pushing has been seen, and |dF|
      has stayed below quiet_force_n for quiet_s since.

    Feed samples in time order with update(). Not thread-safe.
    """

    def __init__(self, thresholds: ScrewingThresholds | None = None):
        self.thresholds = thresholds or ScrewingThresholds()
        self.reset(None)

    def reset(self, t: float | None, settle_s: float | None = None) -> None:
        """Start over at time t: a new baseline, no pushes. With t None the clock
        starts at the next sample. settle_s overrides the thresholds' settle time
        (0 takes the baseline straight away)."""
        self._settle_s = self.thresholds.settle_s if settle_s is None else settle_s
        self._baseline_from = None if t is None else t + self._settle_s
        self._baseline_sum = np.zeros(6)
        self._baseline_n = 0
        self._baseline_started: float | None = None
        self.hold_baseline: np.ndarray | None = None  # the baseline taken after reset
        self.baseline: np.ndarray | None = None       # current reference, after shifts
        self.baseline_at: float | None = None

        self._window: deque[tuple[float, np.ndarray]] = deque()
        self._window_sum = np.zeros(6)
        self.delta: np.ndarray | None = None      # smoothed wrench minus baseline
        self.load: np.ndarray | None = None       # smoothed wrench minus hold_baseline
        self.delta_force: float | None = None     # |dF|, N
        self.delta_torque: float | None = None    # |dT|, Nm

        self.active = False
        self._push_start: float | None = None
        self.pushes: list[tuple[float, float]] = []
        self.level_shifts: list[tuple[float, np.ndarray]] = []
        self.active_s = 0.0
        self._loud_since: float | None = None
        self._last_loud: float | None = None
        self.done_at: float | None = None
        self.t: float | None = None

    @property
    def push_count(self) -> int:
        return len(self.pushes)

    def quiet_for(self) -> float:
        """Time since |dF| last reached quiet_force_n (0 before anything was loud)."""
        if self.t is None or self._last_loud is None:
            return 0.0
        return self.t - self._last_loud

    def update(self, t: float, wrench) -> None:
        th = self.thresholds
        wrench = np.asarray(wrench, dtype=float)
        self.t = t
        if self._baseline_from is None:
            self._baseline_from = t + self._settle_s

        if self.baseline is None:
            if t >= self._baseline_from:
                if self._baseline_n == 0:
                    # The window opens with the first sample, not at _baseline_from:
                    # samples that start late (a reader started mid-hold) still get
                    # averaged over baseline_s instead of one sample becoming the baseline.
                    self._baseline_started = t
                self._baseline_sum += wrench
                self._baseline_n += 1
                if t >= self._baseline_started + th.baseline_s:
                    self.baseline = self._baseline_sum / self._baseline_n
                    self.hold_baseline = self.baseline.copy()
                    self.baseline_at = t
            return

        self._window.append((t, wrench))
        self._window_sum += wrench
        while self._window[0][0] < t - th.smooth_s:
            _, old = self._window.popleft()
            self._window_sum -= old
        mean = self._window_sum / len(self._window)

        if np.linalg.norm((mean - self.baseline)[:3]) < th.quiet_force_n:
            self._loud_since = None
        else:
            self._last_loud = t
            if self._loud_since is None:
                self._loud_since = t
            elif t - self._loud_since > th.max_push_s:
                # Loud for longer than any push: the level itself has moved. Follow
                # it rather than count one endless push.
                self.level_shifts.append((t, mean - self.baseline))
                self.baseline = mean.copy()
                self.active, self._push_start, self._loud_since = False, None, None

        self.delta = mean - self.baseline
        self.load = mean - self.hold_baseline
        self.delta_force = float(np.linalg.norm(self.delta[:3]))
        self.delta_torque = float(np.linalg.norm(self.delta[3:]))

        force = self.delta_force
        if not self.active and force >= th.active_force_n:
            self.active, self._push_start = True, t
        elif self.active and force < th.quiet_force_n:
            self.active = False
            if t - self._push_start >= th.min_push_s:
                self.pushes.append((self._push_start, t))
                self.active_s += t - self._push_start

        if (self.done_at is None and not self.active and self.active_s >= th.min_active_s
                and self.quiet_for() >= th.quiet_s):
            self.done_at = t
