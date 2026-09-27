"""Plots of the UR sensor data: the log written by eval/read_ur_live_data.py, and
a rolling live force/torque view (LiveWrenchPlot, eval/ur_state_reader.py --plot).
Both share one look, so a live view and a plot made afterwards read the same.

read_ur_live_data.py writes one line per sample to
4_execution/logs/ur_live_data_<YYYYmmdd_HHMMSS>.log:

    2026-09-25 20:46:01.770, joint angle: [-1.592, ...] rad, tcp position: (-0.1623, +0.6448, +0.5118) m, force: (-0.13, -0.72, -15.15) N, torque: (+0.08, -0.12, -0.06) Nm

load_sensor_log() parses that back into arrays (_LINE_RE must stay in step
with the f-string in read_ur_live_data.py's main loop); plot_sensor_log()
draws it as four stacked panels sharing one time axis -- joint angles, TCP
position, force, torque. One panel per unit, so no panel needs a second
y-axis.

Usage (with no LOG, plots the newest ur_live_data_*.log in 4_execution/logs/;
the PNG is written beside the log unless --out says otherwise):
    uv run python 4_execution/src/plot_utils.py [LOG] [--out PNG] [--show]
"""

from __future__ import annotations

import argparse
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Same directory as read_ur_live_data.py's LOG_DIR; not imported from there
# because that module needs ur_rtde, which plotting shouldn't.
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}), "
    r"joint angle: \[(?P<joint_angles>[^\]]*)\] rad, "
    r"tcp position: \((?P<tcp_position>[^)]*)\) m, "
    r"force: \((?P<force>[^)]*)\) N, "
    r"torque: \((?P<torque>[^)]*)\) Nm$"
)

# Consecutive samples further apart than this (s) are drawn with a break in the
# line rather than a straight segment across the gap -- e.g. two runs appended
# to the same --log-file.
MAX_GAP_S = 1.0

# --- look ---------------------------------------------------------------------
# Categorical line colors, always taken in this order: x/y/z (and Fx/Fy/Fz,
# Tx/Ty/Tz) use the first three in every panel, so an axis keeps its color
# from panel to panel. The order keeps neighboring hues distinguishable under
# color-vision deficiency; don't shuffle it.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LINE_WIDTH = 1.5  # pt, ~2px at screen resolution
_LINE_STYLE = {"linewidth": LINE_WIDTH, "solid_joinstyle": "round", "solid_capstyle": "round"}

# (SensorLog field, y-axis label with unit, one legend label per column)
PANELS = (
    (
        "joint_angles",
        "Joint angle (rad)",
        ("shoulder pan", "shoulder lift", "elbow", "wrist 1", "wrist 2", "wrist 3"),
    ),
    ("tcp_position", "TCP position (m)", ("x", "y", "z")),
    ("force", "Force (N)", ("Fx", "Fy", "Fz")),
    ("torque", "Torque (Nm)", ("Tx", "Ty", "Tz")),
)


@dataclass
class SensorLog:
    """One sensor log, one row per logged sample."""

    timestamps: np.ndarray  # (N,) Unix time, s
    joint_angles: np.ndarray  # (N, 6) rad
    tcp_position: np.ndarray  # (N, 3) m, base frame
    force: np.ndarray  # (N, 3) N
    torque: np.ndarray  # (N, 3) Nm

    def __len__(self) -> int:
        return len(self.timestamps)


def load_sensor_log(path: str | Path) -> SensorLog:
    """Parse a read_ur_live_data.py log. Lines that don't match the format
    (e.g. the last one, cut short by Ctrl+C mid-write) are skipped and
    counted, not fatal."""
    columns: dict[str, list] = {name: [] for name in _LINE_RE.groupindex}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = _LINE_RE.match(line)
            if match is None:
                skipped += 1
                continue
            columns["timestamp"].append(
                datetime.strptime(match["timestamp"], TIMESTAMP_FORMAT).timestamp()
            )
            for name in ("joint_angles", "tcp_position", "force", "torque"):
                columns[name].append([float(v) for v in match[name].split(",")])

    if skipped:
        print(f"[plot] skipped {skipped} unparseable line(s) in {path}")
    if not columns["timestamp"]:
        raise ValueError(f"no sensor samples in {path}")
    return SensorLog(
        timestamps=np.array(columns["timestamp"]),
        joint_angles=np.array(columns["joint_angles"]),
        tcp_position=np.array(columns["tcp_position"]),
        force=np.array(columns["force"]),
        torque=np.array(columns["torque"]),
    )


def plot_sensor_log(
    log: SensorLog,
    *,
    title: str | None = None,
    max_gap: float = MAX_GAP_S,
) -> Figure:
    """Joint angles, TCP position, force and torque over time, as four
    stacked panels on a shared time axis (seconds since the first sample)."""
    t0 = log.timestamps[0]
    fig, axes = plt.subplots(
        len(PANELS), 1, sharex=True, figsize=(11, 9), layout="constrained"
    )
    fig.set_facecolor(SURFACE)

    for ax, (field, ylabel, labels) in zip(axes, PANELS):
        t, values = _break_gaps(log.timestamps - t0, getattr(log, field), max_gap)
        for column, (label, color) in enumerate(zip(labels, SERIES_COLORS)):
            ax.plot(t, values[:, column], color=color, label=label, **_LINE_STYLE)
        _style_axes(ax, ylabel)
        _legend(ax)

    axes[-1].set_xlabel(
        f"Time since {datetime.fromtimestamp(t0):%H:%M:%S} (s)", color=INK_SECONDARY
    )
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    return fig


class LiveWrenchPlot:
    """Rolling live plot of the TCP force and torque over the last `history_s`
    seconds, drawn like the force/torque panels of plot_sensor_log().

    Each update() redraws and pumps the GUI event loop (plt.pause), which takes
    a few ms -- call it at a decimated rate, not once per RTDE sample.
    """

    def __init__(self, history_s: float = 10.0):
        self.history_s = history_s
        self._t0: float | None = None
        # (seconds since the first update, wrench) -- trimmed by time, not count,
        # so the window stays history_s long whatever rate update() is called at.
        self._samples: deque[tuple[float, np.ndarray]] = deque()

        panels = [panel for panel in PANELS if panel[0] in ("force", "torque")]
        plt.ion()
        self.fig, self._axes = plt.subplots(
            len(panels), 1, sharex=True, figsize=(9, 6), layout="constrained"
        )
        self.fig.set_facecolor(SURFACE)
        self._lines = []
        for ax, (_, ylabel, labels) in zip(self._axes, panels):
            self._lines.append(
                [
                    ax.plot([], [], color=color, label=label, **_LINE_STYLE)[0]
                    for label, color in zip(labels, SERIES_COLORS)
                ]
            )
            _style_axes(ax, ylabel)
            _legend(ax)
        self._axes[-1].set_xlabel("Time (s)", color=INK_SECONDARY)

    def update(self, wrench, timestamp: float | None = None) -> None:
        """Add one [Fx, Fy, Fz, Tx, Ty, Tz] sample (N, Nm) and redraw."""
        now = time.time() if timestamp is None else timestamp
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        self._samples.append((t, np.asarray(wrench[:6], dtype=float)))
        while self._samples[0][0] < t - self.history_s:
            self._samples.popleft()

        times = np.array([sample_t for sample_t, _ in self._samples])
        wrenches = np.array([w for _, w in self._samples])
        for panel, (ax, lines) in enumerate(zip(self._axes, self._lines)):
            for column, line in enumerate(lines):
                line.set_data(times, wrenches[:, 3 * panel + column])
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        plt.close(self.fig)


def _legend(ax: Axes) -> None:
    """Legend outside the plot on the right, so it never covers data."""
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=1.5,
    )


def _break_gaps(
    t: np.ndarray, values: np.ndarray, max_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN row wherever consecutive samples are more than `max_gap`
    apart; matplotlib leaves a break in the line at NaNs."""
    gaps = np.flatnonzero(np.diff(t) > max_gap) + 1
    if gaps.size == 0:
        return t, values
    return np.insert(t, gaps, np.nan), np.insert(values, gaps, np.nan, axis=0)


def _style_axes(ax: Axes, ylabel: str) -> None:
    """Recessive chrome: hairline horizontal grid, no top/right spines, muted
    tick labels -- the data lines are the only strong color on the figure."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0)  # time axis runs exactly first -> last sample
    ax.tick_params(color=AXIS, labelcolor=INK_MUTED, labelsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)


def _newest_log() -> Path:
    # The filename's timestamp sorts chronologically as text.
    logs = sorted(LOG_DIR.glob("ur_live_data_*.log"))
    if not logs:
        raise SystemExit(f"no ur_live_data_*.log in {LOG_DIR}; pass a log path")
    return logs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=None,
        help="sensor log to plot (default: the newest ur_live_data_*.log in "
        "4_execution/logs/)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="PNG to write (default: LOG with .png)"
    )
    parser.add_argument(
        "--show", action="store_true", help="also open the plot in a window"
    )
    args = parser.parse_args()

    path = args.log or _newest_log()
    log = load_sensor_log(path)
    duration = log.timestamps[-1] - log.timestamps[0]
    start = datetime.fromtimestamp(log.timestamps[0])
    fig = plot_sensor_log(
        log,
        title=f"{path.name}  ·  {start:%Y-%m-%d %H:%M:%S}  ·  "
        f"{duration:.1f} s, {len(log)} samples",
    )

    out = args.out or path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
