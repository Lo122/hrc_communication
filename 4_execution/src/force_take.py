"""One force take: the TCP wrench, camera frames and hand-set markers of a trial.

force_logger.py records a take and force_log_review.py reads it back. Layout of
logs/force/<YYYYmmdd_HHMMSS>[_<name>]/:

  force.csv             one row per wrench sample, FORCE_COLUMNS
  video.mp4             raw camera frames (no plot drawn on them)
  video.timestamps.csv  frame, timestamp_s -- when each frame was captured
  markers.csv           t_wall, label -- one row per marker key pressed
  meta.json             how the take was recorded
  logger.log            diagnostics
  summary.png / .txt    written by force_log_review.py

Every time is time.time() on the recording machine, so samples, frames and markers
share one clock and line up without an offset. t_robot is the controller's clock
(RTDE) or the message stamp (ROS), kept only to check for gaps.

The wrench is actual_TCP_force: base frame, at the TCP, with the payload set on the
controller subtracted -- a held panel's weight is still in it unless that payload
includes the panel.

Also holds the OpenCV drawing both tools share, so the live view and the playback
look the same: render_panel() for a time-series panel, render_text() for status lines.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAKES_DIR = ROOT / "logs" / "force"

FORCE_CSV = "force.csv"
MARKERS_CSV = "markers.csv"
VIDEO = "video.mp4"
META = "meta.json"

WRENCH = ("fx", "fy", "fz", "tx", "ty", "tz")
POSE = ("x", "y", "z", "rx", "ry", "rz")
RAW_WRENCH = tuple(f"raw_{name}" for name in WRENCH)
FORCE_COLUMNS = ("t_wall", "t_robot", *WRENCH, *POSE, *RAW_WRENCH)

HOLD_START = "hold_start"
SCREW_START = "screw_start"
SCREW_END = "screw_end"
SCREW_DONE = "screw_done"
RELEASE = "release"
BASELINE = "baseline"

# Keys that set a marker, in the logger and in the review alike.
MARKER_KEYS = {
    "h": HOLD_START,   # robot starts holding the panel; the baseline is taken after it
    "s": SCREW_START,  # screwdriver on a screw
    "e": SCREW_END,    # that screw is in
    "d": SCREW_DONE,   # all screws in -- where a detector should fire
    "r": RELEASE,      # robot lets go of the panel
    "m": "mark",       # anything else worth finding again
    "b": BASELINE,     # re-take the baseline without a hold (bench tests)
}
# Markers that restart the ScrewingMonitor, and the settle time they restart it with
# (None: the thresholds' own). A hold waits for the robot to settle; "b" does not.
MONITOR_RESETS = {HOLD_START: None, BASELINE: 0.0}

KEY_HELP = "h hold  s/e screw start/end  d screw done  r release  m mark  b baseline"


# -- take files ---------------------------------------------------------------

def new_take_dir(root: str | Path = DEFAULT_TAKES_DIR, name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / (f"{stamp}_{name}" if name else stamp)
    path.mkdir(parents=True, exist_ok=False)
    return path


def find_take(path: str | Path | None = None) -> Path:
    """The take at path; for a folder of takes (default logs/force), the newest one."""
    path = Path(path) if path else DEFAULT_TAKES_DIR
    if (path / FORCE_CSV).exists():
        return path
    takes = sorted(p for p in path.iterdir() if (p / FORCE_CSV).exists()) if path.is_dir() else []
    if not takes:
        raise FileNotFoundError(f"No force take (a folder with {FORCE_CSV}) at {path}.")
    return takes[-1]  # names start with the recording time


def load_force(take: Path) -> tuple[np.ndarray, np.ndarray]:
    """(t_wall, wrench (n, 6)) in time order. A row cut off by a crash is skipped."""
    times, wrenches = [], []
    with open(take / FORCE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                wrench = [float(row[name]) for name in WRENCH]
                t = float(row["t_wall"])
            except (TypeError, ValueError):
                continue
            times.append(t)
            wrenches.append(wrench)
    t = np.asarray(times)
    order = np.argsort(t, kind="stable")
    return t[order], np.asarray(wrenches, dtype=float).reshape(-1, 6)[order]


def load_markers(take: Path) -> list[tuple[float, str]]:
    path = take / MARKERS_CSV
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return sorted((float(row["t_wall"]), row["label"]) for row in csv.DictReader(f))


def save_markers(take: Path, markers: list[tuple[float, str]]) -> None:
    with open(take / MARKERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_wall", "label"])
        for t, label in sorted(markers):
            writer.writerow([f"{t:.6f}", label])


def load_frame_times(take: Path) -> np.ndarray | None:
    """Wall-clock capture time of each frame of video.mp4, or None without a video."""
    path = (take / VIDEO).with_suffix(".timestamps.csv")
    if not (take / VIDEO).exists() or not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return np.asarray([float(row["timestamp_s"]) for row in csv.DictReader(f)])


def clock(seconds: float) -> str:
    """m:ss for a time relative to the take's start."""
    seconds = max(0.0, round(seconds, 3))  # 2.0 stored as 1.9999999 still reads 0:02
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


# -- drawing ------------------------------------------------------------------

def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# Dark chart surface, inks, and the first three categorical slots of the dataviz
# reference palette (dark mode), as BGR for OpenCV.
SURFACE = _bgr("#1a1a19")
GRID = _bgr("#383835")
INK = _bgr("#ffffff")      # primary text, and the |dF| / |dT| lines
INK_2 = _bgr("#c3c2b7")    # secondary text, axes, reference lines, markers
AXIS_COLORS = (_bgr("#3987e5"), _bgr("#d95926"), _bgr("#199e70"))  # x, y, z

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42


def put_text(img, text, origin, color=INK_2, scale=FONT_SCALE) -> int:
    """Draw text; returns its width in pixels."""
    cv2.putText(img, text, (int(origin[0]), int(origin[1])), FONT, scale, color, 1, cv2.LINE_AA)
    return cv2.getTextSize(text, FONT, scale, 1)[0][0]


def _nice_step(span: float, ticks: int) -> float:
    raw = max(span, 1e-9) / ticks
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if raw <= factor * power:
            return factor * power
    return 10 * power


def _runs(mask: np.ndarray):
    """(start, stop) of each run of True."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _polyline(xs: np.ndarray, ys: np.ndarray, plot_w: int) -> np.ndarray:
    """Pixel points; more than a few per column become that column's min and max,
    so short spikes -- the pushes this is about -- never drop out of the line."""
    if len(xs) > 4 * plot_w:
        columns = xs.astype(int)
        starts = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
        low, high = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)
        xs = np.repeat(columns[starts], 2)
        ys = np.column_stack([low, high]).ravel()
    return np.column_stack([xs, ys]).round().astype(np.int32)


def render_panel(width: int, height: int, t: np.ndarray, series, t_range, *, title: str,
                 t_origin: float, hlines=(), markers=(), marker_labels: bool = True,
                 cursor: float | None = None, min_span: float = 1.0) -> np.ndarray:
    """One time-series panel on the dark surface.

    series: (label, values, color, current) per line -- values aligned with t, NaN
        breaks the line; current is shown after the label in the legend (or None).
        Later series are drawn on top.
    t_range: (t0, t1) shown. t_origin: the time labeled 0:00 on the x axis.
    hlines: (value, label) reference lines. markers: (t, label) vertical lines,
        labeled only with marker_labels (one labeled panel per view is enough).
    """
    img = np.full((height, width, 3), SURFACE, np.uint8)
    left, right, top, bottom = 46, 10, 26, 20
    plot_w, plot_h = width - left - right, height - top - bottom
    t0, t1 = t_range
    visible = (t >= t0) & (t <= t1)

    shown = [values[visible] for _label, values, _color, _current in series]
    data = np.concatenate([v[np.isfinite(v)] for v in shown]) if shown else np.zeros(0)
    lo = min(0.0, float(data.min())) if data.size else 0.0
    hi = max(0.0, float(data.max())) if data.size else 0.0
    for value, _label in hlines:
        lo, hi = min(lo, value), max(hi, value)
    if hi - lo < min_span:
        hi = lo + min_span
    pad = 0.08 * (hi - lo)
    hi += pad
    lo -= pad if lo < 0 else 0.0

    def x_of(values):
        return left + (values - t0) / max(t1 - t0, 1e-9) * plot_w

    def y_of(values):
        return top + (hi - values) / (hi - lo) * plot_h

    step = _nice_step(hi - lo, 4)
    decimals = max(0, -math.floor(math.log10(step)))
    for value in np.arange(math.ceil(lo / step), math.floor(hi / step) + 1) * step:
        y = int(round(y_of(value)))
        cv2.line(img, (left, y), (left + plot_w, y), GRID, 1)
        put_text(img, f"{value:.{decimals}f}", (4, y + 4))

    t_step = _nice_step(t1 - t0, 5)
    first = math.ceil((t0 - t_origin) / t_step)
    for k in range(first, first + 12):
        tick = t_origin + k * t_step
        if tick > t1:
            break
        x = int(round(x_of(tick)))
        cv2.line(img, (x, top), (x, top + plot_h), GRID, 1)
        put_text(img, clock(tick - t_origin), (x - 12, height - 5))

    for value, label in hlines:
        y = int(round(y_of(value)))
        for x in range(left, left + plot_w, 10):
            cv2.line(img, (x, y), (min(x + 5, left + plot_w), y), INK_2, 1)
        width_px = cv2.getTextSize(label, FONT, FONT_SCALE, 1)[0][0]
        put_text(img, label, (left + plot_w - width_px - 2, y - 4))

    in_range = [(mt, label) for mt, label in markers if t0 <= mt <= t1]
    for mt, _label in in_range:
        x = int(round(x_of(mt)))
        cv2.line(img, (x, top), (x, top + plot_h), INK_2, 1)

    for (_label, values, color, _current), vis in zip(series, shown):
        tv = t[visible]
        for start, stop in _runs(np.isfinite(vis)):
            if stop - start < 2:
                continue
            points = _polyline(x_of(tv[start:stop]), y_of(vis[start:stop]), plot_w)
            cv2.polylines(img, [points], False, color, 2, cv2.LINE_AA)

    if cursor is not None and t0 <= cursor <= t1:
        x = int(round(x_of(cursor)))
        cv2.line(img, (x, top), (x, top + plot_h), INK, 1)

    if marker_labels:
        # On a surface-colored box, over the lines, so a label stays readable on a spike.
        for i, (mt, label) in enumerate(in_range):
            x, y = int(round(x_of(mt))) + 3, top + 12 + 14 * (i % 3)
            (text_w, text_h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + 3), SURFACE, -1)
            put_text(img, label, (x, y))

    x = left + put_text(img, title, (left, 17), INK) + 16
    for label, _values, color, current in series:
        if not label:
            continue
        text = label if current is None or not np.isfinite(current) else f"{label} {current:+.2f}"
        text_w = cv2.getTextSize(text, FONT, FONT_SCALE, 1)[0][0]
        if x + 18 + text_w > width - 4:
            break
        cv2.line(img, (x, 13), (x + 12, 13), color, 2, cv2.LINE_AA)
        x += 18 + put_text(img, text, (x + 18, 17)) + 14
    return img


def render_text(width: int, height: int, lines) -> np.ndarray:
    """Status lines, (text, color) each, on the dark surface."""
    img = np.full((height, width, 3), SURFACE, np.uint8)
    for i, (text, color) in enumerate(lines):
        put_text(img, text, (8, 18 + 19 * i), color)
    return img


def fit_height(frame: np.ndarray, height: int) -> np.ndarray:
    scale = height / frame.shape[0]
    return cv2.resize(frame, (max(1, int(round(frame.shape[1] * scale))), height),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def side_by_side(frame: np.ndarray | None, panels: list[np.ndarray]) -> np.ndarray:
    """Camera frame on the left (already fit to the panels' total height), panels
    stacked on the right."""
    right = np.vstack(panels)
    return right if frame is None else np.hstack([frame, right])
