"""Play back a force take with the force plots in sync, and suggest thresholds.

For a take recorded by force_logger.py (layout: src/force_take.py):
  1. replays the wrench through the same ScrewingMonitor the logger shows live
     (4_execution/force_monitors.py), restarting it at the hold_start / baseline markers;
  2. uses the markers as ground truth to measure |dF| while idle-holding, while
     screwing and after screw done, and suggests ScrewingThresholds from that;
  3. replays again with the suggested thresholds and reports how many pushes the
     monitor counts and when it would fire, against the screw_done marker;
  4. writes summary.png, summary.txt and summary.json into the take;
  5. unless --no-play, plays the video with the plots. The marker keys work here
     too, at the frame shown, so a take can be labeled or corrected afterwards.

Playback keys:
    space play/pause   , . one frame back/forward   [ ] 5 s back/forward
    h s e d r m b      add that marker at this frame   x delete the nearest marker
    a                  plot 2: change since hold <-> absolute force
    q or Esc           quit (changed markers are saved, the old ones to
                       markers.orig.csv, and the summary is written again)

Thresholds given on the command line are used as they are instead of suggested.

Usage:
    uv run python 4_execution/eval/force_log_review.py                  # newest take in logs/force
    uv run python 4_execution/eval/force_log_review.py logs/force/20260925_141500_screw01
    uv run python 4_execution/eval/force_log_review.py <take> --no-play
    uv run python 4_execution/eval/force_log_review.py <take> --active 10 --quiet 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

EXECUTION = Path(__file__).resolve().parents[1]  # 4_execution/
ROOT = EXECUTION.parent
for _path in (EXECUTION / "src", EXECUTION, ROOT):
    sys.path.insert(0, str(_path))

import cv2
import numpy as np

import force_take as take_io
from force_monitors import ScrewingMonitor, ScrewingThresholds
from force_take import AXIS_COLORS, INK, INK_2, clock

WINDOW = "force review"
EDGE_MARGIN_S = 0.3  # left out around each screw marker: a key is never pressed on time
SEEK_S = 5.0
NEAREST_MARKER_S = 2.0

# ScrewingThresholds field -> force_logger.py / this script's flag.
FLAGS = {
    "active_force_n": "--active",
    "quiet_force_n": "--quiet",
    "min_active_s": "--min-active",
    "quiet_s": "--quiet-seconds",
    "max_push_s": "--max-push",
}
PHASES = ("idle while holding", "screwing", "after screw done")


# -- analysis ---------------------------------------------------------------------

@dataclass
class Episode:
    """The monitor between two restarts."""
    start: float
    end: float
    reason: str                  # "take start", hold_start or baseline
    baseline_at: float | None
    pushes: list
    level_shifts: list
    done_at: float | None


@dataclass
class Replay:
    thresholds: ScrewingThresholds
    delta: np.ndarray            # (n, 6) smoothed wrench minus the current reference
    load: np.ndarray             # (n, 6) smoothed wrench minus the hold's baseline
    episodes: list[Episode]

    @property
    def holds(self) -> list[Episode]:
        """Episodes started by a hold or baseline marker; the whole take without any."""
        return [e for e in self.episodes if e.reason != "take start"] or self.episodes[:1]


@dataclass
class HoldLabels:
    """What the markers say about one hold."""
    start: float
    hold_from: float             # the monitor has a baseline from here
    hold_to: float               # release, or the end of the episode
    screws: list[tuple[float, float]]
    done: float | None


def replay(t, wrench, markers, thresholds: ScrewingThresholds, t_origin: float) -> Replay:
    monitor = ScrewingMonitor(thresholds)
    resets = [(mt, label) for mt, label in markers if label in take_io.MONITOR_RESETS]
    delta = np.full((len(t), 6), np.nan)
    load = np.full((len(t), 6), np.nan)
    episodes = []
    start, reason = t_origin, "take start"
    monitor.reset(start)

    def close(end):
        episodes.append(Episode(start, end, reason, monitor.baseline_at, list(monitor.pushes),
                                list(monitor.level_shifts), monitor.done_at))

    k = 0
    for i, now in enumerate(t):
        while k < len(resets) and resets[k][0] <= now:
            close(resets[k][0])
            start, reason = resets[k]
            monitor.reset(start, take_io.MONITOR_RESETS[reason])
            k += 1
        monitor.update(now, wrench[i])
        if monitor.delta is not None:
            delta[i] = monitor.delta
            load[i] = monitor.load
    end = max([t[-1] if len(t) else t_origin] + [mt for mt, _label in markers]) + 1e-3
    for k in range(k, len(resets)):  # restarts after the last sample
        close(resets[k][0])
        start, reason = resets[k]
        monitor.reset(start, take_io.MONITOR_RESETS[reason])
    close(end)
    return Replay(thresholds, delta, load, episodes)


def label_holds(markers, holds: list[Episode], thresholds: ScrewingThresholds) -> list[HoldLabels]:
    result = []
    for ep in holds:
        inside = [(mt, label) for mt, label in markers if ep.start <= mt < ep.end]
        hold_from = ep.baseline_at or ep.start + thresholds.settle_s + thresholds.baseline_s
        hold_to = next((mt for mt, label in inside if label == take_io.RELEASE), ep.end)
        done = next((mt for mt, label in inside if label == take_io.SCREW_DONE), None)
        screws, open_at = [], None
        for mt, label in inside:
            if label == take_io.SCREW_START:
                if open_at is not None:
                    screws.append((open_at, mt))
                open_at = mt
            elif label in (take_io.SCREW_END, take_io.SCREW_DONE, take_io.RELEASE) and open_at is not None:
                screws.append((open_at, mt))
                open_at = None
        if open_at is not None:
            screws.append((open_at, hold_to))
        result.append(HoldLabels(ep.start, hold_from, hold_to, screws, done))
    return result


def phase_masks(t: np.ndarray, holds: list[HoldLabels]) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(len(t), bool) for name in PHASES}
    for hold in holds:
        in_hold = (t >= hold.hold_from) & (t < hold.hold_to)
        screwing = np.zeros(len(t), bool)
        edge = np.zeros(len(t), bool)
        for start, end in hold.screws:
            screwing |= (t >= start) & (t <= end)
            edge |= (np.abs(t - start) < EDGE_MARGIN_S) | (np.abs(t - end) < EDGE_MARGIN_S)
        before_done = t < hold.done - EDGE_MARGIN_S if hold.done else np.ones(len(t), bool)
        masks["idle while holding"] |= in_hold & ~screwing & ~edge & before_done
        masks["screwing"] |= in_hold & screwing & ~edge
        if hold.done:
            masks["after screw done"] |= in_hold & ~screwing & (t >= hold.done + EDGE_MARGIN_S)
    return masks


def _stats(values: np.ndarray, dt: float) -> dict | None:
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"seconds": values.size * dt, "p50": p50, "p95": p95, "p99": p99, "max": values.max()}


def suggest(stats, holds: list[HoldLabels], base: ScrewingThresholds):
    """Thresholds from the labeled phases, and notes on what could not be tuned."""
    values, notes = {}, []
    idle, screwing = stats["force"]["idle while holding"], stats["force"]["screwing"]
    if idle and screwing and screwing["p50"] > idle["p99"]:
        active = idle["p99"] + 0.5 * (screwing["p50"] - idle["p99"])
        # Clear of idle noise, but not so close to it that a light touch breaks the quiet.
        quiet = min(max(1.2 * idle["p99"], 0.35 * active), 0.8 * active)
        values["active_force_n"], values["quiet_force_n"] = round(active, 1), round(quiet, 1)
        if quiet < idle["p99"]:
            notes.append("quiet is below the idle p99: idle noise will break quiet runs. "
                         "Smooth more, or check the signal.")
    elif idle and screwing:
        notes.append("screwing p50 |dF| is not above the idle p99: pushes cannot be told "
                     "from idle holding by |dF| alone. Look at the components and |dT|.")
    else:
        notes.append("mark idle holding and screwing (h, s, e) to tune active/quiet.")

    lengths = [end - start for hold in holds for start, end in hold.screws]
    totals = [sum(end - start for start, end in hold.screws) for hold in holds if hold.screws]
    gaps = [b[0] - a[1] for hold in holds for a, b in zip(hold.screws, hold.screws[1:])]
    if lengths:
        # A real push must never be taken for a level shift.
        values["max_push_s"] = round(max(3.0, 1.5 * max(lengths)), 1)
        # Half the marked screwing: the monitor only counts the pushing part of it.
        values["min_active_s"] = round(max(0.5, 0.5 * min(totals)), 1)
    if gaps:
        # Longer than any pause between two screws, or it fires between them.
        values["quiet_s"] = round(max(1.0, 1.2 * max(gaps)), 1)
    elif lengths:
        notes.append("only one screw per hold: quiet_s needs pauses between screws to tune.")
    return replace(base, **values), notes


class Review:
    def __init__(self, take: Path, overrides: dict):
        self.take = take
        meta_path = take / take_io.META
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.t, self.wrench = take_io.load_force(take)
        if not len(self.t):
            raise SystemExit(f"{take / take_io.FORCE_CSV} has no samples.")
        self.markers = take_io.load_markers(take)
        self.frame_times = take_io.load_frame_times(take)
        self.t_origin = float(self.meta.get("t_start", self.t[0]))
        self.dt = float(np.median(np.diff(self.t))) if len(self.t) > 1 else 0.0
        known = {f.name for f in fields(ScrewingThresholds)}
        self.live = ScrewingThresholds(**{k: v for k, v in self.meta.get("thresholds", {}).items()
                                          if k in known})
        self.overrides = {k: v for k, v in overrides.items() if v is not None}
        self.analyse()

    def analyse(self) -> None:
        base = replace(self.live, **self.overrides)
        self.measured = replay(self.t, self.wrench, self.markers, base, self.t_origin)
        self.holds = label_holds(self.markers, self.measured.holds, base)
        masks = phase_masks(self.t, self.holds)
        push = np.linalg.norm(self.measured.delta[:, :3], axis=1)
        torque = np.linalg.norm(self.measured.delta[:, 3:], axis=1)
        self.stats = {
            "force": {name: _stats(push[mask], self.dt) for name, mask in masks.items()},
            "torque": {name: _stats(torque[mask], self.dt) for name, mask in masks.items()},
        }
        suggested, self.notes = suggest(self.stats, self.holds, base)
        self.suggested = replace(suggested, **self.overrides)
        self.final = replay(self.t, self.wrench, self.markers, self.suggested, self.t_origin)
        self.loads = []
        for hold in self.holds:
            after = (self.t >= (hold.done or np.inf) + EDGE_MARGIN_S) & (self.t < hold.hold_to)
            load = np.nanmean(self.final.load[after, :3], axis=0) if after.any() else None
            self.loads.append(load if load is not None and np.all(np.isfinite(load)) else None)

    # -- report -------------------------------------------------------------------

    def rel(self, t: float) -> str:
        return clock(t - self.t_origin)

    def report(self) -> str:
        duration = self.t[-1] - self.t[0]
        counts = {}
        for _t, label in self.markers:
            counts[label] = counts.get(label, 0) + 1
        lines = [
            f"Take {self.take.name}: {clock(duration)} long, {len(self.t)} force samples "
            f"({1 / self.dt if self.dt else 0:.0f} Hz, {self.meta.get('source', '?')})"
            + ("" if self.frame_times is None else f", {len(self.frame_times)} video frames"),
            "Markers: " + (", ".join(f"{label} {n}" for label, n in counts.items()) or "none"),
            "",
        ]
        for key, unit in (("force", "|dF| (N)"), ("torque", "|dT| (Nm)")):
            lines.append(f"{unit:<22} {'time':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}")
            for name in PHASES:
                s = self.stats[key][name]
                lines.append(f"  {name:<20} " + ("   (not marked)" if s is None else
                             f"{s['seconds']:6.1f}s {s['p50']:7.2f} {s['p95']:7.2f} "
                             f"{s['p99']:7.2f} {s['max']:7.2f}"))
            lines.append("")

        for hold, load, ep, live_ep in zip(self.holds, self.loads, self.final.holds,
                                           self.measured.holds):
            lengths = [end - start for start, end in hold.screws]
            gaps = [b[0] - a[1] for a, b in zip(hold.screws, hold.screws[1:])]
            lines.append(f"Hold from {self.rel(hold.start)} to {self.rel(hold.hold_to)}:")
            if lengths:
                lines.append(f"  marked screws {len(lengths)}, {min(lengths):.1f}-{max(lengths):.1f} s "
                             f"each, {sum(lengths):.1f} s in total"
                             + (f"; pauses between them up to {max(gaps):.1f} s" if gaps else ""))
            if load is not None:
                lines.append(f"  load change after screw done: dF = ({load[0]:+.1f}, {load[1]:+.1f}, "
                             f"{load[2]:+.1f}) N, |dF| {np.linalg.norm(load):.1f} N "
                             f"(~{np.linalg.norm(load) / 9.81:.2f} kg)")
            live_name = "live + given" if self.overrides else "live"
            for name, episode in ((live_name, live_ep), ("suggested", ep)):
                lines.append(f"  {name + ' thresholds:':<26}{len(episode.pushes)} pushes counted, "
                             f"{len(episode.level_shifts)} level shifts; {self._verdict(hold, episode)}")
        lines.append("")

        live, final = asdict(self.live), asdict(self.suggested)
        lines.append("Thresholds" + (" (given ones kept)" if self.overrides else "")
                     + ":   suggested   (live)")
        for name, flag in FLAGS.items():
            mark = "  given" if name in self.overrides else ""
            lines.append(f"  {flag:<16} {final[name]:>8g}   ({live[name]:g}){mark}")
        lines.append("  force_logger.py " + " ".join(f"{flag} {final[name]:g}" for name, flag in FLAGS.items()))
        for note in self.notes:
            lines.append(f"Note: {note}")
        lines.append("Suggestions come from this take only: check them on other takes before trusting them.")
        return "\n".join(lines)

    def _verdict(self, hold: HoldLabels, ep: Episode) -> str:
        if ep.done_at is None:
            return "never fires"
        text = f"fires at {self.rel(ep.done_at)}"
        if hold.done is None:
            return text + " (no screw_done marker to compare with)"
        lag = ep.done_at - hold.done
        if lag < 0:
            return text + f" -- {-lag:.1f} s BEFORE the marked screw done (too early)"
        return text + f" -- {lag:.1f} s after the marked screw done"

    def write_summary(self) -> str:
        text = self.report()
        (self.take / "summary.txt").write_text(text + "\n", encoding="utf-8")
        summary = {
            "suggested": asdict(self.suggested),
            "live": asdict(self.live),
            "given": self.overrides,
            "stats": self.stats,
            "notes": self.notes,
            "holds": [{"start": h.start, "screws": h.screws, "done": h.done,
                       "load_change": None if load is None else load.tolist(),
                       "pushes": ep.pushes, "fires_at": ep.done_at}
                      for h, load, ep in zip(self.holds, self.loads, self.final.holds)],
        }
        (self.take / "summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                                encoding="utf-8")
        plot_summary(self, self.take / "summary.png")
        return text


# -- summary figure -----------------------------------------------------------------

# Light chart surface, inks and the first three categorical slots of the dataviz
# reference palette (light mode).
LIGHT_SURFACE, LIGHT_INK, LIGHT_INK_2, LIGHT_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LIGHT_AXES = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_summary(review: Review, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    t0 = review.t_origin
    x = review.t - t0
    final, th = review.final, review.suggested
    push = np.linalg.norm(final.delta[:, :3], axis=1)
    torque = np.linalg.norm(final.delta[:, 3:], axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.3]})
    fig.patch.set_facecolor(LIGHT_SURFACE)
    for ax in axes:
        ax.set_facecolor(LIGHT_SURFACE)
        ax.grid(color=LIGHT_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(LIGHT_INK_2)
        ax.tick_params(colors=LIGHT_INK_2, labelsize=8)
        for mt, label in review.markers:
            if label not in (take_io.SCREW_START, take_io.SCREW_END):
                ax.axvline(mt - t0, color=LIGHT_INK_2, lw=0.7, ls=":")

    ax = axes[0]
    for hold in review.holds:
        for start, end in hold.screws:
            ax.axvspan(start - t0, end - t0, color=LIGHT_INK_2, alpha=0.10, lw=0)
    ax.plot(x, push, color=LIGHT_INK, lw=1.0)
    top = max(np.nanmax(push) if np.isfinite(push).any() else 0.0, th.active_force_n) * 1.08
    bar_y = -0.04 * top
    for ep in final.holds:
        for start, end in ep.pushes:
            ax.plot([start - t0, end - t0], [bar_y, bar_y], color=LIGHT_AXES[0], lw=4,
                    solid_capstyle="round")
        if ep.done_at is not None:
            ax.axvline(ep.done_at - t0, color=LIGHT_INK, lw=1.2, ls="--")
            ax.text(ep.done_at - t0, top, " monitor fires", color=LIGHT_INK, fontsize=8, va="top")
    for value, name in ((th.active_force_n, "active"), (th.quiet_force_n, "quiet")):
        ax.axhline(value, color=LIGHT_INK_2, lw=0.9, ls="--")
        ax.text(x[-1], value, f"{name} {value:g} N ", color=LIGHT_INK_2, fontsize=8,
                ha="right", va="bottom")
    for mt, label in review.markers:
        if label not in (take_io.SCREW_START, take_io.SCREW_END):
            ax.text(mt - t0, top * 0.9, f" {label}", color=LIGHT_INK_2, fontsize=8, rotation=90, va="top")
    ax.set_ylim(2 * bar_y, top)
    ax.set_ylabel("|dF| push signal (N)", color=LIGHT_INK_2, fontsize=9)
    ax.legend(handles=[
        Line2D([], [], color=LIGHT_INK, lw=1.0, label="|dF| from the current reference"),
        Patch(color=LIGHT_INK_2, alpha=0.18, label="screwing (marked s -> e)"),
        Line2D([], [], color=LIGHT_AXES[0], lw=4, label="pushes the monitor counts"),
        Line2D([], [], color=LIGHT_INK, lw=1.2, ls="--", label="where the monitor fires"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=8, frameon=False,
        labelcolor=LIGHT_INK_2)

    ax = axes[1]
    for i, name in enumerate(("x", "y", "z")):
        ax.plot(x, final.load[:, i], color=LIGHT_AXES[i], lw=1.0, label=f"dF{name}")
    for ep in final.holds:
        for shift_t, _shift in ep.level_shifts:
            ax.axvline(shift_t - t0, color=LIGHT_INK_2, lw=0.9, ls="-.")
    ax.set_ylabel("dF since hold (N)", color=LIGHT_INK_2, fontsize=9)
    handles, _labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=LIGHT_INK_2, lw=0.9, ls="-.", label="level shift"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=8,
              frameon=False, labelcolor=LIGHT_INK_2)

    ax = axes[2]
    ax.plot(x, torque, color=LIGHT_INK, lw=1.0)
    ax.set_ylabel("|dT| (Nm)", color=LIGHT_INK_2, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: clock(value)))
    ax.set_xlabel("time since the take started (m:ss)", color=LIGHT_INK_2, fontsize=9)

    fig.suptitle(f"{review.take.name} -- with thresholds active {th.active_force_n:g} N, "
                 f"quiet {th.quiet_force_n:g} N, quiet for {th.quiet_s:g} s",
                 x=0.01, ha="left", color=LIGHT_INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=LIGHT_SURFACE)
    plt.close(fig)


# -- playback -----------------------------------------------------------------------

class Player:
    def __init__(self, review: Review, args):
        self.review = review
        self.args = args
        self.cap = cv2.VideoCapture(str(review.take / take_io.VIDEO))
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {review.take / take_io.VIDEO}")
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = review.frame_times
        self.n = min(len(times), count) if count > 0 else len(times)
        self.times = times[:self.n]
        self._next = 0
        self._seek_to: int | None = None
        self.changed = False
        self.absolute = False

    def _read(self, index: int):
        if index != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        self._next = index + 1
        return frame if ok else None

    def _on_trackbar(self, position: int) -> None:
        self._seek_to = position

    def run(self) -> None:
        if self.n == 0:
            print("The video has no frames.")
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.createTrackbar("frame", WINDOW, 0, max(1, self.n - 1), self._on_trackbar)
        index, playing = 0, False
        frame = self._read(0)
        while True:
            if frame is not None:
                cv2.imshow(WINDOW, self._render(frame, index, playing))
            if playing and index + 1 < self.n:
                delay = int(1000 * (self.times[index + 1] - self.times[index]))
                key = cv2.waitKey(max(1, delay))
            else:
                playing = False
                key = cv2.waitKey(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            target = index + 1 if playing else index
            if key >= 0:
                key &= 0xFF
                char = chr(key).lower()
                now = self.times[index]
                if key in (ord("q"), 27):
                    break
                if char == " ":
                    playing = not playing
                    target = index
                elif char in ",.":
                    playing = False
                    target = index + (1 if char == "." else -1)
                elif char in "[]":
                    step = SEEK_S if char == "]" else -SEEK_S
                    target = int(np.searchsorted(self.times, now + step))
                elif char in take_io.MARKER_KEYS:
                    self._add_marker(now, take_io.MARKER_KEYS[char])
                elif char == "x":
                    self._delete_marker(now)
                elif char == "a":
                    self.absolute = not self.absolute
            if self._seek_to is not None:
                if self._seek_to != index:
                    target, playing = self._seek_to, False
                self._seek_to = None
            target = int(np.clip(target, 0, self.n - 1))
            if target != index:
                index = target
                frame = self._read(index)
                cv2.setTrackbarPos("frame", WINDOW, index)
                self._seek_to = None
        cv2.destroyAllWindows()
        self.cap.release()

    def _add_marker(self, t: float, label: str) -> None:
        self.review.markers.append((t, label))
        self.review.markers.sort()
        self.changed = True
        print(f"+ {label} at {self.review.rel(t)}")
        if label in take_io.MONITOR_RESETS:
            print("  restarting the monitor there -- replaying ...")
            self.review.analyse()

    def _delete_marker(self, t: float) -> None:
        markers = self.review.markers
        if not markers:
            return
        nearest = min(range(len(markers)), key=lambda i: abs(markers[i][0] - t))
        if abs(markers[nearest][0] - t) > NEAREST_MARKER_S:
            print(f"No marker within {NEAREST_MARKER_S:g} s.")
            return
        mt, label = markers.pop(nearest)
        self.changed = True
        print(f"- {label} at {self.review.rel(mt)}")
        if label in take_io.MONITOR_RESETS:
            print("  replaying ...")
            self.review.analyse()

    def _render(self, frame, index: int, playing: bool) -> np.ndarray:
        review, args = self.review, self.args
        final, th = review.final, review.suggested
        now = self.times[index]
        width, height = args.plot_width, args.view_height
        heights = [int(height * 0.36), int(height * 0.28), int(height * 0.18)]
        heights.append(height - sum(heights))
        t_range = (now - 0.7 * args.plot_seconds, now + 0.3 * args.plot_seconds)
        common = {"t_origin": review.t_origin, "markers": review.markers, "cursor": now}
        at = min(int(np.searchsorted(review.t, now)), len(review.t) - 1)

        def value_at(values) -> str:
            value = values[at]
            return "--" if not np.isfinite(value) else f"{value:.2f}"

        push = np.linalg.norm(final.delta[:, :3], axis=1)
        p1 = take_io.render_panel(
            width, heights[0], review.t, [("", push, INK, None)], t_range,
            title=f"|dF| push signal (N)  {value_at(push)}",
            hlines=[(th.active_force_n, f"active {th.active_force_n:g}"),
                    (th.quiet_force_n, f"quiet {th.quiet_force_n:g}")], **common)
        if self.absolute:
            components, title = review.wrench[:, :3], "F absolute (N)"
        else:
            components, title = final.load[:, :3], "dF since hold (N)"
        p2 = take_io.render_panel(
            width, heights[1], review.t,
            [(name, components[:, i], AXIS_COLORS[i], components[at, i])
             for i, name in enumerate(("x", "y", "z"))],
            t_range, title=title, marker_labels=False, **common)
        torque = np.linalg.norm(final.delta[:, 3:], axis=1)
        p3 = take_io.render_panel(
            width, heights[2], review.t, [("", torque, INK, None)], t_range,
            title=f"|dT| (Nm)  {value_at(torque)}", min_span=0.5, marker_labels=False, **common)

        ep = next((e for e in reversed(final.episodes) if e.start <= now), final.episodes[0])
        pushes = sum(1 for _start, end in ep.pushes if end <= now)
        fired = ("fires at " + review.rel(ep.done_at)) if ep.done_at else "never fires"
        lines = [
            (f"{review.rel(now)}  frame {index + 1}/{self.n}  {'playing' if playing else 'paused'}"
             f"   {review.take.name}", INK),
            (f"monitor: pushes so far {pushes}/{len(ep.pushes)}, {fired}   thresholds: "
             f"active {th.active_force_n:g} quiet {th.quiet_force_n:g} "
             f"quiet-for {th.quiet_s:g} s", INK_2),
            ("space play  , . frame  [ ] 5 s  x delete marker  a view  q quit"
             + ("   (markers changed)" if self.changed else ""), INK_2),
            (take_io.KEY_HELP, INK_2),
        ]
        status = take_io.render_text(width, heights[3], lines)
        return take_io.side_by_side(take_io.fit_height(frame, height), [p1, p2, p3, status])


# -- main -------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", nargs="?", default=None,
                        help="Take folder, or a folder of takes for the newest (default logs/force).")
    parser.add_argument("--no-play", action="store_true", help="Only write the summary.")
    parser.add_argument("--view-height", type=int, default=600)
    parser.add_argument("--plot-width", type=int, default=760)
    parser.add_argument("--plot-seconds", type=float, default=15.0)
    given = parser.add_argument_group("thresholds to use as given instead of suggested")
    given.add_argument("--active", type=float, default=None, help="push starts, N")
    given.add_argument("--quiet", type=float, default=None, help="push ends / quiet, N")
    given.add_argument("--min-active", type=float, default=None, help="pushing needed, s")
    given.add_argument("--quiet-seconds", type=float, default=None, help="quiet before done, s")
    given.add_argument("--max-push", type=float, default=None, help="longer = level shift, s")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    take = take_io.find_take(args.take)
    review = Review(take, {
        "active_force_n": args.active, "quiet_force_n": args.quiet,
        "min_active_s": args.min_active, "quiet_s": args.quiet_seconds, "max_push_s": args.max_push,
    })
    print(review.write_summary())
    print(f"\nWrote summary.png, summary.txt and summary.json to {take}")

    if args.no_play:
        return
    if review.frame_times is None:
        print("This take has no video; nothing to play.")
        return
    player = Player(review, args)
    player.run()
    if player.changed:
        backup = take / "markers.orig.csv"
        if not backup.exists() and (take / take_io.MARKERS_CSV).exists():
            shutil.copyfile(take / take_io.MARKERS_CSV, backup)
        take_io.save_markers(take, review.markers)
        review.analyse()
        print("\nMarkers changed; summary written again:\n")
        print(review.write_summary())


if __name__ == "__main__":
    main()
