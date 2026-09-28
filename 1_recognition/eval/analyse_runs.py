"""Compare recognition runs logged by run_logger.py, and plot the result.

The question this exists to answer: when processing latency makes the model
miss frames (see RecognitionManager._read_frame_on_wall_clock), do its
PREDICTIONS change? Run the same recording twice -- once frame by frame, once
against the wall clock -- and compare:

    uv run python run_recognition.py --video-source "G:\\My Drive\\University of Stuttgart\\ITECH_Thesis\\Videos\\raw\\cam-06\\video__cam-06_uid-04_take-01.mp4" `
        --log-dir results --run-name baseline --no-display 

    uv run python run_recognition.py --video-source take01.mp4 \
        --realtime-playback --loop-hz 1000 \
        --log-dir results --run-name realtime_1x --no-display

    uv run python 1_recognition/eval/analyse_runs.py results --baseline baseline

Runs are aligned on the RECORDING's timeline (video_time_s), not on wall time
or on frame count -- that is the only axis on which "the same moment" means the
same thing in a run that saw every frame and one that saw a third of them.
Every comparison against the baseline is a step ("last known value") lookup at
the baseline's own frame times, which is exactly what a downstream consumer of
the live system would have seen.

Outputs, written to --out (default <input>/analysis):
    summary.csv         one row per run: coverage, latency, agreement
    triggers.csv        every emitted trigger, with its delta vs the baseline
    frame_rate.png      frames processed per second of recording
    latency.png         update() latency distribution (ECDF)
    step_timeline.png   predicted step over time, one band per run
    confidence.png      classifier confidence over time
    progress.png        progress head over time (what fires the triggers)
    idle.png            idle probability over time (multi-head models only)

The CSVs are the table-view twin of the plots: every number in a figure is
readable there too.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# This tool is otherwise pure pandas/matplotlib, but it shares the layer's logging setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")  # file output only -- these runs are usually headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


# Palette: the validated reference instance (light surface) -- categorical slots are
# assigned in fixed order, one per run, so a run keeps its colour across every figure
# here even when the set of runs plotted changes.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Step id is an ORDERED category (step 1 comes before step 2), so it gets the ordinal
# blue ramp rather than categorical hues. Starts at ramp step 250: on a light surface
# nothing lighter clears 2:1 against it.
STEP_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MAX_RUNS = len(SERIES)  # past 8 categorical slots, hues stop being distinguishable


@dataclass
class Run:
    name: str
    frames: pd.DataFrame
    events: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def uses_video_time(self) -> bool:
        """False for a live-camera run, which has no position in a recording
        -- those fall back to wall time and cannot be compared against a
        recorded baseline."""
        return self.frames["video_time_s"].notna().any()

    @property
    def time(self) -> pd.Series:
        return self.frames["video_time_s"] if self.uses_video_time else self.frames["wall_time_s"]

    @property
    def source_fps(self) -> float | None:
        """Recovered from the log itself: consecutive source frame indices and
        their video timestamps give the recording's own rate, without having to
        reopen the video."""
        frames = self.frames.dropna(subset=["frame_index", "video_time_s"])
        if len(frames) < 2:
            return None
        span_t = frames["video_time_s"].iloc[-1] - frames["video_time_s"].iloc[0]
        span_i = frames["frame_index"].iloc[-1] - frames["frame_index"].iloc[0]
        return float(span_i / span_t) if span_t > 0 else None


def load_run(run_dir: Path) -> Run | None:
    frames_path = run_dir / "frames.csv"
    if not frames_path.exists():
        return None
    frames = pd.read_csv(frames_path)
    if frames.empty:
        logger.warning("Skipping %s: frames.csv has no rows", run_dir.name)
        return None

    events_path = run_dir / "events.csv"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()

    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return Run(name=run_dir.name, frames=frames, events=events, meta=meta)


def discover_runs(inputs: list[Path]) -> list[Run]:
    """Each input is either a run directory or a parent holding several."""
    run_dirs: list[Path] = []
    for path in inputs:
        if (path / "frames.csv").exists():
            run_dirs.append(path)
        else:
            run_dirs.extend(sorted(child for child in path.iterdir()
                                   if child.is_dir() and (child / "frames.csv").exists()))

    runs = [run for run in (load_run(d) for d in dict.fromkeys(run_dirs)) if run is not None]
    if len(runs) > MAX_RUNS:
        logger.warning("%d runs found; plotting the first %d (past that, series colours "
                       "stop being distinguishable). Pass the runs you want explicitly "
                       "to choose.", len(runs), MAX_RUNS)
        runs = runs[:MAX_RUNS]
    return runs


# -- comparison ------------------------------------------------------------

def step_lookup(run: Run, column: str, at_times: np.ndarray) -> np.ndarray:
    """Value of `column` that `run` would have been showing at each of
    `at_times` -- last known value, held until the next frame it processed.
    This is the honest way to compare a run that saw every frame against one
    that saw a third of them: a consumer asking at time t gets whatever the
    most recent frame produced, not an interpolation of frames that were
    never processed."""
    valid = run.frames.dropna(subset=["video_time_s", column])
    if valid.empty:
        return np.full(len(at_times), np.nan)
    times = valid["video_time_s"].to_numpy()
    values = valid[column].to_numpy()
    # searchsorted gives the insertion point; -1 makes it the most recent frame at or
    # before each query time. Before the run's first frame there is nothing to report.
    idx = np.searchsorted(times, at_times, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)
    return out


def compare_to_baseline(run: Run, baseline: Run) -> dict:
    """Fraction of the baseline's frames at which `run` was showing the same
    step, and how far its progress estimate sat from the baseline's."""
    if run.name == baseline.name or not (run.uses_video_time and baseline.uses_video_time):
        return {}

    reference = baseline.frames.dropna(subset=["video_time_s", "stable_step_id"])
    if reference.empty:
        return {}
    at_times = reference["video_time_s"].to_numpy()

    result = {}
    theirs = step_lookup(run, "stable_step_id", at_times)
    ours = reference["stable_step_id"].to_numpy()
    both = ~np.isnan(theirs)
    if both.any():
        result["stable_step_agreement"] = round(float(np.mean(theirs[both] == ours[both])), 4)

    ref_raw = baseline.frames.dropna(subset=["video_time_s", "raw_step_id"])
    if not ref_raw.empty:
        raw_times = ref_raw["video_time_s"].to_numpy()
        theirs_raw = step_lookup(run, "raw_step_id", raw_times)
        valid = ~np.isnan(theirs_raw)
        if valid.any():
            result["raw_step_agreement"] = round(
                float(np.mean(theirs_raw[valid] == ref_raw["raw_step_id"].to_numpy()[valid])), 4)

    ref_progress = baseline.frames.dropna(subset=["video_time_s", "progress"])
    if not ref_progress.empty:
        prog_times = ref_progress["video_time_s"].to_numpy()
        theirs_prog = step_lookup(run, "progress", prog_times)
        valid = ~np.isnan(theirs_prog)
        if valid.any():
            delta = theirs_prog[valid] - ref_progress["progress"].to_numpy()[valid]
            result["progress_mae"] = round(float(np.mean(np.abs(delta))), 4)
    return result


def summarise(runs: list[Run], baseline: Run | None) -> pd.DataFrame:
    rows = []
    for run in runs:
        frames = run.frames
        dropped = float(frames["dropped_before"].fillna(0).sum())
        seen = len(frames)
        span = float(run.time.iloc[-1] - run.time.iloc[0]) if len(frames) > 1 else 0.0

        row = {
            "run": run.name,
            "frames_processed": seen,
            "frames_dropped": int(dropped),
            "frame_coverage": round(seen / (seen + dropped), 4) if seen + dropped else None,
            "video_span_s": round(span, 2),
            "effective_fps": round(seen / span, 2) if span > 0 else None,
            "source_fps": round(run.source_fps, 2) if run.source_fps else None,
            "update_ms_mean": round(float(frames["update_ms"].mean()), 2),
            "update_ms_p95": round(float(frames["update_ms"].quantile(0.95)), 2),
            "detection_rate": round(float(frames["detected"].mean()), 4),
            "mean_confidence": _safe_mean(frames["confidence"]),
            "triggers": int(len(run.events)),
            "realtime_playback": run.meta.get("realtime_playback"),
            "playback_speed": run.meta.get("playback_speed"),
        }
        if baseline is not None:
            row.update(compare_to_baseline(run, baseline))
        rows.append(row)
    return pd.DataFrame(rows)


def trigger_table(runs: list[Run], baseline: Run | None) -> pd.DataFrame:
    """Every emitted trigger on the recording's timeline. The delta column is
    what a downstream consumer feels: how much earlier or later the same
    trigger fired than it did in the baseline run."""
    baseline_times: dict = {}
    if baseline is not None and not baseline.events.empty:
        for _, event in baseline.events.iterrows():
            baseline_times.setdefault((event.get("step_id"), event.get("round_id")),
                                      event.get("video_time_s"))

    rows = []
    for run in runs:
        for _, event in run.events.iterrows():
            key = (event.get("step_id"), event.get("round_id"))
            reference = baseline_times.get(key)
            fired = event.get("video_time_s")
            rows.append({
                "run": run.name,
                "event_type": event.get("event_type"),
                "round_id": event.get("round_id"),
                "step_id": event.get("step_id"),
                "video_time_s": fired,
                "baseline_video_time_s": reference,
                "delta_s": round(float(fired - reference), 3)
                if pd.notna(fired) and pd.notna(reference) else None,
            })
    return pd.DataFrame(rows)


def _safe_mean(series: pd.Series):
    value = series.mean()
    return None if pd.isna(value) else round(float(value), 4)


# -- plotting --------------------------------------------------------------

def style_axes(ax, *, xlabel: str, ylabel: str, title: str, subtitle: str | None = None):
    """Recessive chrome: hairline solid grid one shade off the surface, no top/
    right spines, muted tick labels. Never dashed -- dashing reads as
    'projection' when it is only a grid."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=16 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY,
                fontsize=9, va="bottom")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def new_figure(size=(10, 5)):
    fig, ax = plt.subplots(figsize=size, dpi=150, facecolor=SURFACE)
    return fig, ax


def finish(fig, ax, path: Path, *, legend: bool):
    # A legend is always present for >= 2 series; a single series needs none -- the
    # title names it. Identity is never carried by colour alone.
    if legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)


def plot_frame_rate(runs: list[Run], out_dir: Path) -> None:
    """How much of the recording each run actually got to look at, second by
    second. The reference hairline is the recording's own frame rate -- the
    gap below it IS the latency-induced frame loss."""
    fig, ax = new_figure()
    source_fps = next((run.source_fps for run in runs if run.source_fps), None)

    for run, color in zip(runs, SERIES):
        frames = run.frames.dropna(subset=["video_time_s"])
        if frames.empty:
            continue
        seconds = np.floor(frames["video_time_s"]).astype(int)
        counts = seconds.value_counts().sort_index()
        ax.plot(counts.index, counts.to_numpy(), drawstyle="steps-post",
                color=color, linewidth=1.8, label=run.name)

    if source_fps:
        ax.axhline(source_fps, color=MUTED, linewidth=0.8)
        ax.text(ax.get_xlim()[1], source_fps, f" recording: {source_fps:.0f} fps",
                color=MUTED, fontsize=9, va="center", ha="left")

    ax.set_ylim(bottom=0)
    style_axes(ax, xlabel="Time in recording (s)", ylabel="Frames processed per second",
               title="Frames the model actually saw",
               subtitle="Per second of the recording; below the hairline is frame loss to processing latency")
    finish(fig, ax, out_dir / "frame_rate.png", legend=len(runs) >= 2)


def plot_latency(runs: list[Run], out_dir: Path) -> None:
    """ECDF rather than overlaid histograms: with several runs, histogram bars
    occlude each other, while cumulative curves stay readable and make the
    tail -- the part that causes the drops -- directly comparable."""
    fig, ax = new_figure()
    for row, (run, color) in enumerate(zip(runs, SERIES)):
        values = np.sort(run.frames["update_ms"].dropna().to_numpy())
        if values.size == 0:
            continue
        cumulative = np.arange(1, values.size + 1) / values.size
        ax.plot(values, cumulative, color=color, linewidth=1.8, label=run.name)
        p95 = float(np.quantile(values, 0.95))
        # Direct-label the p95 only -- selective labelling, never a number per point.
        ax.plot([p95], [0.95], marker="o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2)
        # Runs with similar latency land their p95 markers on top of each other, so the
        # labels are stacked by run rather than all hung off their own marker.
        ax.annotate(f"{run.name}: p95 {p95:.0f} ms", (p95, 0.95), textcoords="offset points",
                    xytext=(10, -14 * row - 6), color=INK_SECONDARY, fontsize=9)

    ax.set_ylim(0, 1.02)
    style_axes(ax, xlabel="update() latency (ms)", ylabel="Fraction of frames",
               title="Processing latency per frame",
               subtitle="Cumulative distribution of the full recognition step (pose, lift, features, LSTM)")
    finish(fig, ax, out_dir / "latency.png", legend=len(runs) >= 2)


def plot_step_timeline(runs: list[Run], out_dir: Path) -> None:
    """One band per run, coloured by the stabilized step. This is the figure
    that answers the question: if two bands differ, frame loss changed the
    prediction -- and where it changed is legible directly."""
    fig, ax = plt.subplots(figsize=(10, 1.1 * len(runs) + 2.2), dpi=150, facecolor=SURFACE)

    present = sorted({int(value) for run in runs
                      for value in run.frames["stable_step_id"].dropna().unique()})
    if not present:
        plt.close(fig)
        logger.warning("Skipping step_timeline.png: no stabilized step predictions logged")
        return
    # Spread the steps evenly across the ordinal ramp, keeping step order = ramp order.
    color_for = {step: STEP_RAMP[round(i * (len(STEP_RAMP) - 1) / max(len(present) - 1, 1))]
                 for i, step in enumerate(present)}
    if len(present) > 7:
        logger.info("%d distinct steps -- past ~7 classes adjacent ramp steps blur "
                    "together; read exact values from frames.csv.", len(present))

    height = 0.62
    for row, run in enumerate(runs):
        frames = run.frames.dropna(subset=["stable_step_id"])
        y = len(runs) - row - 1
        if frames.empty:
            continue
        times = run.frames["video_time_s"] if run.uses_video_time else run.frames["wall_time_s"]
        times = times.loc[frames.index].to_numpy()
        steps = frames["stable_step_id"].to_numpy().astype(int)

        # Collapse consecutive identical predictions into one band, so the figure shows
        # state changes rather than one rectangle per frame.
        boundaries = np.flatnonzero(np.diff(steps)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(steps)]))
        for start, end in zip(starts, ends):
            t0 = times[start]
            t1 = times[end] if end < len(times) else times[-1]
            width = max(t1 - t0, 1e-6)
            # 2px surface gap between adjacent fills -- a separating ring, not a border.
            ax.broken_barh([(t0, width)], (y - height / 2, height),
                           facecolors=color_for[steps[start]], edgecolor=SURFACE, linewidth=1.0)

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([run.name for run in reversed(runs)], color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(-0.6, len(runs) - 0.4)
    style_axes(ax, xlabel="Time in recording (s)", ylabel="",
               title="Predicted step over time",
               subtitle="Stabilized step id per run; differences between rows are predictions changed by frame loss")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=color_for[s], edgecolor=SURFACE, label=f"step {s}")
                       for s in present],
              frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
              ncol=min(len(present), 6), loc="upper center",
              bbox_to_anchor=(0.5, -0.28 if len(runs) > 2 else -0.45))
    fig.tight_layout()
    fig.savefig(out_dir / "step_timeline.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote step_timeline.png")


def plot_series(runs: list[Run], column: str, out_dir: Path, *,
                filename: str, title: str, subtitle: str, ylabel: str) -> None:
    fig, ax = new_figure()
    plotted = 0
    for run, color in zip(runs, SERIES):
        frames = run.frames.dropna(subset=[column])
        if frames.empty:
            continue
        times = (run.frames["video_time_s"] if run.uses_video_time
                 else run.frames["wall_time_s"]).loc[frames.index]
        ax.plot(times.to_numpy(), frames[column].to_numpy(), color=color,
                linewidth=1.6, label=run.name)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("Skipping %s: no %s values logged", filename, column)
        return

    style_axes(ax, xlabel="Time in recording (s)", ylabel=ylabel, title=title, subtitle=subtitle)
    finish(fig, ax, out_dir / filename, legend=plotted >= 2)


# -- entry point -----------------------------------------------------------

def main() -> int:
    configure_logging("analyse_runs")
    parser = argparse.ArgumentParser(
        description="Compare recognition runs logged with run_recognition.py --log-dir.")
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Run directories, or a parent directory holding several.")
    parser.add_argument("--baseline", default=None,
                        help="Name of the run every other run is compared against -- normally "
                             "the frame-by-frame run (no --realtime-playback), i.e. the model "
                             "with no frames missing. Defaults to the first run found.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for the CSVs and figures. "
                             "Default: <first input>/analysis.")
    args = parser.parse_args()

    runs = discover_runs(args.inputs)
    if not runs:
        logger.error("No runs found (looked for directories containing frames.csv).")
        return 1

    baseline = next((run for run in runs if run.name == args.baseline), None)
    if args.baseline and baseline is None:
        logger.error("Baseline %r not found among: %s", args.baseline,
                     ", ".join(r.name for r in runs))
        return 1
    if baseline is None:
        baseline = runs[0]
    # The baseline leads, so it takes categorical slot 1 in every figure.
    runs = [baseline] + [run for run in runs if run is not baseline]

    out_dir = args.out or (args.inputs[0] / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Runs: %s  (baseline: %s)", ", ".join(run.name for run in runs), baseline.name)

    summary = summarise(runs, baseline)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n" + summary.to_string(index=False) + "\n")

    triggers = trigger_table(runs, baseline)
    if not triggers.empty:
        triggers.to_csv(out_dir / "triggers.csv", index=False)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(triggers.to_string(index=False) + "\n")
    else:
        print("No trigger events logged.\n")

    plot_frame_rate(runs, out_dir)
    plot_latency(runs, out_dir)
    plot_step_timeline(runs, out_dir)
    plot_series(runs, "confidence", out_dir, filename="confidence.png",
                title="Classifier confidence over time",
                subtitle="Max step probability per processed frame",
                ylabel="Confidence")
    plot_series(runs, "progress", out_dir, filename="progress.png",
                title="Progress estimate over time",
                subtitle="The regression head that crosses the thresholds firing the triggers",
                ylabel="Progress")
    # Multi-head models only; runs logged before the column existed lack it.
    idle_runs = [run for run in runs if "idle_prob" in run.frames]
    if idle_runs:
        plot_series(idle_runs, "idle_prob", out_dir, filename="idle.png",
                    title="Idle probability over time",
                    subtitle="Background head: P(nobody is working); idle wins above 0.5",
                    ylabel="P(idle)")

    logger.info("Analysis written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
