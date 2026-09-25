""" analyse the task sequence from the recorded videos (G:\\My Drive\\University of Stuttgart\\ITECH_Thesis\\Videos\\annotations\\ceiling_installation)

    the order of tasks performed by the human
    and possibly the robot, to understand the interaction patterns and task dependencies.
    build probability models to determine what kind of task can be considered next based on the observed sequences.
        -> this would compensate for errors in task recognition (filter the recogition model outputs)
    calculate action average time, std, median -> to understand the typical duration of each task
        -> in real scenarios, this works as safe gaurd of our system to ask human what they are doing now.
        (if recognition model keeps outputting incorrect tasks, the system can use the average action time to detect anomalies and prompt the human for clarification)

Implementation notes:
    - Annotation entries are mapped from ELAN step_ids to the recognition model's task_ids
      with LSTM_HRC/data_proc_3d/src/skeleton_pipeline/dataset/labels.py (to_task_entries +
      _trim_lift_before_overlap), so the tables use the same taxonomy the model outputs.
      That also means labels.EXCLUDE_STEPS applies (T Pose and, currently, Lift / M - Lift).
    - "M - X" mistake tiers fold into their task X; "No Related Task" is never synthesised.
    - The order within a take is by start_frame. Each label file is its own sequence
      (split files like take-03 / take-03-1 are separate rounds, not continuations).
    - Consecutive entries of the same task (e.g. six Screw spans in a row) are merged into
      one "block" for the task-order table; the raw table keeps them as self-transitions.

Outputs (CSV + PNG) go to 2_decision_making/task_sequence_lift/ by default (--out-dir to change;
the committed tables live under 2_decision_making/results/):
    task_sequences.csv                 every task entry per take, in order
    transition_counts[_raw].csv        START/task -> task/END counts
    transition_probabilities[_raw].csv same, row-normalised (P(next | current))
    duration_stats_entry.csv           per annotation span: mean/std/median/... in seconds
    duration_stats_block.csv           per merged block of the same task
    *.png                              heatmaps and duration box plots

Usage:
    uv run python 2_decision_making/src/task_sequence_analysis.py [--annotations-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS_SRC = ROOT.parent / "LSTM_HRC" / "data_proc_3d" / "src"
sys.path.insert(0, str(LABELS_SRC))
from skeleton_pipeline.dataset import labels  # noqa: E402

ANNOTATIONS_DIR = Path(r"G:\My Drive\University of Stuttgart\ITECH_Thesis\Videos\annotations\ceiling_installation")
OUT_DIR = ROOT / "2_decision_making" / "task_sequence_lift"

START_STATE = "START"
END_STATE = "END"
DEFAULT_FPS = 30.0


def load_take(label_path: Path) -> pd.DataFrame:
    """One label__*.json -> its task entries (model taxonomy), sorted by start_frame."""
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta_data", {})
    ms_per_sample = meta.get("ms_per_sample")
    fps = 1000.0 / ms_per_sample if ms_per_sample else DEFAULT_FPS

    task_entries = labels._trim_lift_before_overlap(labels.to_task_entries(data.get("labels", [])))
    task_entries = [entry for entry in task_entries if entry["task_id"] != labels.NO_TASK_ID]
    if not task_entries:
        return pd.DataFrame()

    take = pd.DataFrame(task_entries)
    take = take.sort_values(["start_frame", "end_frame"], kind="stable").reset_index(drop=True)
    take.insert(0, "take", label_path.stem.removeprefix("label__"))
    take["user_id"] = meta.get("user_id")
    take["order"] = np.arange(len(take))
    take["task_name"] = take["task_id"].map(labels.TASK_NAMES)
    take["start_sec"] = take["start_frame"] / fps
    take["end_sec"] = take["end_frame"] / fps
    take["duration_sec"] = (take["end_frame"] - take["start_frame"]) / fps
    # A new block starts whenever the task changes from the previous entry.
    take["block"] = (take["task_id"] != take["task_id"].shift()).cumsum() - 1
    return take[["take", "user_id", "order", "block", "task_id", "task_name", "piece_id",
                 "is_mistake", "start_frame", "end_frame", "start_sec", "end_sec", "duration_sec"]]


def load_all(annotations_dir: Path) -> pd.DataFrame:
    label_paths = sorted(annotations_dir.glob("label__*.json"))
    if not label_paths:
        raise FileNotFoundError(f"No label__*.json files in {annotations_dir}")
    takes = [load_take(path) for path in label_paths]
    return pd.concat([take for take in takes if not take.empty], ignore_index=True)


def to_blocks(entries: pd.DataFrame) -> pd.DataFrame:
    """Merges consecutive entries of the same task within a take into one block."""
    blocks = entries.groupby(["take", "block"], sort=False).agg(
        task_id=("task_id", "first"),
        task_name=("task_name", "first"),
        n_entries=("order", "size"),
        has_mistake=("is_mistake", "any"),
        start_sec=("start_sec", "min"),
        end_sec=("end_sec", "max"),
    ).reset_index()
    blocks["duration_sec"] = blocks["end_sec"] - blocks["start_sec"]
    return blocks


def transition_counts(sequences: pd.DataFrame) -> pd.DataFrame:
    """START/task -> task/END transition counts over each take's task_id sequence."""
    states = [labels.TASK_NAMES[task_id] for task_id in labels.TASK_IDS if task_id != labels.NO_TASK_ID]
    counts = pd.DataFrame(0, index=[START_STATE] + states, columns=states + [END_STATE], dtype=int)
    for _, take in sequences.groupby("take", sort=False):
        names = [START_STATE] + take["task_name"].tolist() + [END_STATE]
        for current, following in zip(names[:-1], names[1:]):
            counts.loc[current, following] += 1
    counts.index.name = "current"
    counts.columns.name = "next"
    return counts


def to_probabilities(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised P(next | current); rows never observed stay all zero."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def duration_stats(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["task_id", "task_name"])["duration_sec"]
    stats = grouped.agg(
        count="count", mean="mean", std="std", median="median", min="min", max="max",
        p90=lambda d: d.quantile(0.90), p95=lambda d: d.quantile(0.95),
    )
    return stats.reset_index().round(3)


def plot_transition_heatmap(probabilities: pd.DataFrame, counts: pd.DataFrame,
                            title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(probabilities.values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(probabilities.shape[1]), probabilities.columns, rotation=35, ha="right")
    ax.set_yticks(range(probabilities.shape[0]), probabilities.index)
    ax.set_xlabel("next task")
    ax.set_ylabel("current task")
    ax.set_title(title)
    for row in range(probabilities.shape[0]):
        for col in range(probabilities.shape[1]):
            n = counts.iat[row, col]
            if n == 0:
                continue
            p = probabilities.iat[row, col]
            ax.text(col, row, f"{p:.2f}\n(n={n})", ha="center", va="center", fontsize=8,
                    color="white" if p > 0.55 else "#222222")
    fig.colorbar(image, ax=ax, label="P(next | current)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_durations(frame: pd.DataFrame, stats: pd.DataFrame, title: str, path: Path) -> None:
    stats = stats.sort_values("task_id")
    data = [frame.loc[frame["task_id"] == task_id, "duration_sec"].values for task_id in stats["task_id"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, showfliers=True, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#c6dbef", edgecolor="#3a6ea5"),
               medianprops=dict(color="#08306b", linewidth=2),
               flierprops=dict(marker="o", markersize=4, markerfacecolor="none", markeredgecolor="#777777"))
    ax.scatter(range(1, len(data) + 1), stats["mean"], marker="D", s=30, color="#d95f02", zorder=3, label="mean")
    ax.set_xticks(range(1, len(data) + 1),
                  [f"{name}\n(n={count})" for name, count in zip(stats["task_name"], stats["count"])])
    ax.set_ylabel("duration [s]")
    ax.set_title(title)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_all(args.annotations_dir)
    blocks = to_blocks(entries)
    print(f"{entries['take'].nunique()} takes, {len(entries)} task entries, {len(blocks)} task blocks")

    entries.to_csv(args.out_dir / "task_sequences.csv", index=False)

    for suffix, sequence, label in (("", blocks, "task order (repeats merged)"),
                                    ("_raw", entries, "every annotation (repeats as self-transitions)")):
        counts = transition_counts(sequence)
        probabilities = to_probabilities(counts)
        counts.to_csv(args.out_dir / f"transition_counts{suffix}.csv")
        probabilities.round(4).to_csv(args.out_dir / f"transition_probabilities{suffix}.csv")
        plot_transition_heatmap(probabilities, counts, f"Task transition probabilities - {label}",
                                args.out_dir / f"transition_probabilities{suffix}.png")

    for name, frame, label in (("entry", entries, "per annotation span"),
                               ("block", blocks, "per merged block of the same task")):
        stats = duration_stats(frame)
        stats.to_csv(args.out_dir / f"duration_stats_{name}.csv", index=False)
        plot_durations(frame, stats, f"Task duration - {label}", args.out_dir / f"duration_{name}.png")
        print(f"\nDuration [s], {label}:\n{stats.to_string(index=False)}")

    print(f"\nTask order probabilities:\n{to_probabilities(transition_counts(blocks)).round(2).to_string()}")
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
