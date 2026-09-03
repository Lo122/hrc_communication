"""Per-panel feature plots -- one PNG per panel group (see
skeleton_pipeline/features/h36m_features.py's compute_all_features), each
showing every column in that panel as a labeled time-series line. Mirrors
data_proc_2d's panel_titles/feature_dataframes grouping (one plot per
feature GROUP, e.g. all 16 joints' speed together on one "Joint Speed"
plot), not one plot per individual scalar column -- 181 tiny individual
plots would be far less useful than ~16 grouped ones for actually reading
these features.

Deliberately matplotlib (not the fast plain-OpenCV renderer in
skeleton_pipeline/render/) -- these are a handful of one-shot static plots
per video, not a per-frame video overlay, so matplotlib's ~100ms/plot cost
(see skeleton_pipeline/render/skeleton_video.py's docstring) is a non-issue
here.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- these are saved-to-disk plots, no display
import matplotlib.pyplot as plt


def _slug(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def plot_panels(feature_dict, panel_groups, timestamps, output_dir, prefix,
                 max_lines_per_plot=20):
    """feature_dict/panel_groups: see compute_all_features()'s return value.
    timestamps: (T,) seconds. Writes one PNG per panel to
    output_dir/<prefix>_<panel_slug>.png. Returns the list of written paths.

    max_lines_per_plot: panels with more columns than this are split across
    multiple figures (same panel title, "(1/2)" etc. suffix) so the legend
    stays readable -- none of this project's current panels need it (at
    most 16 joints/panel), but keeps this robust if that ever changes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for panel_title, columns in panel_groups.items():
        n_chunks = max(1, (len(columns) + max_lines_per_plot - 1) // max_lines_per_plot)
        for chunk_idx in range(n_chunks):
            chunk_columns = columns[chunk_idx * max_lines_per_plot:(chunk_idx + 1) * max_lines_per_plot]
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in chunk_columns:
                ax.plot(timestamps, feature_dict[col], label=col, linewidth=1.0)
            title = panel_title if n_chunks == 1 else f"{panel_title} ({chunk_idx + 1}/{n_chunks})"
            ax.set_title(title)
            ax.set_xlabel("time (s)")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            suffix = _slug(panel_title) if n_chunks == 1 else f"{_slug(panel_title)}_{chunk_idx + 1}"
            out_path = output_dir / f"{prefix}_{suffix}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            written.append(out_path)
    return written
