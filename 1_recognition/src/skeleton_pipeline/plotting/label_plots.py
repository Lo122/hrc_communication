"""Debug plot for the label curves built by skeleton_pipeline.dataset.labels
-- one figure with one panel per label-score DataFrame (step_id/status_id in
both smoothing variants, plus task_progress), sibling to feature_plots.py's
per-feature-panel plots."""
from pathlib import Path


def plot_label_debug(take_id: str, debug_frames: dict, logger,
                      show: bool = False, save: bool = True,
                      save_path: Path = Path()) -> Path | None:
    """debug_frames: {panel_title: pd.DataFrame} as returned by
    skeleton_pipeline.dataset.labels.extract_labels()'s second return value."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(debug_frames), 1, figsize=(12, 2.5 * len(debug_frames)), sharex=True)
    if len(debug_frames) == 1:
        axes = [axes]
    for ax, (title, db) in zip(axes, debug_frames.items()):
        for col in db.columns:
            ax.plot(db.index, db[col], label=str(col), linewidth=1)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=6, ncol=len(db.columns) or 1)
    axes[-1].set_xlabel("frame")
    fig.suptitle(take_id)
    fig.tight_layout()

    saved_path = None
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        saved_path = save_path
        if logger:
            logger.info("  Saved label debug plot -> %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
