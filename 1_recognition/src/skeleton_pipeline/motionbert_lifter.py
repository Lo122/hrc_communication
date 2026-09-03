"""Streaming 2D->3D lifting via MotionBERT's DSTformer
(https://github.com/Walter0807/MotionBERT) -- app/venv-local variant of
world_pose/pose/motionbert_lifter.py's MotionBERTStreamingLifter.

Only real difference from that version: the COCO->H36M keypoint remap uses
skeleton_pipeline/coco_h36m.py (pure numpy) instead of mmpose's
``convert_keypoint_definition`` -- this venv deliberately does not install
mmpose/mmcv/mmdet (see data_proc_3d/app/pyproject.toml's comment), since
this pipeline only needs YOLO 2D detection + MotionBERT, not any of
mmpose's own models. Axis remap/coordinate conventions, docstring caveats
("written from source, NOT run end-to-end when first written -- verify
before trusting") are otherwise identical to the original; see that
module for the full reasoning.

Default config/checkpoint here point at the FULL-SIZE, rootrel:True
checkpoint (FT_MB_release_MB_ft_h36m, config MB_ft_h36m.yaml) rather than
world_pose's own default (the smaller/faster "lite" variant) -- this is
what this project's LSTM training-data generation has settled on after
comparing checkpoints live (see conversation/README notes): FT_MB_release_
MB_ft_h36m and MB_ft_h36m_global share byte-identical architecture
(dim_feat=512, depth=5, clip_len=243) and differ only in whether the
(discarded -- see _postprocess_axes/lift() below) absolute pelvis offset
was part of the training objective, so rootrel:True is the more
appropriate pick since only the root-relative SHAPE is ever kept here.
"""
import collections
import os
from pathlib import Path

import numpy as np

from skeleton_pipeline.coco_h36m import coco_to_h36m_conf, coco_to_h36m_xy

MOTIONBERT_REPO_DIR = Path(os.environ.get(
    "MOTIONBERT_REPO_DIR",
    Path(__file__).resolve().parents[4] / "MotionBERT",
))

DEFAULT_CONFIG = MOTIONBERT_REPO_DIR / "configs" / "pose3d" / "MB_ft_h36m.yaml"
DEFAULT_CHECKPOINT = (
    MOTIONBERT_REPO_DIR / "checkpoint" / "pose3d" / "FT_MB_release_MB_ft_h36m" / "best_epoch.bin")


def _postprocess_axes(points_xyz):
    """See world_pose/pose/motionbert_lifter.py's "Axis remap" docstring
    note -- identical formula, confirmed against MotionBERT's own
    lib/utils/vismo.py visualization code, not guessed."""
    return points_xyz[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])


def _ensure_repo_on_path():
    import sys

    if not MOTIONBERT_REPO_DIR.is_dir():
        raise FileNotFoundError(
            f"MotionBERT repo not found at {MOTIONBERT_REPO_DIR}. Set MOTIONBERT_REPO_DIR "
            "or clone https://github.com/Walter0807/MotionBERT.git there first.")
    repo_dir_str = str(MOTIONBERT_REPO_DIR)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)


def _load_model(config_path, checkpoint_path, device):
    _ensure_repo_on_path()
    import torch
    from lib.utils.learning import load_backbone
    from lib.utils.tools import get_config

    if not Path(config_path).exists():
        raise FileNotFoundError(f"MotionBERT config not found at {config_path}.")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"MotionBERT checkpoint not found at {checkpoint_path}.")

    cfg = get_config(str(config_path))
    model = load_backbone(cfg)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_pos"] if "model_pos" in checkpoint else checkpoint
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model, cfg


class MotionBERTStreamingLifter:
    """Causal-window streaming lift -- see world_pose/pose/
    motionbert_lifter.py's MotionBERTStreamingLifter docstring for the full
    "causal-shifted window approximation" reasoning (only the window's LAST
    position is used/returned each call, so it's live-capable with zero
    added latency, at the cost of not exactly matching a true whole-clip
    offline/non-causal pass)."""

    def __init__(self, config_path=DEFAULT_CONFIG, checkpoint_path=DEFAULT_CHECKPOINT,
                 clip_len=None, device="cpu"):
        self._device = device
        self._model, self._cfg = _load_model(config_path, checkpoint_path, device)
        self._clip_len = int(clip_len or self._cfg.get("clip_len", self._cfg.get("maxlen", 243)))
        self._buffer = collections.deque(maxlen=self._clip_len)

    def reset(self):
        """Clear the rolling buffer -- call between videos/subjects so a
        new clip doesn't inherit stale frames from a previous one."""
        self._buffer.clear()

    def lift(self, keypoints_coco_xy, image_size=None, keypoints_conf=None):
        """keypoints_coco_xy: (17, 2) pixel coords, COCO order.
        Returns (17, 3) H36M order, root-relative, or None if lifting
        failed (e.g. degenerate all-zero input)."""
        import torch
        from lib.utils.utils_data import crop_scale

        keypoints_coco_xy = np.asarray(keypoints_coco_xy, dtype=np.float32)
        if not np.any(keypoints_coco_xy):
            return None
        if keypoints_conf is None:
            keypoints_conf = np.ones(keypoints_coco_xy.shape[0], dtype=np.float32)

        keypoints_h36m_xy = coco_to_h36m_xy(keypoints_coco_xy)
        conf_h36m = coco_to_h36m_conf(keypoints_conf)
        frame = np.concatenate(
            [keypoints_h36m_xy, conf_h36m[:, None]], axis=-1).astype(np.float32)
        self._buffer.append(frame)

        window = np.stack(self._buffer, axis=0)  # (T, 17, 3)
        normalized = crop_scale(window, scale_range=[1.0, 1.0]).astype(np.float32)

        batch = torch.from_numpy(normalized[None]).to(self._device)  # (1, T, 17, 3)
        with torch.no_grad():
            output = self._model(batch)  # (1, T, 17, 3)
        output = output[0].detach().cpu().numpy()

        last = output[-1].copy()
        last = last - last[0]  # force root-relative
        return _postprocess_axes(last)
