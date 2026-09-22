"""Streaming 2D->3D lifting via MotionBERT's DSTformer
(https://github.com/Walter0807/MotionBERT) -- app/venv-local variant of
world_pose/pose/motionbert_lifter.py's MotionBERTStreamingLifter.

Only real difference from that version: the COCO->H36M keypoint remap uses
skeleton_utils/coco_h36m.py (pure numpy) instead of mmpose's
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

from logging_setup import get_logger
from skeleton_utils.coco_h36m import coco_to_h36m_conf, coco_to_h36m_xy

logger = get_logger(__name__)

MOTIONBERT_REPO_DIR = Path(os.environ.get(
    "MOTIONBERT_REPO_DIR",
    # parents[4] is .../codes: this file sits at
    # <repo>/<pkg-parent>/src/skeleton_utils/, and MotionBERT is cloned as a
    # SIBLING of <repo> (.../codes/MotionBERT). Holds for both copies of this
    # file (LSTM_HRC/data_proc_3d/... and hrc_communication/1_recognition/...),
    # which is what lets them stay byte-identical.
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
    offline/non-causal pass).

    half=True runs the DSTformer forward pass in fp16 (CUDA only). Measured
    on an RTX 3060 Laptop at clip_len=81: 31.0 ms -> 17.9 ms per lift, a
    1.73x speedup on this stage. Two things that measurement also showed,
    both worth knowing before reaching for other knobs:

      - There is a ~18 ms FLOOR that is kernel-launch-bound, not
        compute-bound: clip_len 9 / 27 / 41 all cost ~18-19 ms in fp32,
        i.e. cutting the window 9x changes nothing, because DSTformer
        (depth=5, alternating spatial/temporal attention over
        (1, T, 17, 512)) issues hundreds of tiny kernels and the GPU waits
        on launches rather than doing math. fp16 at clip_len=81 lands on
        that same floor, so REDUCING clip_len buys nothing fp16 does not
        already give -- and unlike fp16 it changes the lifter's output
        distribution, which would invalidate an existing training set.
        To go below the floor the lever is CUDA graphs / torch.compile
        (mode="reduce-overhead"), not a smaller window.

      - It does NOT require regenerating training data. fp16 computes the
        same function with coarser rounding, unlike a centered-vs-causal
        or frame-rate change, which alter WHICH function is computed. A/B
        over 400 real replayed frames (byte-identical 2D input,
        clip_len=81): 0 NaN/inf introduced, relative rms 8.6e-4, i.e. 2% of
        the per-joint jitter skeleton_utils.dataset.augment injects
        deliberately (noise_sigma_m=0.01). The comparison is deliberately
        stated as a RATIO: this output is root-relative in crop_scale's
        normalized image-space units, not meters (see depth_anchor.py), so
        an absolute "0.2 mm" would be a unit that does not exist here --
        but both figures live in the same space, so their ratio is exactly
        the quantity that matters.

    fp16's real risk is overflow, not precision: fp16 tops out at 65504 and
    attention intermediates can exceed that, which gives inf/NaN rather than
    a small error. It did not happen in the A/B above, but verify after any
    checkpoint/config change rather than assuming. Prefer bf16 (same
    exponent range as fp32) if a future GPU/torch combination makes it the
    faster path here.
    """

    def __init__(self, config_path=DEFAULT_CONFIG, checkpoint_path=DEFAULT_CHECKPOINT,
                 clip_len=None, device="cpu", half=False):
        self._device = device
        self._model, self._cfg = _load_model(config_path, checkpoint_path, device)
        self._clip_len = int(clip_len or self._cfg.get("clip_len", self._cfg.get("maxlen", 243)))
        self._buffer = collections.deque(maxlen=self._clip_len)

        # fp16 is a CUDA-only win -- on CPU many half kernels fall back to
        # slower paths (or are unimplemented), so silently honouring it there
        # would make things worse, not faster.
        self._half = bool(half) and str(device).startswith("cuda")
        if half and not self._half:
            logger.warning("half=True ignored on device=%r -- fp16 only accelerates CUDA "
                           "here (see this class's docstring).", device)
        if self._half:
            self._model = self._model.half()

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
        if self._half:
            batch = batch.half()
        with torch.no_grad():
            output = self._model(batch)  # (1, T, 17, 3)
        # .float() BEFORE .numpy(), and not optional: everything downstream is
        # float64 numpy, and BoneLengthConstraintFilter in particular is a
        # STATEFUL accumulator. Handing it a float16 array would carry fp16's
        # ~1e-3 relative error into a filter that integrates over the whole
        # clip, which is the one place the rounding could actually compound.
        output = output[0].float().detach().cpu().numpy()

        last = output[-1].copy()
        last = last - last[0]  # force root-relative
        return _postprocess_axes(last)
