"""The step classifiers recognition can run, told apart by what a model directory holds.

legacy (AssistLSTM) -- config.json with "feature_keys", best_model.pth, one norm .npz
    (best_model/3d_skeleton*). One softmax over the steps, one progress value, an
    optional mistake softmax. Fed one feature vector per loop frame, window_size frames.

multi-head (GRU/LSTM) -- four heads: independent task sigmoids (multi-label), a
    progress lane per task, a mistake logit and a background logit ("nobody is
    working"). Fed one feature vector every sample period -- the rate the model was
    trained at, whatever the loop runs at. The same network comes in two layouts:
      export -- config.json with "heads", feature_selection.json, model_weights.pth,
        standardization.npz (best_model/S3_10fps_8s_bg05); rate from "model_rate_hz".
      training run -- config.json with "feature_keys" and an idle head in its
        "loss_weights", best_model.pth, a per-panel norm_stats.npz
        (best_model/3d_skeleton_05, _06); rate input_fps / sample_interval. The
        checkpoint names the parameters lstm/step_head/progress_head/mistake_head/
        idle_head; they are renamed onto multihead_models' rnn/heads.* on load.

Both predict a StepPrediction over config.STEP_NAMES, so the stabilizer, the round
bookkeeping and the decision layer see the same step ids whichever format runs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

LEGACY = "legacy"
MULTI_HEAD = "multi-head"

# Model-config feature keys that are a concatenation of several extractor outputs, in
# this exact order (legacy format). Each component is normalized separately first (the
# norm .npz holds per-component stats), then concatenated -- the order training used.
COMPOSITE_FEATURE_KEYS = {"pol_angles": ("polar_azimuth", "polar_elevation")}

# The step a multi-head model's background head stands for, as config.STEP_NAMES calls
# it (the training corpus says "No Related Task").
IDLE_STEP_NAMES = ("Non Related Task", "No Related Task")

# Transforms a multi-head model's feature_selection.json (or a training run's
# "feature_transforms") may ask for -- all of them applied in
# MultiHeadStepModel.feature_vector.
KNOWN_TRANSFORMS = {"polar_azimuth", "ratios", "standardisation"}

# A training run's checkpoint names -> multihead_models.py's, for the same network.
TRAINING_RUN_KEY_PREFIXES = {
    "lstm.": "rnn.",
    "gru.": "rnn.",
    "shared.": "heads.shared.",
    "step_head.": "heads.task.",
    "progress_head.": "heads.prog.",
    "mistake_head.": "heads.mist.",
    "idle_head.": "heads.bg.",
}

# The frame rate a training run's windows were cut at, when its config doesn't say
# (3d_skeleton_06 does: "input_fps": 30; the dataset is 30 fps throughout).
TRAINING_RUN_DEFAULT_FPS = 30.0


@dataclass
class StepPrediction:
    """One inference, over the model's step_labels.

    step_scores: what the stabilizer smooths and the plots show, one per step. A
        softmax for the legacy format; for the multi-head format task i scores
        P(task i) * (1 - P(idle)) and the idle step scores P(idle), so idle wins
        whenever P(idle) > 0.5: no task can score more than 1 - P(idle) then.
    progress: 0-1, one value (legacy) or one per step (multi-head, 0 for idle).
    mistake_probabilities: [P(no mistake), P(mistake), ...], or None without a
        mistake head.
    idle_probability: the background head's P(nobody is working); None for legacy.
    task_probabilities: each step's own probability, 0-1 independently of the others
        -- the multi-head task sigmoids, with P(idle) at the idle step. Unlike
        step_scores they are not scaled by 1 - P(idle). None for legacy, whose
        softmax step_scores already are its probabilities.
    """

    step_scores: np.ndarray
    progress: float | np.ndarray
    mistake_probabilities: np.ndarray | None = None
    idle_probability: float | None = None
    task_probabilities: np.ndarray | None = None

    @property
    def raw_step_id(self) -> int:
        return int(np.argmax(self.step_scores))

    @property
    def confidence(self) -> float:
        return float(np.max(self.step_scores))

    @property
    def step_progress(self) -> np.ndarray | None:
        """Progress per step, or None for a model with one progress value."""
        return self.progress if isinstance(self.progress, np.ndarray) else None

    def progress_of(self, step_id: int | None) -> float:
        """Progress of that step: the one value for the legacy format, that step's
        lane for the multi-head format (0 for no step)."""
        if not isinstance(self.progress, np.ndarray):
            return float(self.progress)
        if step_id is None or not 0 <= step_id < len(self.progress):
            return 0.0
        return float(self.progress[step_id])

    def is_finite(self) -> bool:
        values = [self.step_scores, np.atleast_1d(self.progress)]
        if self.mistake_probabilities is not None:
            values.append(self.mistake_probabilities)
        if self.idle_probability is not None:
            values.append(np.atleast_1d(self.idle_probability))
        if self.task_probabilities is not None:
            values.append(self.task_probabilities)
        return all(np.all(np.isfinite(value)) for value in values)


class StepModel:
    """What RecognitionManager needs from a step classifier. Reading the directory
    needs only json/numpy; load() builds the network once torch is imported."""

    format: str
    model_dir: Path
    window_size: int
    # Seconds between the feature vectors the model is fed; None for every loop frame.
    sample_period_s: float | None = None
    step_labels: list[str]

    @property
    def num_steps(self) -> int:
        return len(self.step_labels)

    def load(self, torch, device) -> None:
        raise NotImplementedError

    def feature_vector(self, features: dict) -> np.ndarray:
        """One frame's extractor output as the model's normalized input vector."""
        raise NotImplementedError

    def predict(self, window: np.ndarray) -> StepPrediction:
        """window: (window_size, input_dim) feature vectors, oldest first."""
        raise NotImplementedError

    def describe(self) -> str:
        rate = ("every frame" if self.sample_period_s is None
                else f"{1.0 / self.sample_period_s:g} Hz")
        return (f"{self.format} step model {self.model_dir.name}: window {self.window_size} "
                f"samples at {rate}, {self.num_steps} steps")


def open_step_model(model_dir: str | Path, *, model_path=None, model_config_path=None,
                    norm_path=None, feature_keys=None) -> StepModel:
    """The step model in model_dir, in whichever format its files say. The keyword
    overrides apply to the legacy format (they predate the multi-head one)."""
    model_dir = Path(model_dir)
    config_path = Path(model_config_path) if model_config_path else model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)
    if "heads" in model_config and (model_dir / "feature_selection.json").exists():
        return MultiHeadStepModel(model_dir, model_config)
    if is_multi_head_training_run(model_config):
        return MultiHeadStepModel(model_dir, model_config)
    if "feature_keys" in model_config:
        return LegacyStepModel(model_dir, model_config, model_path=model_path,
                               norm_path=norm_path, feature_keys=feature_keys)
    raise ValueError(
        f"Unknown step model format in {model_dir}: config.json has neither 'heads' (with "
        f"feature_selection.json beside it) nor 'feature_keys'.")


def is_multi_head_training_run(model_config: dict) -> bool:
    """A training run's config for the 4-head network (best_model/3d_skeleton_05):
    'feature_keys' like the legacy format, but trained with an idle head."""
    return "feature_keys" in model_config and (
        "idle" in model_config.get("loss_weights", {})
        or "idle_target" in model_config
        or "idle" in model_config.get("decoding", {}))


class LegacyStepModel(StepModel):
    """AssistLSTM: softmax steps, one progress value, optional mistake softmax."""

    format = LEGACY

    def __init__(self, model_dir: Path, model_config: dict, *, model_path=None,
                 norm_path=None, feature_keys=None):
        self.model_dir = model_dir
        self.model_config = model_config
        self.model_path = Path(model_path) if model_path is not None else self._find_model_path()
        self.norm_path = Path(norm_path) if norm_path is not None else self._find_norm_path()
        self.feature_keys = list(feature_keys or model_config["feature_keys"])
        self.window_size = int(model_config["window_size"])
        num_steps = int(model_config["num_steps"])
        self.step_labels = (list(config.STEP_NAMES) if len(config.STEP_NAMES) == num_steps
                            else [f"step {i}" for i in range(num_steps)])
        self._torch = self._device = self._model = self._norm = None

    def load(self, torch, device) -> None:
        from feature_utils.feature_normalizer import NormRealTime

        # AssistLSTM's definition lives beside the checkpoint in best_model/, one
        # level above the per-variant subdirectory the weights sit in.
        definition_dir = str(self.model_dir.parent)
        if definition_dir not in sys.path:
            sys.path.insert(0, definition_dir)
        from LSTM_model_train import AssistLSTM

        # num_mistakes/num_layers/dropout must mirror the values the checkpoint was
        # trained with, or load_state_dict rejects the weights (a mistake_head trained
        # into the checkpoint has no place to go in a head-less model). Older configs
        # predate these keys, hence the defaults.
        num_mistakes = self.model_config.get("num_mistakes")
        model = AssistLSTM(
            input_dim=int(self.model_config["input_dim"]),
            hidden_dim=int(self.model_config["hidden_dim"]),
            num_steps=self.num_steps,
            num_layers=int(self.model_config.get("num_layers", 1)),
            dropout=float(self.model_config.get("dropout", 0.5)),
            num_mistakes=int(num_mistakes) if num_mistakes is not None else None,
        ).to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()
        self._torch, self._device, self._model = torch, device, model
        self._norm = NormRealTime(str(self.norm_path), self.feature_keys)

    def feature_vector(self, features: dict) -> np.ndarray:
        features = self._norm.normalize_features(features)
        values = []
        for key in self.feature_keys:
            for part_key in COMPOSITE_FEATURE_KEYS.get(key, (key,)):
                try:
                    value = features[part_key]
                except KeyError:
                    raise KeyError(
                        f"Feature '{part_key}' (for model config feature_key '{key}') is not "
                        f"produced by the feature extractor. Available features: "
                        f"{sorted(features)}.") from None
                if isinstance(value, self._torch.Tensor):
                    value = value.detach().cpu().numpy()
                values.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(values, axis=0)

    def predict(self, window: np.ndarray) -> StepPrediction:
        torch = self._torch
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            # Models trained with a mistake head return a third output; models
            # without one return two.
            outputs = self._model(x)
            step_probs = torch.softmax(outputs[0], dim=1).squeeze(0).cpu().numpy()
            progress = float(outputs[1].item())
            mistake_probs = (torch.softmax(outputs[2], dim=1).squeeze(0).cpu().numpy()
                             if len(outputs) > 2 else None)
        return StepPrediction(step_probs, progress, mistake_probs)

    def _find_model_path(self) -> Path:
        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected trained model at {model_path}.")
        return model_path

    def _find_norm_path(self) -> Path:
        candidates = sorted(self.model_dir.glob("*.npz"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No norm .npz file found in {self.model_dir}.")
        raise FileExistsError(f"Expected exactly one norm .npz file in {self.model_dir}, "
                              f"found {len(candidates)}.")


class MultiHeadStepModel(StepModel):
    """The 4-head GRU/LSTM (best_model/multihead_models.py): multi-label task sigmoids,
    per-task progress lanes, a mistake logit and a background logit. Reads either
    layout (module docstring); everything after reading the directory is shared."""

    format = MULTI_HEAD

    def __init__(self, model_dir: Path, model_config: dict):
        self.model_dir = model_dir
        self.model_config = model_config
        self.input_dim = int(model_config["input_dim"])
        # Set by the layout reader: panels, panel_sizes (per panel, as the vector is
        # built -- azimuth becomes a sin and a cos block), mean, std, window_size,
        # sample_period_s, classes, and for load(): _network, _weight_files, _key_prefixes.
        if (model_dir / "feature_selection.json").exists():
            self._read_export()
        else:
            self._read_training_run()
        if self.mean.size != self.input_dim or sum(self.panel_sizes.values()) != self.input_dim:
            raise ValueError(f"{model_dir.name}: the feature panels add up to "
                             f"{sum(self.panel_sizes.values())} columns and the stats to "
                             f"{self.mean.size}; config.json says {self.input_dim} inputs.")

        self.step_labels = list(config.STEP_NAMES)
        missing = [name for name in self.classes if name not in self.step_labels]
        if missing:
            raise ValueError(f"{model_dir.name}: classes {missing} are not in config.STEP_NAMES.")
        idle = [name for name in IDLE_STEP_NAMES if name in self.step_labels]
        if not idle:
            self.step_labels.append(IDLE_STEP_NAMES[0])
            idle = [IDLE_STEP_NAMES[0]]
        self.class_steps = np.array([self.step_labels.index(name) for name in self.classes])
        self.idle_step = self.step_labels.index(idle[0])
        self._torch = self._device = self._model = None

    def _read_export(self) -> None:
        """best_model/S3_10fps_8s_bg05: feature_selection.json + standardization.npz."""
        name = self.model_dir.name
        with (self.model_dir / "feature_selection.json").open("r", encoding="utf-8") as file:
            selection = json.load(file)
        self.panels = list(selection["panels"])
        self._check_transforms(selection.get("transforms", {}))
        if selection.get("joints_dropped"):
            raise ValueError(f"{name}: dropped joints {selection['joints_dropped']} "
                             f"are not supported; the extractor emits every joint.")

        stats = np.load(self.model_dir / "standardization.npz", allow_pickle=True)
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        columns = [str(column) for column in stats["columns"]]
        if columns != list(selection["column_order"]):
            raise ValueError(f"{name}: standardization.npz columns do not match "
                             f"feature_selection.json's column_order.")
        self.panel_sizes = dict(selection["columns_per_panel"])

        window = selection["window"]
        self.window_size = int(window["samples"])
        self.sample_period_s = 1.0 / float(window["model_rate_hz"])
        self.classes = list(self.model_config["classes"])

        self._network = (self.model_config.get("architecture", "gru"),
                         int(self.model_config.get("hidden_dim", 128)),
                         int(self.model_config.get("num_layers", 1)))
        self._weight_files = ["model_weights.pth", "model_bundle.pt"]
        self._key_prefixes = {}

    def _read_training_run(self) -> None:
        """best_model/3d_skeleton_05: feature_keys in config.json + a per-panel
        norm_stats.npz ("<panel>_mean" / "<panel>_std", azimuth as its 32 sin/cos
        columns) + best_model.pth."""
        name, cfg = self.model_dir.name, self.model_config
        self.panels = list(cfg["feature_keys"])
        self._check_transforms(cfg.get("feature_transforms", {}))

        norm_path = self.model_dir / cfg.get("normalization_file", "norm_stats.npz")
        stats = np.load(norm_path, allow_pickle=True)
        missing = [panel for panel in self.panels
                   if f"{panel}_mean" not in stats.files or f"{panel}_std" not in stats.files]
        if missing:
            raise ValueError(f"{name}: {norm_path.name} has no stats for panels {missing}.")
        means = [np.asarray(stats[f"{panel}_mean"], dtype=np.float32).reshape(-1)
                 for panel in self.panels]
        stds = [np.asarray(stats[f"{panel}_std"], dtype=np.float32).reshape(-1)
                for panel in self.panels]
        self.panel_sizes = {panel: mean.size for panel, mean in zip(self.panels, means)}
        self.mean, self.std = np.concatenate(means), np.concatenate(stds)

        if cfg.get("mode", "seq2one") != "seq2one" or int(cfg.get("predict_offset", 0)) != 0:
            raise ValueError(f"{name}: only seq2one models predicting the window's last "
                             f"frame run online (mode {cfg.get('mode')!r}, predict_offset "
                             f"{cfg.get('predict_offset')}).")
        fps = float(cfg.get("input_fps", TRAINING_RUN_DEFAULT_FPS))
        self.window_size = int(cfg["window_size"])
        self.sample_period_s = int(cfg.get("sample_interval", 1)) / fps

        num_tasks = int(cfg["num_steps"])
        self.classes = list(cfg.get("task_names") or [
            step for step in config.STEP_NAMES if step not in IDLE_STEP_NAMES][:num_tasks])
        if len(self.classes) != num_tasks:
            raise ValueError(f"{name}: {num_tasks} task heads but {len(self.classes)} task "
                             f"names ({self.classes}).")

        self._network = (str(cfg.get("model", "LSTM")).lower(), int(cfg.get("hidden_dim", 128)),
                         int(cfg.get("num_layers", 1)))
        self._weight_files = [file for file in (cfg.get("model_file"), "best_model.pth",
                                                "model_weights.pth") if file]
        self._key_prefixes = TRAINING_RUN_KEY_PREFIXES

    def _check_transforms(self, transforms: dict) -> None:
        unknown = set(transforms) - KNOWN_TRANSFORMS
        if unknown:
            raise ValueError(f"{self.model_dir.name}: unsupported feature transforms "
                             f"{sorted(unknown)}.")

    def load(self, torch, device) -> None:
        definition_dir = str(self.model_dir.parent)
        if definition_dir not in sys.path:
            sys.path.insert(0, definition_dir)
        from multihead_models import GRUNet, LSTMNet, build

        architecture, hidden_dim, num_layers = self._network
        networks = {"lstm": LSTMNet, "gru": GRUNet}
        model = (networks[architecture](self.input_dim, hidden_dim, num_layers, len(self.classes))
                 if architecture in networks
                 else build(architecture, self.input_dim, n_tasks=len(self.classes))).to(device)
        model.load_state_dict(self._state_dict(torch, device))
        model.eval()
        self._torch, self._device, self._model = torch, device, model

    def _state_dict(self, torch, device) -> dict:
        """The first weights file present, its keys renamed onto multihead_models'."""
        path = next((self.model_dir / file for file in self._weight_files
                     if (self.model_dir / file).exists()), None)
        if path is None:
            raise FileNotFoundError(f"{self.model_dir.name}: none of {self._weight_files} found.")
        state = torch.load(path, map_location=device, weights_only=path.suffix != ".pt")
        for wrapper in ("state_dict", "model_state_dict"):  # a bundle or a training checkpoint
            if isinstance(state.get(wrapper), dict):
                state = state[wrapper]
        renamed = {}
        for key, value in state.items():
            prefix = next((p for p in self._key_prefixes if key.startswith(p)), None)
            renamed[key if prefix is None else self._key_prefixes[prefix] + key[len(prefix):]] = value
        return renamed

    def feature_vector(self, features: dict) -> np.ndarray:
        blocks = []
        for panel in self.panels:
            try:
                values = np.asarray(features[panel], dtype=np.float32).reshape(-1)
            except KeyError:
                raise KeyError(f"Feature panel '{panel}' ({self.model_dir.name}) is not produced "
                               f"by the feature extractor. Available: {sorted(features)}.") from None
            if panel == "polar_azimuth":  # wraps at +/-180: degrees -> sin block, cos block
                radians = np.deg2rad(values)
                values = np.concatenate([np.sin(radians), np.cos(radians)])
            elif panel == "ratios":  # spikes when the lifter collapses shoulder width
                values = np.clip(values, 0.0, 4.0)
            if values.size != self.panel_sizes[panel]:
                raise ValueError(f"Feature panel '{panel}' has {values.size} columns; "
                                 f"{self.model_dir.name} expects {self.panel_sizes[panel]}.")
            blocks.append(values)
        return (np.concatenate(blocks) - self.mean) / self.std

    def predict(self, window: np.ndarray) -> StepPrediction:
        torch = self._torch
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(x)
            tasks = torch.sigmoid(out["task"]).squeeze(0).cpu().numpy()
            lanes = out["prog"].squeeze(0).cpu().numpy()
            idle = float(torch.sigmoid(out["bg"]).item())
            mistake = float(torch.sigmoid(out["mistake"]).item())
        scores = np.zeros(self.num_steps, dtype=np.float32)
        scores[self.class_steps] = tasks * (1.0 - idle)
        scores[self.idle_step] = idle
        probabilities = np.zeros(self.num_steps, dtype=np.float32)
        probabilities[self.class_steps] = tasks
        probabilities[self.idle_step] = idle
        progress = np.zeros(self.num_steps, dtype=np.float32)
        progress[self.class_steps] = np.clip(lanes, 0.0, 1.0)  # a linear head: clip to 0-1
        return StepPrediction(scores, progress, np.array([1.0 - mistake, mistake], dtype=np.float32),
                              idle_probability=idle, task_probabilities=probabilities)
