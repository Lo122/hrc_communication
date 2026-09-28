"""Subject-disjoint data layer for the ceiling-panel recognition benchmark.

Everything the DATASET_CARD warns about is handled here, in one place, so every
model in the comparison sees exactly the same data and the same split:

  * split by uid (subject), never by file or segment -- all 3 camera views of a
    take are one performance and must stay on the same side
  * augmented copies join the TRAINING split only, and only for training uids
  * normalisation statistics computed on the TRAIN subjects only
  * azimuth encoded as sin/cos (it wraps at +/-180 and a network cannot know that)
  * ratios winsorised (they spike 8-10x when the lifter collapses shoulder width)
  * NaN frames masked out of windows rather than silently poisoning the LSTM

Label targets (see `labels.py`, which is the read-only source of truth):

  * `task_id` is an argmax with NO background guard -- on a frame where every lane
    is 0.0 (real idle time) it returns class 0, "Pull Cables". Measured corpus-wide,
    8.15% of frames are such background and 61.9% of all "Pull Cables" frames are
    actually idle. So we carry an explicit `bg` flag and never train the task head
    on a background frame's phantom label.
  * `task_id_plateau_vector` is flat 1.0 across each annotated span -- the correct
    multi-label BCE target. `task_id_vector` (asymmetric_peak) peaks only at the
    span midpoint and is an argmax tie-breaker, not a span-presence signal.
"""
from __future__ import annotations
import glob, os, re, json
import numpy as np
import torch

PANEL_ORDER = [
    "joint_speed", "joint_acceleration",
    "joint_velocity_x", "joint_velocity_y", "joint_velocity_z",
    "joint_acceleration_x", "joint_acceleration_y", "joint_acceleration_z",
    "position_x_relative_to_pelvis", "position_y_relative_to_pelvis",
    "position_z_relative_to_pelvis",
    "polar_azimuth", "polar_elevation", "joint_angles", "ratios",
    "distance_from_center",
]
_ALL_PANELS = list(PANEL_ORDER)
# --- OLD corpus (original/), 7 lanes -------------------------------------
ALL_TASK_NAMES = ["Pull Cables", "Lift", "Align", "Screw", "Connect", "Clamp", "Place"]
LIFT_LANE = 1

# --- NEW corpus (hrc_communication/other/), 8 lanes -----------------------
# The annotators rebuilt the labels with two of our measured findings applied:
#   * Lane 1 (Lift) is empty in all 93 takes -- verified, 0 active frames.
#     Dropping Lift was worth +0.084 macro-F1 on 15/15 folds, so it is now
#     baked into the data rather than removed in code.
#   * Lane 7 "No Related Task" is an EXPLICIT human annotation of idle time,
#     18.2% of the corpus. Previously this was inferred (all-lanes-zero) and,
#     before that, silently mislabelled as class 0 -- 61.9% of "Pull Cables"
#     frames were actually idle.
# The six real classes are numerically identical to the old corpus (verified
# lane by lane); only their column order differs.
NEW_TASK_NAMES = ["Pull Cables", "Lift", "Place", "Align", "Screw",
                  "Connect Cables", "Clamp Coupling", "No Related Task"]
NEW_LIFT_LANE = 1
NEW_IDLE_LANE = 7

# Lift is an umbrella span covering the whole carry-and-hang sequence, with Place
# (94% nested) and Align (86%) living inside it. Two measured consequences:
#   * it is the lane the model hallucinates -- predicted >=0.5 on 73% of frames
#     where it is genuinely absent, which is why gating the trigger on it fails
#   * dropping it costs almost nothing: only 3.2% of frames have Lift as their
#     ONLY active class, and those correctly become background
# Kept as a switch so the 7-class arms stay reproducible for the thesis table.
DROP_LIFT = True


# --- Joint selection -----------------------------------------------------
# Column order of every per-joint panel: the 16 non-pelvis H36M joints, as
# defined by FEATURE_JOINTS in data_proc_3d/.../features/h36m_features.py
# (the live runtime's skeleton3d_pipeline.py uses the same order).
FEATURE_JOINT_NAMES = ["r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
                       "spine", "thorax", "neck", "head", "l_shoulder", "l_elbow",
                       "l_wrist", "r_shoulder", "r_elbow", "r_wrist"]
# joint_angles columns, in compute_joint_angles() order, named by the joint the
# angle is measured AT -- so dropping a knee also drops the knee angle.
ANGLE_JOINTS = ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder", "l_hip", "r_hip",
                "l_knee", "r_knee", "neck"]
LOWER_BODY = {"r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"}
# Joints removed from every per-joint panel. Empty = all 16 (the default).
# The 2 ratio columns are unaffected: they only involve shoulders/elbows/wrists.
DROP_JOINTS = set()

# Panel presets. loso.py/train.py select these by name via --panels; the
# evaluators re-apply the same preset from a saved run's config, so a model is
# always fed the exact columns it was trained on.
_REDUCED = ["position_x_relative_to_pelvis", "position_y_relative_to_pelvis",
            "position_z_relative_to_pelvis", "polar_elevation", "joint_angles",
            "distance_from_center"]
PANEL_PRESETS = {
    "reduced": _REDUCED,
    "reduced_vel": _REDUCED + ["joint_velocity_x", "joint_velocity_y", "joint_velocity_z"],
    "selected": ["joint_angles", "position_z_relative_to_pelvis", "joint_speed",
                 "distance_from_center", "polar_azimuth",
                 "position_y_relative_to_pelvis", "joint_acceleration"],
}


def apply_run_config(cfg):
    """Set the module switches a saved run was trained with (panels, idle
    handling, dropped joints). Missing keys fall back to the defaults."""
    global PANEL_ORDER, IDLE_AS_CLASS, DROP_JOINTS
    panels = cfg.get("panels", "all")
    PANEL_ORDER = list(PANEL_PRESETS.get(panels, _ALL_PANELS))
    IDLE_AS_CLASS = bool(cfg.get("idle_as_class", False))
    global DROP_LIFT
    DROP_LIFT = not bool(cfg.get("keep_lift", False))
    dj = cfg.get("drop_joints") or ""
    DROP_JOINTS = (set(LOWER_BODY) if dj == "lower"
                   else {j.strip() for j in dj.split(",") if j.strip()})


# Set by WindowSet/load_take when the 8-lane corpus is detected.
NEW_SCHEMA = False
IDLE_AS_CLASS = False     # True -> keep "No Related Task" as a 7th trainable lane


def task_names(drop_lift=None, new_schema=None, idle_as_class=None):
    new = NEW_SCHEMA if new_schema is None else new_schema
    idle = IDLE_AS_CLASS if idle_as_class is None else idle_as_class
    if new:
        d = DROP_LIFT if drop_lift is None else drop_lift
        n = [x for i, x in enumerate(NEW_TASK_NAMES)
             if (not d or i != NEW_LIFT_LANE) and (idle or i != NEW_IDLE_LANE)]
        return n
    d = DROP_LIFT if drop_lift is None else drop_lift
    return ([n for i, n in enumerate(ALL_TASK_NAMES) if i != LIFT_LANE]
            if d else list(ALL_TASK_NAMES))


def n_tasks(drop_lift=None):
    return len(task_names(drop_lift))


def _keep_lanes(drop_lift, n_lanes=7, idle_as_class=None):
    """Which label columns to keep. The 8-lane corpus always drops the empty
    Lift lane; the idle lane is kept only when it is being trained as a class."""
    if n_lanes == 8:
        # Lift lane: empty in hrc_communication/other (Lift removed), but a real
        # trimmed class in the f30a14d relabel. drop_lift=False keeps it.
        idle = IDLE_AS_CLASS if idle_as_class is None else idle_as_class
        return [i for i in range(8)
                if (not drop_lift or i != NEW_LIFT_LANE) and (idle or i != NEW_IDLE_LANE)]
    return ([i for i in range(len(ALL_TASK_NAMES)) if i != LIFT_LANE]
            if drop_lift else list(range(len(ALL_TASK_NAMES))))


# Module-level defaults, kept for callers that do not thread the flag through.
TASK_NAMES = task_names()
N_TASKS = n_tasks()


def take_key(path: str) -> tuple[int, str]:
    """(uid, take) -- the grouping key. Camera is deliberately NOT part of it."""
    b = os.path.basename(path)
    uid = int(re.search(r"uid-(\d+)", b).group(1))
    take = re.search(r"take-([\d-]+?)(?:_aug|_seg|\.pt)", b).group(1)
    return uid, take


def cam_take_key(path: str) -> tuple[str, int, str]:
    """(cam, uid, take) -- identifies one physical recording, for dedup."""
    b = os.path.basename(path)
    cam = re.search(r"cam-(\d+)", b).group(1)
    uid, take = take_key(path)
    return cam, uid, take


def dedup_augmented(paths):
    """The mirror copies _aug-01/02/03 of a take are byte-identical: mirroring is
    deterministic and the varying seeds only drive `rotation`/`noise`, both None
    in this corpus. Keeping all three would weight mirrored samples 3x against
    originals, so keep exactly one per physical recording."""
    seen, out = set(), []
    for p in sorted(paths):
        k = cam_take_key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


_TAKE_CACHE = {}


def load_take(path: str, drop_lift=None):
    """Cached front for _load_take.

    LOSO rebuilds train/val/test for each of 15 folds, and before this cache
    every fold re-read all ~186 .pt files from disk -- the GPU sat at 0% for
    most of each fold. The loaded arrays depend only on the file and on three
    switches, so they are keyed on exactly those and reused across folds.
    Results are bit-identical; only wall time changes.
    """
    key = (path, DROP_LIFT if drop_lift is None else drop_lift,
           IDLE_AS_CLASS, tuple(PANEL_ORDER), tuple(sorted(DROP_JOINTS)))
    if key not in _TAKE_CACHE:
        _TAKE_CACHE[key] = _load_take(path, drop_lift)
    return _TAKE_CACHE[key]


def _load_take(path: str, drop_lift=None):
    """-> dict of arrays, all aligned on the frame axis.

    X          (T, 251) float32   251 = 235 raw cols, azimuth's 16 deg -> 32 sin/cos
    task       (T,)     int64     argmax label (background-contaminated, see module doc)
    plateau    (T, C)   float32   flat-top per-class scores -- the BCE target
    prog_vec   (T, C)   float32   per-lane progress, 0-1
    mistake    (T,)     int64
    prog       (T,)     float32   scalar progress of the argmax lane, 0-1
    bg         (T,)     bool      True where NO task is annotated
    valid      (T,)     bool      finite features
    """
    drop_lift = DROP_LIFT if drop_lift is None else drop_lift
    d = torch.load(path, map_location="cpu", weights_only=False)
    f, L = d["features"], d["labels"]
    n_lanes = L["task_id_plateau_vector"].shape[1]
    keep = _keep_lanes(drop_lift, n_lanes)
    jk = [i for i, j in enumerate(FEATURE_JOINT_NAMES) if j not in DROP_JOINTS]
    ak = [i for i, j in enumerate(ANGLE_JOINTS) if j not in DROP_JOINTS]
    cols = []
    for k in PANEL_ORDER:
        a = f[k].numpy().astype(np.float32)
        if k == "polar_azimuth":                      # wraps at +/-180 -> sin/cos
            r = np.deg2rad(a[:, jk])
            cols += [np.sin(r), np.cos(r)]
        elif k == "ratios":                           # shared-denominator spikes
            cols.append(np.clip(a, 0.0, 4.0))
        elif k == "joint_angles":
            cols.append(a[:, ak])
        else:
            cols.append(a[:, jk])
    X = np.concatenate(cols, axis=1)

    full_plateau = L["task_id_plateau_vector"].numpy().astype(np.float32)
    plateau = full_plateau[:, keep]
    if n_lanes == 8:
        # The new corpus annotates idle time explicitly, so read it rather than
        # infer it. 18.2% of frames corpus-wide -- against 15.4% when it was
        # inferred from all-lanes-zero, i.e. the annotators marked idle spans
        # the old heuristic missed.
        bg = full_plateau[:, NEW_IDLE_LANE] >= 0.5
        if IDLE_AS_CLASS:
            # The idle lane is a trainable column, so it must not also be
            # masked out of the task metrics.
            bg = np.zeros(len(full_plateau), dtype=bool)
    else:
        # Old corpus: no idle annotation exists, so derive it. Verified to agree
        # with (task_id_prob == 0) on 100.00% of frames. Computed AFTER dropping
        # lanes so Lift-only frames correctly become background.
        bg = plateau.max(axis=1) <= 0.0

    return {
        "X": X,
        "task": L["task_id"].numpy().astype(np.int64),
        "plateau": plateau,
        # Eval-only tie-breaker. The plateau curve saturates at exactly 1.0, so
        # 18% of windows have two or more lanes tied at the max and a plain
        # argmax resolves every one of them toward the lowest class index --
        # which erases Place (index 6) entirely (9.6% of windows -> 0.5%).
        # The peak curve exists precisely to break these ties (labels.py:288-291).
        "peak": L["task_id_vector"].numpy().astype(np.float32)[:, keep],
        "prog_vec": L["task_progress_vector"].numpy().astype(np.float32)[:, keep] / 100.0,
        "mistake": L["mistake"].numpy().astype(np.int64),
        "prog": L["task_progress"].numpy().astype(np.float32) / 100.0,
        "bg": bg,
        "valid": np.isfinite(X).all(axis=1),
        "n_lanes": n_lanes,
    }


def subject_split(paths, test_uids, val_uids, aug_paths=None):
    """Split by subject. Augmented files, if given, join TRAIN only and only for
    uids that are already in train -- a mirrored copy of a held-out subject would
    be a direct leak."""
    tr = [p for p in paths if take_key(p)[0] not in test_uids | val_uids]
    va = [p for p in paths if take_key(p)[0] in val_uids]
    te = [p for p in paths if take_key(p)[0] in test_uids]
    if aug_paths:
        held = test_uids | val_uids
        tr = tr + [p for p in dedup_augmented(aug_paths)
                   if take_key(p)[0] not in held]
    return tr, va, te


class WindowView:
    """Behaves like an (N, win, D) float tensor without storing it.

    Supports what the training/eval code uses: len(), .shape, integer and
    slice indexing, and indexing with a LongTensor/ndarray of window ids.
    Indexing returns a real torch tensor of shape (B, win, D).
    """

    def __init__(self, flat, starts, win, stride=1):
        self.flat = torch.from_numpy(flat)
        self.starts = torch.from_numpy(np.asarray(starts, dtype=np.int64))
        self.win = win
        # offsets of the `win` samples inside one window; stride > 1 takes every
        # stride-th frame, e.g. stride 3 turns 30 fps footage into 10 fps input
        self._ar = torch.arange(win) * stride

    def __len__(self):
        return len(self.starts)

    @property
    def shape(self):
        return (len(self.starts), self.win, self.flat.shape[1])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.flat[int(self.starts[idx]) + self._ar]
        st = self.starts[idx]
        if st.dim() == 0:
            st = st.unsqueeze(0)
        return self.flat[st.unsqueeze(1) + self._ar]

    def to(self, *a, **k):
        raise TypeError("index a batch first, e.g. X[j].to(dev)")


class WindowSet:
    """Causal sliding windows. Every label is read at the window's LAST frame, so
    each window is a legitimate real-time prediction point."""

    def __init__(self, paths, win=120, hop=10, stats=None, drop_lift=None, stride=1):
        """win = number of samples the model sees; stride = frames between them.
        The window therefore covers (win-1)*stride+1 raw frames. stride=3 on this
        30 fps corpus reproduces the ~10 Hz rate at which the live loop appends
        feature vectors, so the recurrent model sees the same time step in
        training as in deployment."""
        self.win, self.hop, self.stride = win, hop, stride
        span = (win - 1) * stride + 1
        self.span = span
        self.drop_lift = DROP_LIFT if drop_lift is None else drop_lift
        self.task_names = task_names(self.drop_lift)   # refined below once loaded
        self.Xs, self.idx = [], []
        self.task, self.plateau, self.prog_vec = [], [], []
        self.mistake, self.prog, self.bg, self.peak = [], [], [], []
        self._n_lanes = 7
        for p in paths:
            d = load_take(p, self.drop_lift)
            self._n_lanes = d["n_lanes"]
            ti = len(self.Xs)
            self.Xs.append(d["X"])
            self.task.append(d["task"]); self.plateau.append(d["plateau"])
            self.prog_vec.append(d["prog_vec"]); self.mistake.append(d["mistake"])
            self.prog.append(d["prog"]); self.bg.append(d["bg"])
            self.peak.append(d["peak"])
            # a window is usable only if every one of its frames has features
            c = np.concatenate([[0], np.cumsum(d["valid"])])
            for e in range(span, len(d["X"]) + 1, hop):
                if c[e] - c[e - span] == span:
                    self.idx.append((ti, e))
        self.idx = np.array(self.idx, dtype=np.int64)
        # 8 raw lanes = the new schema ("No Related Task" added). The names follow
        # the switches actually in force, not the kept-lane count: with trimmed
        # Lift kept, 7 lanes can mean "Lift + 6 tasks" or "6 tasks + idle".
        if self._n_lanes == 8:
            self.task_names = task_names(drop_lift=self.drop_lift, new_schema=True,
                                         idle_as_class=IDLE_AS_CLASS)
            n_kept = self.plateau[0].shape[1] if self.plateau else len(self.task_names)
            assert n_kept == len(self.task_names), (n_kept, self.task_names)

        if stats is None:                              # TRAIN ONLY
            sample = np.concatenate([X[::7] for X in self.Xs])
            sample = sample[np.isfinite(sample).all(1)]
            mu, sd = sample.mean(0), sample.std(0)
            self.stats = (mu.astype(np.float32), (sd + 1e-6).astype(np.float32))
        else:
            self.stats = stats

    def __len__(self):
        return len(self.idx)

    def tensors(self):
        """Returns (X, y). X is a lazy WindowView, y a dict of target tensors.

        X used to be fully materialised as an (N, win, D) array. At hop 10 the
        windows overlap 12x, so that copied every frame twelve times: 20.9 GB
        and ~5 minutes of CPU per fold at win=120, and roughly double at
        win=240 -- enough to exhaust memory. The frames are now normalised
        once into one contiguous array and each batch gathers only its own
        windows. Values are bit-identical to the old path.
        """
        mu, sd = self.stats
        ti, e = self.idx[:, 0], self.idx[:, 1] - 1     # label at last frame
        gather = lambda src: np.array([src[a][b] for a, b in zip(ti, e)])
        y = {
            "task": torch.from_numpy(gather(self.task)),
            "plateau": torch.from_numpy(gather(self.plateau)),
            "prog_vec": torch.from_numpy(gather(self.prog_vec)),
            "mistake": torch.from_numpy(gather(self.mistake)),
            "prog": torch.from_numpy(gather(self.prog)),
            "bg": torch.from_numpy(gather(self.bg).astype(np.float32)),
            "peak": torch.from_numpy(gather(self.peak)),
        }
        offsets = np.concatenate([[0], np.cumsum([len(x) for x in self.Xs])[:-1]])
        flat = np.concatenate([(x - mu) / sd for x in self.Xs]).astype(np.float32)
        starts = offsets[self.idx[:, 0]] + self.idx[:, 1] - self.span
        return WindowView(flat, starts, self.win, self.stride), y

    def take_index(self):
        """Which take each window came from -- so smoothing never crosses takes."""
        return self.idx[:, 0]
