"""3D kinematic feature calculation for H36M-17 root-relative skeleton
sequences -- adapted from data_proc_2d's 2D feature pipeline (see
data_proc_2d/results/video_analysis_yolo_smoothed/source/feature_results.json
for the panel/feature-name schema this mirrors).

*** IMPORTANT: this is a FRESH reimplementation, not a port of the
original 2D code ***. data_proc_2d's actual feature_extraction.py/
feature_analysis.py source no longer exists in this repo (only Python-3.14
.pyc bytecode remains in data_proc_2d/src/__pycache__/, undecompilable with
the Python 3.11 interpreters available in this environment, and this repo
has no git history for it either). So the exact original formulas
(smoothing choice, precise "polar"/"distance from center" reference
frame, etc.) are unknown/unrecoverable -- what follows is a sensible,
clearly-documented reimplementation matching the JSON's feature NAMES and
panel groupings, not guaranteed numerically identical to the lost 2D
pipeline's output. If you find the original source later, this is the
module to reconcile against it.

Differences from the 2D schema, and why:
  - COCO's nose/eyes/ears don't exist in H36M-17 (pelvis, hips, knees,
    ankles, spine, thorax, neck, head, shoulders, elbows, wrists instead)
    -- per-joint panels below iterate H36M's joint set, not COCO's.
  - pelvis (joint 0) is excluded from every per-joint panel: this
    pipeline's skeletons are root-relative (pelvis forced to (0,0,0), see
    skeleton_utils/motionbert_lifter.py), so pelvis's own velocity/
    position/etc. is trivially always zero -- not a meaningful feature.
  - 2D's single "polar angle" (one angle is enough to describe a 2D
    direction) becomes an azimuth/elevation PAIR here (two angles are
    needed to describe a 3D direction) -- see compute_polar().
  - Joint angles here are computed from true 3D vectors (dot-product/
    arccos), not a 2D image-plane projection -- more accurate, not just a
    stand-in, since 3D avoids the projection distortion an arm moving
    toward/away from the camera would cause in a 2D angle estimate.

Input convention: (T, 17, 3) root-relative H36M skeletons, meters, this
project's (X right, Y forward/depth, Z up) axis convention -- NaN rows
(no detection that frame) propagate as NaN into every derived feature for
that frame rather than being interpolated over.

Why Savitzky-Golay for velocity/acceleration (not a plain frame-to-frame
difference): a raw diff is a high-pass operation -- a single bad frame from
the lifter (a momentary bad detection from occlusion/motion blur/fast
motion) becomes a huge one-frame velocity spike, and differencing AGAIN for
acceleration (a plain second difference) amplifies that spike further
still. In practice this showed up as isolated ~10 m/s / ~300 m/s^2 spikes
against an otherwise near-zero baseline -- not real human motion (an
adult's hand doesn't hit 30g accelerations during a task-progress-relevant
reach). Savitzky-Golay fits a local polynomial to a short window of the
RAW position signal and differentiates that polynomial analytically, which
is both smooth (spreads a single glitch's influence gently across its
window instead of concentrating it into a spike) and, unlike a naive
moving-average-then-diff, doesn't introduce a phase lag/systematic
underestimate of genuine fast motion. See _savgol_derivative()'s docstring
for how NaN gaps (missing detections) are handled.
"""
import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter

H36M_JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "neck", "head", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]
(PELVIS, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE,
 SPINE, THORAX, NECK, HEAD,
 L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST) = range(17)

# Joints carrying real signal in a root-relative (pelvis-centered) frame --
# pelvis itself is excluded from every per-joint panel (see module docstring).
FEATURE_JOINTS = [i for i in range(17) if i != PELVIS]


def _joint_name(i):
    return H36M_JOINT_NAMES[i]


def _valid_runs(mask):
    """mask: (T,) bool. Returns [(start, stop), ...] for each maximal
    contiguous True run (half-open, like a slice)."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


def _causal_run(run_positions, fps, window_length, polyorder, deriv):
    """One contiguous non-NaN run -> the deriv-th derivative at each frame,
    fitted over that frame and the ones BEFORE it only.

    Reproduces skeleton3d_pipeline.StreamingH36MFeatureExtractor's
    _fit_derivative exactly, which is the entire point, so it copies three
    behaviours a plain "shift the window" approximation would get wrong:

      - the window GROWS. Live, the extractor's deque is not full yet at the
        start, so frame i is fitted over min(i+1, window_length) samples, not
        over window_length padded ones.
      - below polyorder + 2 samples it does not fit at all, returning the raw
        latest position (deriv=0) or zero (deriv>0). Same margin the centered
        path uses, and for the same reason: fewer samples than that gives a
        degenerate exact-interpolation polynomial with unstable derivatives.
      - the fit is evaluated at the LATEST sample, not the window centre.

    savgol_coeffs(..., pos=n-1, use="dot") is the fixed-delta equivalent of
    live's np.polyfit against real timestamps: identical for the uniformly
    spaced samples an offline video decode produces, and much cheaper. Live
    keeps the polyfit form because its frame spacing is NOT uniform (see that
    class's docstring); here it is, by construction.
    """
    run_len = run_positions.shape[0]
    out = np.empty_like(run_positions)
    coeff_cache = {}
    for i in range(run_len):
        n = min(i + 1, window_length)
        if n < polyorder + 2:
            out[i] = run_positions[i] if deriv == 0 else 0.0
            continue
        coeffs = coeff_cache.get(n)
        if coeffs is None:
            coeffs = savgol_coeffs(n, polyorder, deriv=deriv, delta=1.0 / fps,
                                   pos=n - 1, use="dot")
            coeff_cache[n] = coeffs
        window = run_positions[i - n + 1:i + 1]          # (n, 17, 3), chronological
        out[i] = np.tensordot(coeffs, window, axes=(0, 0))
    return out


def _savgol_derivative(positions, fps, window_length, polyorder, deriv, causal=False):
    """positions: (T, 17, 3), may contain fully-NaN rows (missing
    detections, see module docstring). Returns the deriv-th derivative via
    Savitzky-Golay, applied independently within each contiguous run of
    non-NaN frames (a NaN gap would otherwise smear across the whole
    filter window, corrupting frames well outside the actual gap) --
    frames in a run shorter than the window (too little data to fit a
    reliable polynomial to, e.g. right after a lost-and-reacquired
    detection) are left NaN rather than guessed at.

    causal=True fits over TRAILING windows instead of centered ones, which
    is what the live pipeline necessarily does -- it cannot see future
    frames. Left False, the training features are produced by an estimator
    deployment cannot reproduce: a centered window at window_length=9 uses
    4 frames of lookahead, i.e. 400 ms at the 10 Hz the live loop actually
    runs at. The centered fit is the better ESTIMATOR (no phase lag, lower
    variance); it is simply not one inference can run, and matching the two
    sides beats being individually better on either. See _causal_run for
    exactly what is reproduced."""
    out = np.full_like(positions, np.nan)
    valid = ~np.isnan(positions).any(axis=(1, 2))
    for start, stop in _valid_runs(valid):
        run_len = stop - start
        if causal:
            out[start:stop] = _causal_run(
                positions[start:stop], fps, window_length, polyorder, deriv)
            continue
        wl = min(window_length, run_len if run_len % 2 == 1 else run_len - 1)
        if wl < polyorder + 2:
            continue  # too short a run to fit/differentiate reliably -- leave NaN
        out[start:stop] = savgol_filter(
            positions[start:stop], window_length=wl, polyorder=polyorder,
            deriv=deriv, delta=1.0 / fps, axis=0)
    return out


def compute_velocity(positions, fps, window_length=9, polyorder=3, causal=False):
    """positions: (T, 17, 3). Returns (T, 17, 3) velocity, m/s -- see
    module docstring's "Why Savitzky-Golay" note. window_length (odd,
    frames) / polyorder are the smoothing-vs-responsiveness knobs: at 30fps
    the default window_length=9 is a ~0.3s smoothing window -- widen it
    for noisier input, narrow it if genuinely fast motion is being
    over-smoothed. causal: see _savgol_derivative."""
    return _savgol_derivative(positions, fps, window_length, polyorder, deriv=1,
                              causal=causal)


def compute_acceleration(positions, fps, window_length=9, polyorder=3, causal=False):
    """positions: (T, 17, 3) -- note this takes POSITIONS directly, NOT
    compute_velocity()'s output: Savitzky-Golay differentiates the
    ORIGINAL position signal twice in one analytic step, which is far
    better-behaved than differencing an already-differenced (and thus
    already noise-amplified) velocity signal a second time -- see module
    docstring. causal: see _savgol_derivative."""
    return _savgol_derivative(positions, fps, window_length, polyorder, deriv=2,
                              causal=causal)


def compute_smoothed_positions(positions, fps, window_length=9, polyorder=3, causal=False):
    """positions: (T, 17, 3). Returns the deriv=0 (smoothed, not
    differentiated) Savitzky-Golay fit -- used for every feature computed
    DIRECTLY from position (joint angles, polar azimuth/elevation, ratios,
    distance-from-center, the "Position" panel), for the same reason
    velocity/acceleration are smoothed: a single bad-lifter frame otherwise
    leaks a one-frame glitch into these features too. Unlike velocity/
    acceleration, a bounded quantity like an angle can't blow up from a
    tiny position glitch the way multiplying by fps/fps^2 does -- so this
    is a smaller, but not zero, effect. Smoothing POSITION (not the angles
    themselves) also sidesteps the failure mode of naively averaging a
    wrap-around angle (e.g. azimuth flipping between +179deg/-179deg would
    average toward 0deg, not +-180deg, if smoothed directly). causal: see
    _savgol_derivative."""
    return _savgol_derivative(positions, fps, window_length, polyorder, deriv=0,
                              causal=causal)


def compute_polar(positions):
    """positions: (T, 17, 3), root-relative (pelvis at origin). Returns
    (azimuth_deg, elevation_deg), each (T, 17): azimuth = atan2(y, x)
    (direction within the horizontal XY-plane), elevation = atan2(z,
    sqrt(x^2+y^2)) (angle above/below horizontal) -- the 3D generalization
    of the 2D pipeline's single "polar angle" (only one angle is needed to
    describe a 2D direction; 3D needs two)."""
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return azimuth, elevation


def _angle_between(v1, v2):
    """v1, v2: (T, 3). Returns (T,) angle in degrees between the two
    vectors at each frame (dot-product/arccos). NaN where either vector is
    ~zero-length (degenerate, e.g. missing detection that frame)."""
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = n1 * n2
    cos_angle = np.divide(np.sum(v1 * v2, axis=-1), denom,
                           out=np.full(denom.shape, np.nan), where=denom > 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_joint_angles(positions):
    """positions: (T, 17, 3). Returns dict of 9 (T,) angle-in-degrees
    arrays: the angle AT each named joint between its two adjacent bone
    vectors (standard 3-point joint-angle convention)."""
    p = positions
    return {
        "left_elbow_angle_deg": _angle_between(
            p[:, L_SHOULDER] - p[:, L_ELBOW], p[:, L_WRIST] - p[:, L_ELBOW]),
        "right_elbow_angle_deg": _angle_between(
            p[:, R_SHOULDER] - p[:, R_ELBOW], p[:, R_WRIST] - p[:, R_ELBOW]),
        "left_shoulder_angle_deg": _angle_between(
            p[:, THORAX] - p[:, L_SHOULDER], p[:, L_ELBOW] - p[:, L_SHOULDER]),
        "right_shoulder_angle_deg": _angle_between(
            p[:, THORAX] - p[:, R_SHOULDER], p[:, R_ELBOW] - p[:, R_SHOULDER]),
        "left_hip_angle_deg": _angle_between(
            p[:, SPINE] - p[:, L_HIP], p[:, L_KNEE] - p[:, L_HIP]),
        "right_hip_angle_deg": _angle_between(
            p[:, SPINE] - p[:, R_HIP], p[:, R_KNEE] - p[:, R_HIP]),
        "left_knee_angle_deg": _angle_between(
            p[:, L_HIP] - p[:, L_KNEE], p[:, L_ANKLE] - p[:, L_KNEE]),
        "right_knee_angle_deg": _angle_between(
            p[:, R_HIP] - p[:, R_KNEE], p[:, R_ANKLE] - p[:, R_KNEE]),
        "neck_angle_deg": _angle_between(
            p[:, THORAX] - p[:, NECK], p[:, HEAD] - p[:, NECK]),
    }


def compute_ratios(positions, shoulder_width=None):
    """positions: (T, 17, 3). Returns dict with elbow_over_shoulder_ratio
    and wrist_over_shoulder_ratio (T,): mean(left, right) distance of that
    joint from the shoulder midpoint, divided by shoulder width -- a
    scale-invariant "how far the arm reaches relative to this person's own
    shoulder width" measure (naming/intent inferred from the JSON schema,
    see module docstring's reimplementation caveat).

    The denominator is this person's MEDIAN shoulder width over the clip, not
    the per-frame value. Shoulder width is an anatomical constant -- "this
    person's own shoulder width" is a property of the person, not of the
    frame -- so any per-frame variation is monocular-lifter noise, and
    dividing by it injects that noise into the feature on every frame.

    It also fails catastrophically, not just noisily. When the subject turns
    edge-on to the camera, both shoulders project to almost the same 2D
    point, the lifter has little evidence for their separation in depth, and
    it can collapse them onto each other. On cam-07_uid-02_take-01 at 193.9s
    the shoulder width fell from 0.251m to 0.027m -- a tenth of normal --
    while the wrist and elbow distances stayed flat at 0.280m/0.228m. The
    arms had not moved at all, yet wrist_over_shoulder_ratio spiked to 10.4
    and the elbow ratio to 8.5 (baseline ~1.0 for both; they spike together
    precisely because they share this denominator).

    BoneLengthConstraintFilter cannot prevent that: the H36M bone tree
    constrains thorax->l_shoulder and thorax->r_shoulder, but there is no
    l_shoulder--r_shoulder edge, so both shoulders can sit at their correct
    distance from the thorax while folded onto the same side -- every bone
    length valid, shoulder width ~0.

    shoulder_width: pass a scalar to use that as the denominator instead of
    this clip's own median. This exists because the clip median is NOT
    computable live -- it is a whole-clip statistic, the same class of
    lookahead as a centered (rather than causal) Savitzky-Golay window. A
    streaming caller must supply a CAUSAL estimate of the same anatomical
    quantity instead; see skeleton3d_pipeline.py's
    StreamingH36MFeatureExtractor, which keeps a running median over the
    widths it has seen so far. Note the residual train/inference gap that
    leaves: offline every frame of a clip divides by one converged median,
    while live the denominator is still settling over the first seconds of a
    session. It converges, but it is not identical -- the same caveat that
    applies to bone_length_targets/body_scale_m in
    generate_lstm_training_data.py, which are likewise read off after the
    whole clip.
    """
    shoulder_mid = (positions[:, L_SHOULDER] + positions[:, R_SHOULDER]) / 2.0

    if shoulder_width is None:
        per_frame_width = np.linalg.norm(
            positions[:, L_SHOULDER] - positions[:, R_SHOULDER], axis=-1)
        per_frame_width = np.where(per_frame_width > 1e-8, per_frame_width, np.nan)
        shoulder_width = np.nanmedian(per_frame_width) if per_frame_width.size else np.nan

    shoulder_width = float(shoulder_width)
    if not np.isfinite(shoulder_width) or shoulder_width <= 1e-8:
        shoulder_width = np.nan  # degenerate clip; ratios become NaN, not inf

    def _mean_dist_ratio(joint_l, joint_r):
        d_l = np.linalg.norm(positions[:, joint_l] - shoulder_mid, axis=-1)
        d_r = np.linalg.norm(positions[:, joint_r] - shoulder_mid, axis=-1)
        return (d_l + d_r) / 2.0 / shoulder_width

    return {
        "elbow_over_shoulder_ratio": _mean_dist_ratio(L_ELBOW, R_ELBOW),
        "wrist_over_shoulder_ratio": _mean_dist_ratio(L_WRIST, R_WRIST),
    }


def compute_distance_from_center_ratio(positions):
    """positions: (T, 17, 3), root-relative (pelvis at origin). Returns
    (T, 17): each joint's distance from the pelvis, normalized by torso
    height (pelvis-to-thorax distance) for scale invariance across
    subjects/videos."""
    torso_height = np.linalg.norm(positions[:, THORAX] - positions[:, PELVIS], axis=-1)
    torso_height = np.where(torso_height > 1e-8, torso_height, np.nan)
    dist = np.linalg.norm(positions, axis=-1)  # already relative to pelvis=origin
    return dist / torso_height[:, None]


def compute_all_features(positions, fps, window_length=9, polyorder=3, causal=False):
    """positions: (T, 17, 3) root-relative H36M skeleton sequence, meters.
    window_length/polyorder: passed to compute_velocity/compute_acceleration's
    Savitzky-Golay filter -- see those functions' docstrings.

    causal=True computes every derived panel with TRAILING windows, matching
    what skeleton3d_pipeline.StreamingH36MFeatureExtractor must do live. Set
    it for any data a deployed model will be trained on; see
    _savgol_derivative for why the centered default cannot be reproduced at
    inference time.

    Returns (feature_dict, panel_groups):
      feature_dict: {column_name: (T,) ndarray} -- every scalar feature.
      panel_groups: {panel_title: [column_name, ...]} -- mirrors
      data_proc_2d's feature_results.json structure (panel_titles +
      feature_dataframes), adapted for H36M's joint set (see module
      docstring). Iterating panel_groups in insertion order reproduces a
      stable panel ordering across runs.
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocity = compute_velocity(positions, fps, window_length, polyorder, causal)        # (T, 17, 3)
    acceleration = compute_acceleration(positions, fps, window_length, polyorder, causal)  # (T, 17, 3)
    # Smoothed (not raw) position for everything computed directly from
    # position -- see compute_smoothed_positions()'s docstring on why
    # angles/polar/ratios/distance-from-center need this too, just via a
    # gentler mechanism than velocity/acceleration's amplified spikes.
    smoothed_positions = compute_smoothed_positions(positions, fps, window_length, polyorder,
                                                    causal)
    azimuth, elevation = compute_polar(smoothed_positions)         # (T, 17) each
    joint_angles = compute_joint_angles(smoothed_positions)
    ratios = compute_ratios(smoothed_positions)
    dist_ratio = compute_distance_from_center_ratio(smoothed_positions)  # (T, 17)

    speed = np.linalg.norm(velocity, axis=-1)          # (T, 17)
    accel_mag = np.linalg.norm(acceleration, axis=-1)   # (T, 17)

    feature_dict = {}
    panel_groups = {}

    def _add_panel(title, columns_values):
        panel_groups[title] = list(columns_values.keys())
        feature_dict.update(columns_values)

    _add_panel("Joint Speed", {
        f"{_joint_name(j)}_speed": speed[:, j] for j in FEATURE_JOINTS})
    _add_panel("Joint Acceleration", {
        f"{_joint_name(j)}_acceleration": accel_mag[:, j] for j in FEATURE_JOINTS})
    for axis_idx, axis_name in enumerate("xyz"):
        _add_panel(f"Joint Velocity {axis_name.upper()}", {
            f"{_joint_name(j)}_velocity_{axis_name}": velocity[:, j, axis_idx]
            for j in FEATURE_JOINTS})
    for axis_idx, axis_name in enumerate("xyz"):
        _add_panel(f"Joint Acceleration {axis_name.upper()}", {
            f"{_joint_name(j)}_acceleration_{axis_name}": acceleration[:, j, axis_idx]
            for j in FEATURE_JOINTS})
    for axis_idx, axis_name in enumerate("xyz"):
        _add_panel(f"Position {axis_name.upper()} (relative to pelvis)", {
            f"{_joint_name(j)}_position_{axis_name}": smoothed_positions[:, j, axis_idx]
            for j in FEATURE_JOINTS})
    _add_panel("Polar Azimuth", {
        f"{_joint_name(j)}_azimuth_deg": azimuth[:, j] for j in FEATURE_JOINTS})
    _add_panel("Polar Elevation", {
        f"{_joint_name(j)}_elevation_deg": elevation[:, j] for j in FEATURE_JOINTS})
    _add_panel("Joint Angles", joint_angles)
    _add_panel("Ratios", ratios)
    _add_panel("Distance from Center", {
        f"{_joint_name(j)}_distance_from_center_ratio": dist_ratio[:, j]
        for j in FEATURE_JOINTS})

    return feature_dict, panel_groups
