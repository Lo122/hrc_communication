"""Event-level trigger evaluation -- the metric the robot actually cares about.

Per-frame macro-F1 answers "is this frame Align?". The runtime never asks that.
It asks "has the human finished raising the panel, so should I offer free-drive?"
-- a one-shot decision per assembly cycle. Measured corpus-wide:

  * 91% of Align spans begin while Lift is still active
  * they begin at the 39-54% point of the enclosing Lift span
  * 89% are preceded by the state (Lift + Place)

So the trigger is a TRANSITION detector, not a frame classifier. A frame-level
model that is only 0.51 on Align can still catch nearly every transition, because
it gets ~300 frames of evidence per event and only has to be right once.

Because the runtime ASKS permission (RobotTaskState.R_WAITING_FREE_DRIVE in
2_decision_making/task_manager.py), a false alarm costs one declined prompt while
a miss costs the assistance entirely. The operating point is therefore tuned for
recall, and this script reports the full latency/miss/false-alarm trade-off
rather than a single threshold.

Reported per arm:
  detection rate  -- fraction of true Align onsets caught within `tolerance`
  latency         -- seconds between the true onset and the model firing
                     (negative = fired early, which is GOOD for proactivity)
  false alarms    -- spurious firings per minute of footage
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from data import WindowSet, subject_split, load_take, TASK_NAMES, take_key
from models import build
from engine import predict

# Lane indices. With Lift dropped the vocabulary is
#   0 Pull, 1 Align, 2 Screw, 3 Connect, 4 Clamp, 5 Place
# With Lift kept it is the original 7-class order.
ALIGN_6, PLACE_6 = 1, 5
ALIGN_7, LIFT_7, PLACE_7 = 2, 1, 6
ALIGN, LIFT, PLACE = ALIGN_7, LIFT_7, PLACE_7      # legacy 7-class default


def lanes(n_classes):
    """-> (align, place) for the active vocabulary."""
    return (ALIGN_6, PLACE_6) if n_classes == 6 else (ALIGN_7, PLACE_7)


def gate_score(prob, prog, n_classes, smooth=9):
    """The deployable trigger signal: P(Align) x P(Place) x progress.

    Measured on the 7-class models (no retraining): 72% detection at 1.08 false
    alarms per event, against 75% / 2.19 for the raw Align score. Place gates
    better than Lift (active before 93% of Align onsets vs 90%) and, unlike Lift,
    is physically distinctive -- it is the only class with net upward wrist
    velocity. Progress adds the timing: it climbs from a median of 31 to 72 in the
    two seconds before the trigger should fire.
    """
    from scipy.ndimage import uniform_filter1d
    a, pl = lanes(n_classes)
    s = prob[:, a] * prob[:, pl] * np.clip(prog.max(axis=1), 0, 1)
    return uniform_filter1d(s, smooth)


def take_uid(p):
    return take_key(p)[0]


def onsets(mask, min_gap=30):
    """Start frame of each contiguous True run, merging runs closer than min_gap."""
    m = mask.astype(int)
    starts = list(np.flatnonzero(np.diff(m) == 1) + 1)
    if m[0]:
        starts = [0] + starts
    out = []
    for s in starts:
        if not out or s - out[-1] > min_gap:
            out.append(s)
    return np.array(out, dtype=int)


def eval_trigger(score, truth_onsets, thr, tol_s, fps, hop, persist=3):
    """Fire when `score` stays above `thr` for `persist` consecutive windows.
    -> (n_detected, latencies_in_seconds, n_false_alarms)"""
    hot = score >= thr
    if persist > 1:                      # debounce: require a sustained crossing
        k = np.ones(persist, dtype=int)
        hot = np.convolve(hot.astype(int), k, mode="same") >= persist
    fires = onsets(hot, min_gap=int(2 * fps / hop))
    tol_w = tol_s * fps / hop
    used, lat, det = set(), [], 0
    for t in truth_onsets:
        cand = [f for f in fires if abs(f - t) <= tol_w and f not in used]
        if cand:
            f = min(cand, key=lambda z: abs(z - t))
            used.add(f); det += 1
            lat.append((f - t) * hop / fps)
    return det, np.array(lat), len(fires) - len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["plateau_noaug"])
    ap.add_argument("--data", default="original")
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument("--tol", type=float, default=3.0, help="match window, seconds")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))

    print("Event-level free-drive trigger  (tolerance +/-%.0fs, permission-asking"
          " runtime -> favour recall)\n" % a.tol)
    for arm in a.arms:
        folds = sorted(glob.glob("bench/results/folds/%s_uid*.pth" % arm))
        if not folds:
            print("  [%s] no fold checkpoints -- rerun loso.py with --save-models\n" % arm)
            continue
        # Pool every fold: each contributes its own held-out subject, so the
        # event counts below are over all 15 subjects, never a single lucky split.
        agg = {t: [0, 0, [], 0, 0.0] for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
        for fp in folds:
            ck = torch.load(fp, map_location="cpu", weights_only=False)
            te_p = [p for p in paths if take_uid(p) in set(ck["test_uids"])]
            te = WindowSet(te_p, ck["win"], ck["hop"], stats=ck["stats"])
            X, y = te.tensors()
            model = build(ck["model"], ck["dim"]).to(dev)
            model.load_state_dict(ck["state_dict"])
            prob = torch.sigmoid(predict(model, X, dev)["task"]).numpy()
            ti = te.take_index()
            mins = len(prob) * ck["hop"] / 30 / 60
            for thr in agg:
                for t in np.unique(ti):
                    m = ti == t
                    truth = onsets(y["plateau"].numpy()[m][:, ALIGN] >= 0.5)
                    d, lat, fa = eval_trigger(prob[m][:, ALIGN], truth, thr,
                                              a.tol, 30, ck["hop"])
                    agg[thr][0] += d; agg[thr][1] += len(truth)
                    agg[thr][2].append(lat); agg[thr][3] += fa
                agg[thr][4] += mins
        print("  %s   (%d folds pooled)" % (arm, len(folds)))
        print("    %5s %10s %12s %12s %13s" % ("thr", "detected", "median lat",
                                               "early %", "false/min"))
        for thr in sorted(agg):
            D, T, L, FA, M = agg[thr]
            L = np.concatenate(L) if L else np.array([])
            early = 100 * (L < 0).mean() if len(L) else float("nan")
            print("    %5.1f %9.0f%% %11.1fs %11.0f%% %12.2f"
                  % (thr, 100 * D / max(T, 1), np.median(L) if len(L) else np.nan,
                     early, FA / max(M, 1e-9)))
        print("    %d true Align onsets across all held-out subjects\n"
              % agg[0.5][1])
        print("    %d true Align onsets across all held-out subjects\n"
              % agg[0.5][1])

if __name__ == "__main__":
    main()
