"""Leave-one-subject-out cross-validation.

Why this exists: bench/split_var.py measured a 0.085 macro-F1 spread across six
random choices of test subjects, with the same model and data. That is 3.3x the
entire gap between GRU / LSTM / LSTM-2 / TCN. A single split therefore cannot
distinguish two arms, and every comparison must be LOSO mean +/- sd.

One fold per subject: that subject is test, two more (fixed seed) are val, the
rest train. Augmented copies join train only, and only for training subjects.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from data import WindowSet, subject_split, TASK_NAMES, N_TASKS
from models import build
from engine import train_one, evaluate


def _parse_cw(s):
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru")
    ap.add_argument("--targets", default="vector", choices=["hard", "vector", "peak"])
    ap.add_argument("--aug", default="none", choices=["none", "mirror"])
    ap.add_argument("--data", default="original")
    ap.add_argument("--aug-dir", default="augmented_mirror")
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1,
                    help="frames between model samples; 3 = 10 fps, matching the live loop")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--folds", type=int, default=15, help="subjects to hold out")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pos-weight-cap", type=float, default=2.0,
                    help="clamp per-lane pos_weight; None-like 0 disables")
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--drop-joints", default="",
                    help="'lower' (hips, knees, ankles) or a comma list of joint names")
    ap.add_argument("--class-weight", default="",
                    help='per-class loss emphasis, e.g. "Screw=2,Clamp Coupling=2"')
    ap.add_argument("--bg-weight", type=float, default=0.2,
                    help="loss weight on the background head")
    ap.add_argument("--idle-as-class", action="store_true",
                    help="new corpus: train 'No Related Task' as a 7th lane "
                         "instead of routing it to the bg head")
    ap.add_argument("--panels", default="all",
                    help="'all' (16 panels, 251 dims), 'reduced' (6 panels, 89), "
                         "'reduced_vel' (6 panels + velocity x/y/z, 137), "
                         "'selected' (7 data-driven panels, 121)")
    ap.add_argument("--keep-lift", action="store_true",
                    help="keep the 7-class vocabulary (Lift included)")
    ap.add_argument("--save-models", action="store_true",
                    help="keep each fold's weights for event-level evaluation")
    a = ap.parse_args()

    import data as _D
    _D.IDLE_AS_CLASS = a.idle_as_class
    if a.drop_joints:
        _D.DROP_JOINTS = (set(_D.LOWER_BODY) if a.drop_joints == "lower"
                          else {j.strip() for j in a.drop_joints.split(",")})
        bad = _D.DROP_JOINTS - set(_D.FEATURE_JOINT_NAMES)
        if bad:
            raise SystemExit("unknown joints: %s" % sorted(bad))
    REDUCED = ["position_x_relative_to_pelvis",
               "position_y_relative_to_pelvis",
               "position_z_relative_to_pelvis",
               "polar_elevation", "joint_angles",
               "distance_from_center"]
    if a.panels == "reduced":
        # The six panels requested by the annotation team. Note this discards
        # every velocity/acceleration channel, including the vertical wrist
        # velocity that is Place's single most distinctive signal.
        _D.PANEL_ORDER = REDUCED
    elif a.panels == "selected":
        # Data-driven set from permutation importance on the 15 N1 models
        # (bench/feat_importance.py). Keeps the seven panels whose shuffling
        # cost >= 0.057 macro-F1; drops the six per-axis velocity/acceleration
        # panels (each <= 0.028 -- the magnitudes carry what matters), ratios
        # (0.013), and position_x / polar_elevation, which are 85-92%
        # linearly predictable from the panels kept. 121 dims.
        _D.PANEL_ORDER = ["joint_angles", "position_z_relative_to_pelvis",
                          "joint_speed", "distance_from_center", "polar_azimuth",
                          "position_y_relative_to_pelvis", "joint_acceleration"]
    elif a.panels == "reduced_vel":
        # The same six panels plus the three per-axis velocity panels, added at
        # the annotation team's request. velocity_z carries the upward wrist
        # motion that separates Place; joint_speed (the magnitude) is left out
        # because it is derivable from x/y/z.
        _D.PANEL_ORDER = REDUCED + ["joint_velocity_x", "joint_velocity_y",
                                    "joint_velocity_z"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))
    aug = sorted(glob.glob(os.path.join(a.aug_dir, "*.pt"))) if a.aug == "mirror" else None
    ALL = sorted({int(re.search(r"uid-(\d+)", os.path.basename(p)).group(1))
                  for p in paths})
    tag = a.tag or "%s_%s_%s" % (a.model, a.targets, a.aug)
    print("=== LOSO  model=%s targets=%s aug=%s  %d folds over %d subjects ==="
          % (a.model, a.targets, a.aug, min(a.folds, len(ALL)), len(ALL)))

    runs, t_start = [], time.time()
    for fi, test_uid in enumerate(ALL[:a.folds]):
        rest = [u for u in ALL if u != test_uid]
        va_u = set(np.random.RandomState(1000 + test_uid).permutation(rest)[:2].tolist())
        te_u = {test_uid}
        tr_p, va_p, te_p = subject_split(paths, te_u, va_u, aug_paths=aug)

        tr = WindowSet(tr_p, a.win, a.hop, stride=a.stride, drop_lift=not a.keep_lift)
        va = WindowSet(va_p, a.win, a.hop, stride=a.stride, stats=tr.stats, drop_lift=not a.keep_lift)
        te = WindowSet(te_p, a.win, a.hop, stride=a.stride, stats=tr.stats, drop_lift=not a.keep_lift)
        Xtr, ytr = tr.tensors(); Xva, yva = va.tensors(); Xte, yte = te.tensors()

        torch.manual_seed(fi)
        model = build(a.model, Xtr.shape[-1], n_tasks=ytr["plateau"].shape[1]).to(dev)
        model, best_val, state = train_one(model, Xtr, ytr, Xva, yva, dev, a.targets,
                                           a.epochs, a.bs, verbose=False,
                                           pos_weight_cap=(a.pos_weight_cap or None),
                                           focal_gamma=a.focal_gamma, names=tr.task_names,
                                           bg_weight=a.bg_weight,
                                           class_weight=_parse_cw(a.class_weight))
        if a.save_models:
            os.makedirs("bench/results/folds", exist_ok=True)
            torch.save({"state_dict": state, "stats": tr.stats, "dim": Xtr.shape[-1],
                        "model": a.model, "targets": a.targets,
                        "test_uids": sorted(te_u), "win": a.win, "hop": a.hop,
                        "stride": a.stride},
                       "bench/results/folds/%s_uid%02d.pth" % (tag, test_uid))
        r = evaluate(model, Xte, yte, dev, a.targets, tr.task_names)
        r.update(test_uid=test_uid, val_uids=sorted(va_u), val_macro_f1=best_val,
                 n_train_windows=len(Xtr), n_test_windows=len(Xte))
        runs.append(r)
        print("  fold %2d/%d  held-out uid %2d  macro-F1 %.3f  bal-acc %.3f"
              "  progMAE %.1f  (%d test win, %.0fs elapsed)"
              % (fi + 1, min(a.folds, len(ALL)), test_uid, r["macro_f1"],
                 r["balanced_acc"], r["progress_mae"], len(Xte), time.time() - t_start))

    f1 = np.array([r["macro_f1"] for r in runs])
    ba = np.array([r["balanced_acc"] for r in runs])
    mae = np.array([r["progress_mae"] for r in runs])
    print("\n=== %s: LOSO over %d folds ===" % (tag, len(runs)))
    print("  macro-F1      %.3f +/- %.3f   [%.3f, %.3f]"
          % (f1.mean(), f1.std(), f1.min(), f1.max()))
    print("  balanced-acc  %.3f +/- %.3f" % (ba.mean(), ba.std()))
    print("  progress MAE  %.1f +/- %.1f" % (mae.mean(), mae.std()))
    if "lane_ap" in runs[0]:
        for k in ["lane_ap", "lane_f1", "bg_f1"]:
            v = np.array([r[k] for r in runs])
            print("  %-13s %.3f +/- %.3f" % (k, v.mean(), v.std()))
    print("\n  per-class recall (mean +/- sd over folds):")
    # Iterate the class names the run actually produced. TASK_NAMES is the
    # module default (old corpus), so on the new corpus it names "Connect"
    # where the results say "Connect Cables" -- which crashed N1 after all 15
    # folds had trained, before the JSON was written.
    for n in runs[0]["per_class_recall"]:
        v = np.array([r["per_class_recall"][n] for r in runs])
        print("    %-12s %.2f +/- %.2f   [%.2f, %.2f]"
              % (n, v.mean(), v.std(), v.min(), v.max()))

    out = "bench/results/loso_%s.json" % tag
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"config": vars(a), "runs": runs,
               "summary": {"macro_f1_mean": float(f1.mean()),
                           "macro_f1_sd": float(f1.std()),
                           "balanced_acc_mean": float(ba.mean()),
                           "progress_mae_mean": float(mae.mean())}},
              open(out, "w"), indent=2)
    print("\n[saved] %s  (%.1f min total)" % (out, (time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
