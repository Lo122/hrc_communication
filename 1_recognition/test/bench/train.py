"""Train + evaluate one model on a single subject-disjoint split.

Reports macro-F1 and per-class recall, never accuracy alone -- Screw is ~48% of
frames, so accuracy rewards a model that always says Screw.

NOTE: a single split is only for quick iteration. Measured split-to-split spread
is 0.085 macro-F1 (bench/split_var.py), which is 3.3x the gap between models, so
any comparison between arms must use bench/loso.py instead.
"""
import argparse, glob, json, os, sys, time
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
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-uids", default="13,14,15")
    ap.add_argument("--val-uids", default="11,12")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pos-weight-cap", type=float, default=2.0)
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
    ap.add_argument("--keep-lift", action="store_true")
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
    te_u = {int(x) for x in a.test_uids.split(",")}
    va_u = {int(x) for x in a.val_uids.split(",")}
    paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))
    aug = sorted(glob.glob(os.path.join(a.aug_dir, "*.pt"))) if a.aug == "mirror" else None
    tr_p, va_p, te_p = subject_split(paths, te_u, va_u, aug_paths=aug)
    n_aug = sum(1 for p in tr_p if "augmented" in p)
    print("[split] train %d files (%d augmented) / val %d / test %d"
          "  test uids %s, val uids %s"
          % (len(tr_p), n_aug, len(va_p), len(te_p), sorted(te_u), sorted(va_u)))

    t0 = time.time()
    tr = WindowSet(tr_p, a.win, a.hop, stride=a.stride, drop_lift=not a.keep_lift)
    va = WindowSet(va_p, a.win, a.hop, stride=a.stride, stats=tr.stats, drop_lift=not a.keep_lift)
    te = WindowSet(te_p, a.win, a.hop, stride=a.stride, stats=tr.stats, drop_lift=not a.keep_lift)
    Xtr, ytr = tr.tensors(); Xva, yva = va.tensors(); Xte, yte = te.tensors()
    print("[data] windows train %d val %d test %d  dim %d  (%.0fs)"
          % (len(Xtr), len(Xva), len(Xte), Xtr.shape[-1], time.time() - t0))
    print("[data] background frames in test: %.1f%%"
          % (100 * yte["bg"].numpy().mean()))

    model = build(a.model, Xtr.shape[-1], n_tasks=ytr["plateau"].shape[1]).to(dev)
    model, best_val, state = train_one(model, Xtr, ytr, Xva, yva, dev, a.targets,
                                       a.epochs, a.bs, a.lr,
                                       pos_weight_cap=(a.pos_weight_cap or None),
                                       focal_gamma=a.focal_gamma, names=tr.task_names,
                                           bg_weight=a.bg_weight,
                                           class_weight=_parse_cw(a.class_weight))
    res = evaluate(model, Xte, yte, dev, a.targets, tr.task_names)
    res.update(model=a.model, targets=a.targets, aug=a.aug, val_macro_f1=best_val,
               test_uids=sorted(te_u), n_params=sum(p.numel() for p in model.parameters()))

    print("\n=== TEST (held-out subjects, background-masked) ===")
    print("  macro-F1      %.3f" % res["macro_f1"])
    print("  balanced-acc  %.3f   (chance %.3f)" % (res["balanced_acc"], 1 / N_TASKS))
    print("  accuracy      %.3f   <- do not report this alone" % res["accuracy"])
    if "lane_ap" in res:
        print("  lane AP       %.3f   (multi-label view)" % res["lane_ap"])
        print("  lane F1       %.3f" % res["lane_f1"])
        print("  background F1 %.3f" % res["bg_f1"])
    print("  mistake F1    %.3f" % res["mistake_f1"])
    print("  progress MAE  %.1f / 100" % res["progress_mae"])
    print("  per-class recall:")
    for k, v in res["per_class_recall"].items():
        print("    %-12s %.2f" % (k, v))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        torch.save({"state_dict": state, "stats": tr.stats, "config": vars(a),
                    "dim": Xtr.shape[-1]}, a.out.replace(".json", ".pth"))
        print("\n[saved] %s + .pth" % a.out)


if __name__ == "__main__":
    main()
