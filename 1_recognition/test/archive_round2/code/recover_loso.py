"""Rebuild bench/results/loso_<tag>.json from saved fold weights.

For runs that trained all folds but died before writing the JSON. Every fold
checkpoint stores the best-validation weights and the training-set norm stats,
so re-evaluating on each held-out subject reproduces the test metrics exactly.
What cannot be recovered is val_macro_f1 (the validation score at the chosen
epoch); it is recorded as null.
"""
import argparse, glob, json, os, re, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import WindowSet, take_key
from models import build
from engine import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="hrc_communication/other/original")
ap.add_argument("--targets", default="peak")
ap.add_argument("--config", default="{}", help="JSON of the original CLI args")
a = ap.parse_args()
cfg = json.loads(a.config)
D.IDLE_AS_CLASS = bool(cfg.get("idle_as_class", False))

dev = "cuda" if torch.cuda.is_available() else "cpu"
paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))
runs = []
for fp in sorted(glob.glob("bench/results/folds/%s_uid*.pth" % a.tag)):
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    uid = ck["test_uids"][0]
    te_p = [p for p in paths if take_key(p)[0] == uid]
    te = WindowSet(te_p, ck["win"], ck["hop"], stats=ck["stats"], stride=ck.get("stride", 1))
    X, y = te.tensors()
    m = build(ck["model"], ck["dim"], n_tasks=y["plateau"].shape[1]).to(dev)
    m.load_state_dict(ck["state_dict"])
    r = evaluate(m, X, y, dev, a.targets, te.task_names)
    rest = [u for u in range(1, 16) if u != uid]
    r.update(test_uid=uid,
             val_uids=sorted(np.random.RandomState(1000 + uid).permutation(rest)[:2].tolist()),
             val_macro_f1=None, n_test_windows=len(X), recovered=True)
    runs.append(r)
    print("  uid %2d  macro-F1 %.3f" % (uid, r["macro_f1"]))

f1 = np.array([r["macro_f1"] for r in runs])
out = "bench/results/loso_%s.json" % a.tag
json.dump({"config": cfg, "runs": runs, "recovered_from_fold_weights": True,
           "summary": {"macro_f1_mean": float(f1.mean()), "macro_f1_sd": float(f1.std()),
                       "balanced_acc_mean": float(np.mean([r["balanced_acc"] for r in runs])),
                       "progress_mae_mean": float(np.mean([r["progress_mae"] for r in runs]))}},
          open(out, "w"), indent=2)
print("[recovered] %s   macro-F1 %.3f +/- %.3f  (%d folds)" % (out, f1.mean(), f1.std(), len(runs)))
