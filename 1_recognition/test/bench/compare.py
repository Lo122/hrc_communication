"""Final comparison table across LOSO arms, with a paired significance test.

Folds are paired: every arm holds out the same 15 subjects, so the per-fold
differences can be tested directly rather than comparing two means with
overlapping error bars. With 15 folds and sd ~0.03, only differences above
roughly 0.02 are resolvable -- state that rather than ranking noise.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import TASK_NAMES

ARMS = [
    ("A_gru_hard", "baseline  hard argmax"),
    ("plateau_noaug", "plateau   no aug"),
    ("plateau_aug", "plateau   + mirror"),
    ("peak_noaug", "peak      no aug"),
    ("peak_aug", "peak      + mirror"),
]

loaded = {}
for tag, label in ARMS:
    p = "bench/results/loso_%s.json" % tag
    if os.path.exists(p):
        loaded[tag] = (label, json.load(open(p)))

if not loaded:
    print("no LOSO results yet"); sys.exit(0)

print("=" * 78)
print("LOSO comparison -- %d folds, one held-out subject each" % len(
    next(iter(loaded.values()))[1]["runs"]))
print("=" * 78)
print("%-26s %14s %14s %12s" % ("arm", "macro-F1", "bal-acc", "progMAE"))
for tag, (label, d) in loaded.items():
    f1 = np.array([r["macro_f1"] for r in d["runs"]])
    ba = np.array([r["balanced_acc"] for r in d["runs"]])
    mae = np.array([r["progress_mae"] for r in d["runs"]])
    print("%-26s  %.3f +/- %.3f  %.3f +/- %.3f  %5.1f +/- %.1f"
          % (label, f1.mean(), f1.std(), ba.mean(), ba.std(), mae.mean(), mae.std()))

print("\nper-class recall (mean +/- sd over folds)")
print("%-26s " % "arm" + " ".join("%12s" % n[:11] for n in TASK_NAMES))
for tag, (label, d) in loaded.items():
    cells = []
    for n in TASK_NAMES:
        v = np.array([r["per_class_recall"][n] for r in d["runs"]])
        cells.append("%5.2f+-%.2f" % (v.mean(), v.std()))
    print("%-26s " % label + " ".join("%12s" % c for c in cells))

# paired comparisons on the shared folds
def paired(a, b):
    ra = {r["test_uid"]: r["macro_f1"] for r in loaded[a][1]["runs"]}
    rb = {r["test_uid"]: r["macro_f1"] for r in loaded[b][1]["runs"]}
    common = sorted(set(ra) & set(rb))
    d = np.array([rb[u] - ra[u] for u in common])
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
    except Exception:
        p = float("nan")
    return d, p, common

print("\npaired per-fold differences (positive = second arm better)")
for a, b in [("plateau_noaug", "plateau_aug"),
             ("peak_noaug", "peak_aug"),
             ("plateau_noaug", "peak_noaug"),
             ("plateau_aug", "peak_aug"),
             ("A_gru_hard", "plateau_noaug")]:
    if a in loaded and b in loaded:
        d, p, common = paired(a, b)
        win = int((d > 0).sum())
        print("  %-22s -> %-22s  mean %+0.3f  sd %.3f  wins %d/%d  wilcoxon p=%.3f"
              % (a, b, d.mean(), d.std(), win, len(d), p))
