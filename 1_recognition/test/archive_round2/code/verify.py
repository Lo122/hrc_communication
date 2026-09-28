"""Assertions from the plan's Verification section. Run before any training."""
import glob, os, sys, re
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from data import load_take, dedup_augmented, subject_split, cam_take_key, take_key

ok = True
def check(name, cond, detail=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("   " + detail if detail else ""))
    ok = ok and cond

print("1. background mask is exact (7-class: plateau.max==0 <-> task_id_prob==0)")
agree = tot = 0
for p in sorted(glob.glob("original/*.pt"))[:25]:
    d = load_take(p, drop_lift=False)     # the identity only holds for all 7 lanes
    prob = torch.load(p, map_location="cpu", weights_only=False)["labels"]["task_id_prob"].numpy()
    agree += int((d["bg"] == (prob <= 0.0)).sum()); tot += len(prob)
check("exact agreement", agree == tot, "%.2f%% of %d frames" % (100*agree/tot, tot))

print("1b. dropping Lift only ever ADDS background, never removes it")
grew = shrank = newly = tot6 = 0
for p in sorted(glob.glob("original/*.pt"))[:25]:
    d7 = load_take(p, drop_lift=False); d6 = load_take(p, drop_lift=True)
    shrank += int((d7["bg"] & ~d6["bg"]).sum())
    newly += int((d6["bg"] & ~d7["bg"]).sum()); tot6 += len(d6["bg"])
check("no frame loses background status", shrank == 0,
      "%d frames newly background (%.2f%%)" % (newly, 100*newly/tot6))

print("\n2. progress gather identity  prog_vec[t, task[t]] == prog[t]")
worst = 0.0
for p in sorted(glob.glob("original/*.pt"))[:25]:
    d = load_take(p, drop_lift=False)     # task_id indexes the full 7-lane vector
    sel = d["prog_vec"][np.arange(len(d["task"])), d["task"]]
    worst = max(worst, float(np.abs(sel - d["prog"]).max()))
check("max abs diff == 0", worst == 0.0, "max diff %.6f" % worst)

print("\n3. no augmentation leak")
orig = sorted(glob.glob("original/*.pt"))
aug = sorted(glob.glob("augmented_mirror/*.pt"))
te_u, va_u = {13, 14, 15}, {11, 12}
tr, va, te = subject_split(orig, te_u, va_u, aug_paths=aug)
tr_u = {take_key(p)[0] for p in tr}
check("train uids disjoint from val", not (tr_u & va_u), "train %s" % sorted(tr_u))
check("train uids disjoint from test", not (tr_u & te_u))
check("no aug file in val", not any("augmented" in p for p in va))
check("no aug file in test", not any("augmented" in p for p in te))
aug_in_train = [p for p in tr if "augmented" in p]
check("no held-out uid among aug files in train",
      not ({take_key(p)[0] for p in aug_in_train} & (te_u | va_u)),
      "%d aug files in train" % len(aug_in_train))

print("\n4. dedup keeps exactly one mirror file per (cam, uid, take)")
dd = dedup_augmented(aug)
keys = [cam_take_key(p) for p in dd]
check("one per recording", len(keys) == len(set(keys)), "%d aug -> %d after dedup" % (len(aug), len(dd)))

print("\n5. mirror labels match their source take")
src = "original/features__cam-05_uid-01_take-02.pt"
mir = "augmented_mirror/features__cam-05_uid-01_take-02_aug-01.pt"
if os.path.exists(mir):
    a, b = load_take(src), load_take(mir)
    n = min(len(a["task"]), len(b["task"]))
    check("task_id matches on overlap", bool((a["task"][:n] == b["task"][:n]).all()),
          "%d frames (mirror is untrimmed: %d vs %d)" % (n, len(b["task"]), len(a["task"])))
    check("plateau matches on overlap", bool(np.allclose(a["plateau"][:n], b["plateau"][:n])))

print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
