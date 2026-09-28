"""How much does the reported score depend on WHICH subjects land in test?

Same model, same hyperparameters, same data -- only the random choice of the
3 test subjects and 2 val subjects changes. If the spread across seeds is large
relative to the gap between models, then the model comparison is noise.
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from data import WindowSet, subject_split, TASK_NAMES, N_TASKS
from models import build
from train import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gru")
ap.add_argument("--data", default="original")
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--win", type=int, default=120)
ap.add_argument("--hop", type=int, default=10)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--out", default="bench/results/split_variance.json")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))
ALL = sorted({int(__import__("re").search(r"uid-(\d+)", os.path.basename(p)).group(1))
              for p in paths})
print("subjects:", ALL)

runs = []
for seed in range(a.seeds):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ALL)
    te_u, va_u = set(perm[:3].tolist()), set(perm[3:5].tolist())
    tr_p, va_p, te_p = subject_split(paths, te_u, va_u)

    tr = WindowSet(tr_p, a.win, a.hop)
    va = WindowSet(va_p, a.win, a.hop, stats=tr.stats)
    te = WindowSet(te_p, a.win, a.hop, stats=tr.stats)
    Xtr, ytr, mtr, ptr = tr.tensors()
    Xva, yva, mva, pva = va.tensors()
    Xte, yte, mte, pte = te.tensors()

    torch.manual_seed(seed)
    model = build(a.model, Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    cnt = np.bincount(ytr.numpy(), minlength=N_TASKS).astype(np.float32)
    w = torch.tensor(cnt.sum() / (N_TASKS * np.maximum(cnt, 1))).to(dev)
    ce = nn.CrossEntropyLoss(weight=w)
    pw = float((mtr == 0).sum()) / max(float((mtr == 1).sum()), 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw).to(dev))
    mse = nn.MSELoss()

    best, best_state, n = -1.0, None, len(Xtr)
    for ep in range(a.epochs):
        model.train()
        perm_i = torch.randperm(n)
        for i in range(0, n, a.bs):
            j = perm_i[i:i+a.bs]
            lt, lm, lp = model(Xtr[j].to(dev))
            loss = (ce(lt, ytr[j].to(dev)) + 0.3*bce(lm, mtr[j].float().to(dev))
                    + 0.3*mse(lp, ptr[j].to(dev)))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = evaluate(model, Xva, yva, mva, pva, dev)
        if v["macro_f1"] > best:
            best = v["macro_f1"]
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    r = evaluate(model, Xte, yte, mte, pte, dev)
    r["seed"] = seed; r["test_uids"] = sorted(te_u); r["val_uids"] = sorted(va_u)
    r["n_train_windows"] = len(Xtr); r["n_test_windows"] = len(Xte)
    runs.append(r)
    print("seed %d  test uids %-14s  macro-F1 %.3f  bal-acc %.3f  acc %.3f  progMAE %.1f"
          % (seed, str(sorted(te_u)), r["macro_f1"], r["balanced_acc"],
             r["accuracy"], r["progress_mae"]))

f1 = np.array([r["macro_f1"] for r in runs])
ba = np.array([r["balanced_acc"] for r in runs])
print("\n=== spread across %d random subject splits (%s) ===" % (a.seeds, a.model))
print("  macro-F1     mean %.3f  sd %.3f  min %.3f  max %.3f  range %.3f"
      % (f1.mean(), f1.std(), f1.min(), f1.max(), f1.max()-f1.min()))
print("  balanced-acc mean %.3f  sd %.3f  min %.3f  max %.3f  range %.3f"
      % (ba.mean(), ba.std(), ba.min(), ba.max(), ba.max()-ba.min()))
print("\n  per-class recall by seed:")
print("  %-6s %-16s " % ("seed", "test uids") + " ".join("%7s" % n[:7] for n in TASK_NAMES))
for r in runs:
    print("  %-6d %-16s " % (r["seed"], str(r["test_uids"]))
          + " ".join("%7.2f" % v for v in r["per_class_recall"].values()))
pc = {n: np.array([r["per_class_recall"][n] for r in runs]) for n in TASK_NAMES}
print("\n  %-10s %6s %6s %6s %6s" % ("class", "mean", "sd", "min", "max"))
for n, v in pc.items():
    print("  %-10s %6.2f %6.2f %6.2f %6.2f" % (n, v.mean(), v.std(), v.min(), v.max()))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(runs, open(a.out, "w"), indent=2)
print("\n[saved]", a.out)
