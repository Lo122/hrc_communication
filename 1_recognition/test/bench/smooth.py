"""Post-hoc temporal smoothing, applied to LANE PROBABILITIES rather than to the
argmax output -- smoothing a discrete label sequence throws away the model's
confidence, which is exactly what the transition prior needs.

Two filters, neither requiring retraining:

  * median filter -- kills isolated single-window flips
  * Viterbi       -- decodes the most likely label SEQUENCE using the task
                     transition matrix estimated from the TRAINING subjects

The assembly process is a strong loop (Connect->Clamp 248x, Lift->Place 263x
corpus-wide), so the sequence prior carries real information the per-window
classifier discards. `hrc_communication/1_recognition/step_stablizier.py` already
exists to do this job in the runtime -- it just isn't fed transition probabilities.
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from data import WindowSet, subject_split, load_take, TASK_NAMES, N_TASKS
from models import build
from engine import predict
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.ndimage import median_filter

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="bench/results/gru_vector.pth")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg, dev = ck["config"], "cuda" if torch.cuda.is_available() else "cpu"
te_u = {int(x) for x in str(cfg["test_uids"]).split(",")}
va_u = {int(x) for x in str(cfg["val_uids"]).split(",")}
paths = sorted(glob.glob(os.path.join(cfg["data"], "*.pt")))
tr_p, _, te_p = subject_split(paths, te_u, va_u)

# transition matrix from TRAIN subjects only
A = np.ones((N_TASKS, N_TASKS)) * 1e-3
for p in tr_p:
    d = load_take(p)
    keep = ~d["bg"] & d["valid"]
    y = (d["plateau"] + 1e-3 * d["peak"]).argmax(1)[keep]
    for x, z in zip(y[:-1], y[1:]):
        A[x, z] += 1
A /= A.sum(1, keepdims=True)
prior = np.log(A)

te = WindowSet(te_p, cfg["win"], cfg["hop"], stats=ck["stats"])
X, y = te.tensors()
model = build(cfg["model"], ck["dim"]).to(dev)
model.load_state_dict(ck["state_dict"])
o = predict(model, X, dev)

logit = o["task"].numpy()
LP = logit - np.log(np.exp(logit).sum(1, keepdims=True))   # log-softmax over lanes
keep = ~y["bg"].numpy().astype(bool)
true = (y["plateau"].numpy() + 1e-3 * y["peak"].numpy()).argmax(1)
ti = te.take_index()


def rep(name, pred):
    f = f1_score(true[keep], pred[keep], average="macro", zero_division=0)
    print("  %-24s macro-F1 %.3f   bal-acc %.3f"
          % (name, f, balanced_accuracy_score(true[keep], pred[keep])))
    return f


print("\n=== post-hoc temporal smoothing (%s, held-out subjects) ===" % cfg["model"])
raw = LP.argmax(1)
rep("raw argmax", raw)

for k in (5, 9, 15, 25):
    out = raw.copy()
    for t in np.unique(ti):                       # never smooth across takes
        m = ti == t
        out[m] = median_filter(raw[m], size=k, mode="nearest")
    rep("median filter k=%d" % k, out)

for scale in (0.3, 0.6, 1.0, 2.0):
    out = np.empty_like(raw)
    for t in np.unique(ti):
        m = np.flatnonzero(ti == t)
        lp = LP[m]; T = len(lp)
        dp = np.full((T, N_TASKS), -1e18); bp = np.zeros((T, N_TASKS), int)
        dp[0] = lp[0]
        for i in range(1, T):
            sc = dp[i - 1][:, None] + prior * scale
            bp[i] = sc.argmax(0); dp[i] = sc.max(0) + lp[i]
        path = np.zeros(T, int); path[-1] = dp[-1].argmax()
        for i in range(T - 1, 0, -1):
            path[i - 1] = bp[i, path[i]]
        out[m] = path
    rep("viterbi (prior x%.1f)" % scale, out)
