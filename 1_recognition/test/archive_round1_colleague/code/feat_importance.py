"""Permutation importance per feature panel, per class, on trained LOSO models.

For each held-out subject's model, shuffle one panel's columns across windows
(keeping each window's internal time structure, but breaking its link to the
label) and measure how much each class's F1 drops. A panel whose shuffling
costs nothing is not being used. Averaged over all 15 folds.

Caveat: correlated panels share credit. If position and polar angles carry the
same information, shuffling either alone costs little because the other still
supplies it -- so low importance + high redundancy means "replaceable", not
"useless". Read this together with feat_redundancy.py.
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import WindowSet, take_key, PANEL_ORDER
from models import build
from sklearn.metrics import f1_score

TAG = sys.argv[1] if len(sys.argv) > 1 else "N1_full_bg"
dev = "cuda" if torch.cuda.is_available() else "cpu"
paths = sorted(glob.glob("hrc_communication/other/original/*.pt"))

# column ranges of each panel in the 251-dim input
sizes = {"polar_azimuth": 32, "joint_angles": 9, "ratios": 2}
cols, c0 = {}, 0
for k in PANEL_ORDER:
    n = sizes.get(k, 16); cols[k] = list(range(c0, c0 + n)); c0 += n
assert c0 == 251

@torch.no_grad()
def run(m, X, perm=None, pcols=None, bs=512):
    out_t, out_b = [], []
    for i in range(0, len(X), bs):
        idx = torch.arange(i, min(i + bs, len(X)))
        xb = X[idx].clone()
        if perm is not None:
            xb[:, :, pcols] = X[perm[idx]][:, :, pcols]
        o = m(xb.to(dev))
        out_t.append(o["task"].argmax(1).cpu()); out_b.append((o["bg"] > 0).cpu())
    return torch.cat(out_t).numpy(), torch.cat(out_b).numpy()

def scores(pred, pbg, true, keep, tbg, C):
    f = f1_score(true[keep], pred[keep], labels=range(C), average=None, zero_division=0)
    return np.concatenate([f, [f.mean(), f1_score(tbg, pbg, zero_division=0)]])

acc = {k: [] for k in PANEL_ORDER}; base_all = []; names = None
for fp in sorted(glob.glob("bench/results/folds/%s_uid*.pth" % TAG)):
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    uid = ck["test_uids"][0]
    te = WindowSet([p for p in paths if take_key(p)[0] == uid], ck["win"], ck["hop"], stats=ck["stats"], stride=ck.get("stride", 1))
    X, y = te.tensors(); names = names or te.task_names; C = len(names)
    m = build(ck["model"], ck["dim"], n_tasks=C).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
    pl, pk = y["plateau"].numpy(), y["peak"].numpy()
    tbg = y["bg"].numpy().astype(bool); keep = ~tbg
    true = (pl + 1e-3 * pk).argmax(1)
    p0, b0 = run(m, X)
    base = scores(p0, b0, true, keep, tbg, C); base_all.append(base)
    g = torch.Generator().manual_seed(uid)
    perm = torch.randperm(len(X), generator=g)
    for k in PANEL_ORDER:
        p1, b1 = run(m, X, perm, cols[k])
        acc[k].append(base - scores(p1, b1, true, keep, tbg, C))
    print("  fold uid %2d done  (base macro-F1 %.3f)" % (uid, base[C]), flush=True)

labels = names + ["MACRO", "idle(bg)"]
B = np.mean(base_all, 0)
D = {k: np.mean(v, 0) for k, v in acc.items()}
order = sorted(PANEL_ORDER, key=lambda k: -D[k][len(names)])
print("\nF1 DROP when each panel is shuffled (higher = more important), mean over %d folds" % len(base_all))
print("%-30s" % "panel" + "".join("%9s" % l[:8] for l in labels))
print("%-30s" % "(baseline F1)" + "".join("%9.3f" % v for v in B))
for k in order:
    print("%-30s" % k + "".join("%+9.3f" % v for v in D[k]))
json.dump({"tag": TAG, "labels": labels, "baseline": B.tolist(),
           "drop": {k: v.tolist() for k, v in D.items()},
           "per_fold": {k: np.array(v).tolist() for k, v in acc.items()}},
          open("bench/results/feat_importance_%s.json" % TAG, "w"), indent=1)
