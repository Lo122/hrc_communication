"""How much of each feature panel is already contained in the others?

Many panels are derived from the same 3D joint positions: speed is the norm of
velocity x/y/z, polar angles and distance_from_center are functions of position,
acceleration is the derivative of velocity. For each panel, fit a ridge
regression predicting it from all OTHER panels and report R^2. A panel with
R^2 near 1 adds almost nothing new; near 0 means it is unique information.
Computed per frame on a subject-disjoint sample, training panels only.
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import PANEL_ORDER

paths = sorted(glob.glob("hrc_communication/other/original/*.pt"))
rng = np.random.RandomState(0)
blocks = {k: [] for k in PANEL_ORDER}
for p in paths[::3]:                                   # one camera view per take
    f = torch.load(p, map_location="cpu", weights_only=False)["features"]
    T = f[PANEL_ORDER[0]].shape[0]
    idx = rng.choice(T, min(T, 1500), replace=False)
    for k in PANEL_ORDER:
        a = f[k].numpy()[idx].astype(np.float64)
        if k == "polar_azimuth":
            r = np.deg2rad(a); a = np.concatenate([np.sin(r), np.cos(r)], 1)
        if k == "ratios":
            a = np.clip(a, 0, 4)
        blocks[k].append(a)
M = {k: np.concatenate(v) for k, v in blocks.items()}
ok = np.all([np.isfinite(v).all(1) for v in M.values()], axis=0)
M = {k: (v[ok] - v[ok].mean(0)) / (v[ok].std(0) + 1e-9) for k, v in M.items()}
n = len(next(iter(M.values())))
tr = np.arange(n) < int(0.7 * n)

print("frames used: %d\n" % n)
print("%-32s %5s %9s   %s" % ("panel", "cols", "R2 from", "most similar other panel"))
print("%-32s %5s %9s" % ("", "", "others"))
rows = []
for k in PANEL_ORDER:
    Y = M[k]
    X = np.concatenate([M[o] for o in PANEL_ORDER if o != k], 1)
    lam = 1.0
    W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ Y[tr])
    pred = X[~tr] @ W
    r2 = 1 - ((Y[~tr] - pred) ** 2).sum() / ((Y[~tr] - Y[~tr].mean(0)) ** 2).sum()
    # single most similar panel
    best, bo = -1, None
    for o in PANEL_ORDER:
        if o == k: continue
        Xo = M[o]
        Wo = np.linalg.solve(Xo[tr].T @ Xo[tr] + lam * np.eye(Xo.shape[1]), Xo[tr].T @ Y[tr])
        r2o = 1 - ((Y[~tr] - Xo[~tr] @ Wo) ** 2).sum() / ((Y[~tr] - Y[~tr].mean(0)) ** 2).sum()
        if r2o > best: best, bo = r2o, o
    rows.append((k, Y.shape[1], r2, bo, best))
    print("%-32s %5d %9.2f   %s (%.2f)" % (k, Y.shape[1], r2, bo, best))
np.save("bench/results/feat_redundancy.npy", np.array(rows, dtype=object), allow_pickle=True)
