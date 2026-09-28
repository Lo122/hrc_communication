"""Permutation importance per JOINT (and joint group), per class.

For a joint, every column that belongs to it is shuffled together across
windows: its speed, acceleration, velocity/acceleration x/y/z, position x/y/z,
azimuth (sin and cos), elevation, distance-from-center, and any joint angle
measured AT it (e.g. left_knee_angle_deg -> l_knee). The 2 ratio columns are
never shuffled here (they mix elbows, wrists and shoulders).

Layout of the 251-dim input follows data.PANEL_ORDER: each per-joint panel lists
the 16 non-pelvis H36M joints in h36m_features.FEATURE_JOINTS order; the azimuth
panel is 16 sin columns then 16 cos columns.
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

JOINTS = ["r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine", "thorax",
          "neck", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist"]
ANGLE_AT = ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder", "l_hip", "r_hip",
            "l_knee", "r_knee", "neck"]           # compute_joint_angles order

jcols = {j: [] for j in JOINTS}
c0 = 0
for k in PANEL_ORDER:
    if k == "polar_azimuth":
        for part in range(2):                      # sin block, then cos block
            for i, j in enumerate(JOINTS): jcols[j].append(c0 + part * 16 + i)
        c0 += 32
    elif k == "joint_angles":
        for i, j in enumerate(ANGLE_AT): jcols[j].append(c0 + i)
        c0 += 9
    elif k == "ratios":
        c0 += 2
    else:
        for i, j in enumerate(JOINTS): jcols[j].append(c0 + i)
        c0 += 16
assert c0 == 251

GROUPS = {
    "LOWER BODY (hips,knees,ankles)": ["r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"],
    "LEGS (knees,ankles)":            ["r_knee", "r_ankle", "l_knee", "l_ankle"],
    "TORSO+HEAD (spine..head)":       ["spine", "thorax", "neck", "head"],
    "ARMS (shoulder,elbow,wrist)":    ["l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist"],
    "WRISTS only":                    ["l_wrist", "r_wrist"],
}
tests = {j: jcols[j] for j in JOINTS}
for g, js in GROUPS.items():
    tests[g] = sorted(c for j in js for c in jcols[j])

@torch.no_grad()
def run(m, X, perm=None, pc=None, bs=512):
    pt, pb = [], []
    for i in range(0, len(X), bs):
        idx = torch.arange(i, min(i + bs, len(X)))
        xb = X[idx].clone()
        if perm is not None:
            xb[:, :, pc] = X[perm[idx]][:, :, pc]
        o = m(xb.to(dev))
        pt.append(o["task"].argmax(1).cpu()); pb.append((o["bg"] > 0).cpu())
    return torch.cat(pt).numpy(), torch.cat(pb).numpy()

def score(p, b, t, keep, tbg, C):
    f = f1_score(t[keep], p[keep], labels=range(C), average=None, zero_division=0)
    return np.concatenate([f, [f.mean(), f1_score(tbg, b, zero_division=0)]])

acc = {k: [] for k in tests}; names = None; base_all = []
for fp in sorted(glob.glob("bench/results/folds/%s_uid*.pth" % TAG)):
    ck = torch.load(fp, map_location="cpu", weights_only=False); uid = ck["test_uids"][0]
    te = WindowSet([p for p in paths if take_key(p)[0] == uid], ck["win"], ck["hop"], stats=ck["stats"], stride=ck.get("stride", 1))
    X, y = te.tensors(); names = names or te.task_names; C = len(names)
    m = build(ck["model"], ck["dim"], n_tasks=C).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
    pl, pk = y["plateau"].numpy(), y["peak"].numpy()
    tbg = y["bg"].numpy().astype(bool); keep = ~tbg; t = (pl + 1e-3 * pk).argmax(1)
    base = score(*run(m, X), t, keep, tbg, C); base_all.append(base)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(uid))
    for k, pc in tests.items():
        acc[k].append(base - score(*run(m, X, perm, pc), t, keep, tbg, C))
    print("  uid %2d done" % uid, flush=True)

labels = names + ["MACRO", "idle"]
D = {k: np.mean(v, 0) for k, v in acc.items()}
SD = {k: np.std([x[len(names)] for x in v]) for k, v in acc.items()}
def show(keys, title):
    print("\n" + title)
    print("%-32s" % "" + "".join("%9s" % l[:8] for l in labels) + "   sd(MACRO)")
    for k in keys:
        print("%-32s" % k + "".join("%+9.3f" % v for v in D[k]) + "   %.3f" % SD[k])
show(list(GROUPS), "GROUPS -- F1 drop when the whole group is shuffled")
show(sorted(JOINTS, key=lambda j: -D[j][len(names)]), "SINGLE JOINTS -- sorted by macro-F1 drop")
json.dump({"tag": TAG, "labels": labels, "cols": jcols,
           "drop": {k: v.tolist() for k, v in D.items()},
           "per_fold": {k: np.array(v).tolist() for k, v in acc.items()}},
          open("bench/results/feat_joint_importance_%s.json" % TAG, "w"), indent=1)
