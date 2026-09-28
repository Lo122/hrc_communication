"""Fair head-to-head: idle as a bg head vs idle as a 7th class.

Both arms are turned into the same 7-way decision per window:
  bg-head arm : 'idle' if the bg logit > 0, else argmax over the 6 task lanes
  class arm   : argmax over 7 lanes (the 7th is 'No Related Task')
Ground truth is the same 7-way label for both (idle if the frame is annotated
'No Related Task', else the strongest task lane). Scored on every window,
nothing masked, so a task frame wrongly called idle costs both arms equally.
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import WindowSet, take_key
from models import build
from sklearn.metrics import f1_score
dev = "cuda" if torch.cuda.is_available() else "cpu"
paths = sorted(glob.glob("hrc_communication/other/original/*.pt"))

def seven_way(tag):
    cfg = json.load(open("bench/results/loso_%s.json" % tag))["config"]
    idle_cls = bool(cfg.get("idle_as_class"))
    per_fold = []
    for fp in sorted(glob.glob("bench/results/folds/%s_uid*.pth" % tag)):
        ck = torch.load(fp, map_location="cpu", weights_only=False); uid = ck["test_uids"][0]
        # labels come from the bg-head layout (6 task lanes + bg flag) for BOTH arms
        D.apply_run_config(dict(cfg, idle_as_class=False))
        te_lab = WindowSet([p for p in paths if take_key(p)[0] == uid], ck["win"], ck["hop"],
                           stats=ck["stats"], stride=ck.get("stride", 1))
        _, yl = te_lab.tensors()
        truth = np.where(yl["bg"].numpy() > 0.5, 6,
                         (yl["plateau"].numpy() + 1e-3 * yl["peak"].numpy()).argmax(1))
        D.apply_run_config(cfg)
        te = WindowSet([p for p in paths if take_key(p)[0] == uid], ck["win"], ck["hop"],
                       stats=ck["stats"], stride=ck.get("stride", 1))
        X, y = te.tensors(); C = y["plateau"].shape[1]
        m = build(ck["model"], ck["dim"], n_tasks=C).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
        T, B = [], []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                o = m(X[i:i + 512].to(dev)); T.append(o["task"].cpu()); B.append(o["bg"].cpu())
        T, B = torch.cat(T).numpy(), torch.cat(B).numpy()
        # class arm: lane 6 of the 7 is already 'No Related Task' (new-corpus order)
        pred = T.argmax(1) if idle_cls else np.where(B > 0, 6, T.argmax(1))
        f = f1_score(truth, pred, labels=range(7), average=None, zero_division=0)
        per_fold.append(f)
    return np.array(per_fold)

names = ["Pull", "Place", "Align", "Screw", "Connect", "Clamp", "IDLE"]
res = {t: seven_way(t) for t in ["N3_red_bg", "N4_red_idle", "N5_redvel_bg", "N6_redvel_idle", "N1_full_bg"]}
print("%-16s" % "7-way F1" + "".join("%8s" % n for n in names) + "   task6   all7")
for t, f in res.items():
    m = f.mean(0)
    print("%-16s" % t + "".join("%8.2f" % v for v in m) + "   %.3f  %.3f" % (m[:6].mean(), m.mean()))
from scipy.stats import wilcoxon
for a, b in [("N3_red_bg", "N4_red_idle"), ("N5_redvel_bg", "N6_redvel_idle")]:
    d = res[b][:, :6].mean(1) - res[a][:, :6].mean(1)
    print("  %s -> %s   task6 %+0.3f  class-arm wins %d/15  p=%.4f"
          % (a, b, d.mean(), (d > 0).sum(), wilcoxon(d).pvalue))

json.dump({"names": names, "per_fold_f1": {t: f.tolist() for t, f in res.items()}},
          open("bench/results/fair_idle_round1.json", "w"), indent=1)
print("[saved] bench/results/fair_idle_round1.json")
