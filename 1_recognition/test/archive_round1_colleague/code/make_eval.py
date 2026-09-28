"""Per-arm evaluation artefacts in the format model_lstm/runs_3d/exp_* uses:
metrics.csv, confusion matrices, per-class breakdown.

Their pipeline reports a single train/val/test split. Ours is 15-fold LOSO, so
every confusion matrix here is pooled across all 15 held-out subjects, and
metrics.csv carries one row per fold plus mean/sd. The per-fold spread is given
explicitly because a single split cannot show it -- measured at +/-0.08 macro-F1,
which is larger than most of the differences being compared.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import WindowSet, take_key
from models import build
from engine import predict
from sklearn.metrics import confusion_matrix, f1_score

NL = "\n"


def pooled(tag, data_dir, dev):
    # Re-apply the arm's own feature/label settings -- N3-N6 were trained on
    # reduced panels and N4/N6 with idle as a class, so the default 251-dim
    # input would not match their weights.
    cfg_path = "bench/results/loso_%s.json" % tag
    if os.path.exists(cfg_path):
        D.apply_run_config(json.load(open(cfg_path)).get("config", {}))
    paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    rows, cm, names = [], None, None
    for fp in sorted(glob.glob("bench/results/folds/%s_uid*.pth" % tag)):
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        te_p = [p for p in paths if take_key(p)[0] in set(ck["test_uids"])]
        te = WindowSet(te_p, ck["win"], ck["hop"], stats=ck["stats"], stride=ck.get("stride", 1))
        X, y = te.tensors()
        if X.shape[-1] != ck["dim"]:
            raise SystemExit("%s: model expects %d features, data gives %d -- "
                             "config not applied" % (tag, ck["dim"], X.shape[-1]))
        names = names or te.task_names
        C = y["plateau"].shape[1]
        m = build(ck["model"], ck["dim"], n_tasks=C).to(dev)
        m.load_state_dict(ck["state_dict"])
        o = predict(m, X, dev)
        pl, pk = y["plateau"].numpy(), y["peak"].numpy()
        keep = ~y["bg"].numpy().astype(bool)
        # Ground truth is the strongest plateau lane with peak used to break
        # ties: the plateau saturates at exactly 1.0, so without this a plain
        # argmax resolves every tie toward the lowest class index.
        true = (pl + 1e-3 * pk).argmax(1)[keep]
        pred = o["task"].numpy().argmax(1)[keep]
        act = (pl >= 0.5) & keep[:, None]
        prog_t, prog_p = y["prog_vec"].numpy(), o["prog"].numpy()
        cmf = confusion_matrix(true, pred, labels=range(C))
        cm = cmf if cm is None else cm + cmf
        # Comparable score across arms. When idle is a trained class, plain
        # macro-F1 averages 7 classes; when it is routed to the bg head it
        # averages 6. Averaging over a different class set moves the number
        # without the model changing, so every arm is also scored on the same
        # six task classes over frames whose true label is a task. Predicting
        # "idle" on such a frame counts as a miss.
        idle = names.index("No Related Task") if "No Related Task" in names else None
        task_ids = [i for i in range(C) if i != idle]
        tmask = np.ones(len(true), bool) if idle is None else (true != idle)
        f1_task6 = float(f1_score(true[tmask], pred[tmask], labels=task_ids,
                                  average="macro", zero_division=0))
        rows.append({
            "test_uid": ck["test_uids"][0],
            "n_windows": int(keep.sum()),
            "macro_f1_task6": f1_task6,
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "accuracy": float((pred == true).mean()),
            "progress_mae": (float(np.abs(prog_p[act] - prog_t[act]).mean() * 100)
                             if act.any() else float("nan")),
            "mistake_f1": float(f1_score(y["mistake"].numpy(),
                                         (o["mistake"].numpy() > 0).astype(int),
                                         zero_division=0)),
            "bg_f1": float(f1_score(y["bg"].numpy().astype(bool),
                                    o["bg"].numpy() > 0, zero_division=0)),
        })
    return names, cm, rows


def write_csv(path, header, lines):
    with open(path, "w") as f:
        f.write(header + NL)
        for ln in lines:
            f.write(ln + NL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--data", default="hrc_communication/other/original")
    ap.add_argument("--out", default="bench/eval")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    for tag in a.tags:
        names, cm, rows = pooled(tag, a.data, dev)
        if cm is None:
            print("  [%s] no fold weights found" % tag)
            continue
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)

        keys = list(rows[0])
        num = [k for k in keys if k != "test_uid"]
        arr = {k: np.array([r[k] for r in rows], dtype=float) for k in num}
        write_csv(os.path.join(d, "metrics.csv"), ",".join(keys),
                  [",".join(str(r[k]) for k in keys) for r in rows]
                  + ["mean," + ",".join("%.6f" % arr[k].mean() for k in num),
                     "sd," + ",".join("%.6f" % arr[k].std() for k in num)])

        row, col = cm.sum(1, keepdims=True), cm.sum(0)
        write_csv(os.path.join(d, "confusion_matrix.csv"),
                  "true_vs_pred," + ",".join(names) + ",support",
                  [names[i] + "," + ",".join(str(v) for v in cm[i])
                   + ",%d" % row[i, 0] for i in range(len(names))])

        per = []
        for i, n in enumerate(names):
            rec = cm[i, i] / max(row[i, 0], 1)
            pre = cm[i, i] / max(col[i], 1)
            per.append("%s,%.4f,%.4f,%.4f,%d"
                       % (n, rec, pre, 2 * rec * pre / max(rec + pre, 1e-9), row[i, 0]))
        write_csv(os.path.join(d, "per_class.csv"),
                  "class,recall,precision,f1,support", per)

        json.dump({"names": names, "matrix": cm.tolist(),
                   "row_normalised": np.round(cm / np.maximum(row, 1), 4).tolist(),
                   "per_fold": rows},
                  open(os.path.join(d, "evaluation.json"), "w"), indent=2)

        f6, f1 = arr["macro_f1_task6"], arr["macro_f1"]
        print("  %-18s macro-F1(6 tasks) %.3f +/- %.3f   macro-F1(all) %.3f   "
              "bg-F1 %.3f   %d folds"
              % (tag, f6.mean(), f6.std(), f1.mean(), arr["bg_f1"].mean(), len(rows)))


if __name__ == "__main__":
    main()
