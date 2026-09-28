"""Export a deployable model from a finished LOSO run.

LOSO arms only ESTIMATE performance: each fold holds one subject out, so no
fold model has seen the whole corpus. The shipped model is retrained with the
run's exact settings in two stages:

  1. train on 13 subjects, select the epoch on 2 held-out subjects
  2. retrain on all 15 subjects for that many epochs, keep the last epoch

Everything the runtime needs to reproduce the model's input is written next to
the weights, so the model can never be fed columns it was not trained on:

  models/<name>/
    model_weights.pth           state_dict only
    model_bundle.pt             weights + standardisation + feature selection + config
    standardization.npz         per-column mean/std, and the column names they apply to
    standardization_by_panel.npz  the same stats keyed "<panel>_mean"/"<panel>_std"
                                  (the layout 1_recognition/norm_feat_rlt.py reads)
    feature_selection.json      panels, joints, exact column order, transforms,
                                window/stride
    config.json                 architecture, heads, classes, trigger, provenance
    README.md

Usage:
  python bench/deploy.py --from-run S2_10fps_8s --out-dir models/S2_10fps_8s
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import WindowSet, dedup_augmented, take_key
from models import build
from engine import train_one, evaluate


def _parse_cw(s):
    if not s:
        return None
    return {k.strip(): float(v) for k, v in (p.split("=") for p in s.split(","))}


def column_names(sample_path):
    """Exact column order of the model input, mirroring data._load_take."""
    md = torch.load(sample_path, map_location="cpu", weights_only=False)["metadata"]
    pc = md["panel_columns"]
    jk = [i for i, j in enumerate(D.FEATURE_JOINT_NAMES) if j not in D.DROP_JOINTS]
    ak = [i for i, j in enumerate(D.ANGLE_JOINTS) if j not in D.DROP_JOINTS]
    names, panels = [], []
    for k in D.PANEL_ORDER:
        cols = pc[k]
        if k == "polar_azimuth":
            sel = [cols[i] for i in jk]
            block = [c + "_sin" for c in sel] + [c + "_cos" for c in sel]
        elif k == "joint_angles":
            block = [cols[i] for i in ak]
        elif k == "ratios":
            block = list(cols)
        else:
            block = [cols[i] for i in jk]
        names += block
        panels += [(k, len(block))]
    return names, panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True, help="tag of a finished bench/loso.py run")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-subjects", default="",
                    help="2 subjects for epoch selection; default: fixed random pair")
    a = ap.parse_args()

    run_path = "bench/results/loso_%s.json" % a.from_run
    run = json.load(open(run_path))
    cfg = run["config"]
    D.apply_run_config(cfg)
    out = a.out_dir or os.path.join("models", a.from_run)
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    drop_lift = not cfg.get("keep_lift", False)
    stride = cfg.get("stride", 1)

    paths = sorted(glob.glob(os.path.join(cfg["data"], "*.pt")))
    aug = (dedup_augmented(sorted(glob.glob(os.path.join(cfg["aug_dir"], "*.pt"))))
           if cfg.get("aug") == "mirror" else [])
    uids = sorted({take_key(p)[0] for p in paths})
    va_u = ({int(x) for x in a.val_subjects.split(",")} if a.val_subjects
            else set(np.random.RandomState(0).permutation(uids)[:2].tolist()))
    kw = dict(pos_weight_cap=(cfg.get("pos_weight_cap") or None),
              focal_gamma=cfg.get("focal_gamma", 0.0),
              bg_weight=cfg.get("bg_weight", 0.2),
              class_weight=_parse_cw(cfg.get("class_weight", "")))

    def windows(files, stats=None):
        return WindowSet(files, cfg["win"], cfg["hop"], stats=stats,
                         drop_lift=drop_lift, stride=stride)

    # ---- stage 1: choose the epoch count on held-out subjects ----
    t0 = time.time()
    tr1 = windows([p for p in paths + aug if take_key(p)[0] not in va_u])
    va1 = windows([p for p in paths if take_key(p)[0] in va_u], tr1.stats)
    X1, y1 = tr1.tensors(); Xv, yv = va1.tensors()
    names = tr1.task_names
    torch.manual_seed(0)
    m1 = build(cfg["model"], X1.shape[-1], n_tasks=y1["plateau"].shape[1]).to(dev)
    m1, v_best, _ = train_one(m1, X1, y1, Xv, yv, dev, cfg["targets"], cfg["epochs"],
                              cfg["bs"], verbose=False, names=names, **kw)
    n_ep = int(getattr(m1, "best_epoch", cfg["epochs"]))
    val = evaluate(m1, Xv, yv, dev, cfg["targets"], names)
    print("[stage 1] val subjects %s: best epoch %d, macro-F1 %.3f  (%.0fs)"
          % (sorted(va_u), n_ep, val["macro_f1"], time.time() - t0), flush=True)
    del X1, y1, tr1

    # ---- stage 2: all subjects, that many epochs, keep the last ----
    tr = windows(paths + aug)
    X, y = tr.tensors()
    torch.manual_seed(0)
    model = build(cfg["model"], X.shape[-1], n_tasks=y["plateau"].shape[1]).to(dev)
    model, _, state = train_one(model, X, y, Xv, yv, dev, cfg["targets"], n_ep, cfg["bs"],
                                verbose=False, names=names, keep_last=True, **kw)
    print("[stage 2] all %d subjects, %d epochs, %d windows  (%.0fs)"
          % (len(uids), n_ep, len(X), time.time() - t0), flush=True)

    # ---- write artefacts ----
    mu, sd = tr.stats
    cols, panel_sizes = column_names(paths[0])
    assert len(cols) == X.shape[-1] == len(mu), (len(cols), X.shape[-1], len(mu))

    torch.save(state, os.path.join(out, "model_weights.pth"))
    np.savez(os.path.join(out, "standardization.npz"), mean=mu, std=sd,
             columns=np.array(cols))
    by_panel, c0 = {}, 0
    for k, n in panel_sizes:
        by_panel[k + "_mean"] = mu[c0:c0 + n]; by_panel[k + "_std"] = sd[c0:c0 + n]; c0 += n
    np.savez(os.path.join(out, "standardization_by_panel.npz"), **by_panel)

    feat = {
        "input_dim": len(cols),
        "panels": [k for k, _ in panel_sizes],
        "columns_per_panel": {k: n for k, n in panel_sizes},
        "joints_used": [j for j in D.FEATURE_JOINT_NAMES if j not in D.DROP_JOINTS],
        "joints_dropped": sorted(D.DROP_JOINTS),
        "column_order": cols,
        "transforms": {
            "polar_azimuth": "degrees -> radians -> [sin block, cos block]; "
                             "names end in _sin / _cos",
            "ratios": "clipped to [0, 4] before standardisation",
            "standardisation": "(x - mean) / std, per column, from standardization.npz",
        },
        "window": {"samples": cfg["win"], "stride_frames": stride,
                   "source_fps": 30, "model_rate_hz": 30 / stride,
                   "covers_seconds": round((cfg["win"] - 1) * stride / 30 + 1 / 30, 2),
                   "note": "the model counts samples: feed one feature vector per "
                           "1/model_rate_hz seconds"},
    }
    json.dump(feat, open(os.path.join(out, "feature_selection.json"), "w"), indent=2)

    idle_class = bool(cfg.get("idle_as_class"))
    loso = run["runs"]
    f1 = np.array([r["macro_f1"] for r in loso])
    conf = {
        "run": a.from_run,
        "architecture": cfg["model"], "hidden_dim": 128, "num_layers": 1,
        "input_dim": len(cols),
        "classes": names,
        "heads": {
            "task": "%d sigmoid logits, multi-label (NOT softmax); argmax for one label" % len(names),
            "prog": "%d lanes, progress 0-1 per class" % len(names),
            "mistake": "1 logit",
            "bg": ("unused (idle is a class)" if idle_class
                   else "1 logit: 'No Related Task' / idle; sigmoid, threshold 0.5 by default"),
        },
        "idle_handling": "7th class" if idle_class else "separate bg head",
        "training": {k: cfg.get(k) for k in ("targets", "aug", "epochs", "bs", "pos_weight_cap",
                                             "focal_gamma", "bg_weight", "class_weight")},
        "final_epochs": n_ep,
        "trigger": {
            "formula": "sigmoid(task)[Align] * sigmoid(task)[Place] * clip(max(prog), 0, 1), "
                       "smoothed over 3 s, fire on a sustained crossing, once per cycle",
            "align_lane": names.index("Align") if "Align" in names else None,
            "place_lane": names.index("Place") if "Place" in names else None,
            "threshold": None,
            "validated": False,
            "note": "The formula was validated on earlier models only. Re-run the "
                    "event-level trigger evaluation on this model before relying on it.",
        },
        "provenance": {
            "data": cfg["data"], "labels": "LSTM_HRC@f30a14d" if "f30a" in cfg["data"] else cfg["data"],
            "stage1_val_subjects": sorted(va_u), "stage1_val_macro_f1": round(val["macro_f1"], 4),
            "loso_macro_f1_mean": round(float(f1.mean()), 4),
            "loso_macro_f1_sd": round(float(f1.std()), 4),
            "exported": time.strftime("%Y-%m-%d %H:%M"),
        },
    }
    json.dump(conf, open(os.path.join(out, "config.json"), "w"), indent=2)
    torch.save({"state_dict": state, "stats": (mu, sd), "columns": cols,
                "feature_selection": feat, "config": conf},
               os.path.join(out, "model_bundle.pt"))

    readme = [
        "# %s" % a.from_run, "",
        "Deployable recognition model. Retrained on all %d subjects for %d epochs "
        "(epoch count chosen on held-out subjects %s)." % (len(uids), n_ep, sorted(va_u)), "",
        "Expected accuracy (15-fold leave-one-subject-out of the same settings): "
        "macro-F1 %.3f +/- %.3f." % (f1.mean(), f1.std()), "",
        "## Files", "",
        "| File | Contents |", "|---|---|",
        "| model_weights.pth | state_dict (`models.build('%s', %d, n_tasks=%d)`) |"
        % (cfg["model"], len(cols), len(names)),
        "| standardization.npz | `mean`, `std`, `columns` -- one entry per input column |",
        "| standardization_by_panel.npz | same stats keyed `<panel>_mean` / `<panel>_std` |",
        "| feature_selection.json | panels, joints, exact column order, transforms, window |",
        "| config.json | classes, heads, training settings, trigger, provenance |",
        "| model_bundle.pt | all of the above in one file |", "",
        "## Input", "",
        "- %d columns in the order of `feature_selection.json` -> `column_order`." % len(cols),
        "- polar_azimuth is converted to sin/cos BEFORE standardisation; ratios are clipped to [0, 4].",
        "- Window: %d samples, one every %d source frames = %.0f Hz. "
        "Feed the model at that rate." % (cfg["win"], stride, 30 / stride),
        "",
        "## Output", "",
        "- task: %d independent sigmoid logits, classes %s." % (len(names), names),
        "- bg: idle probability (sigmoid).",
        "",
        "## Not yet done", "",
        "- The trigger formula in config.json has not been re-validated on this model.",
        "- 1_recognition/recognition_manager.py still loads the older 2-head AssistLSTM "
        "and needs updating for this 4-head model.",
    ]
    open(os.path.join(out, "README.md"), "w").write("\n".join(readme) + "\n")

    print("[deploy] wrote %s/" % out)
    for f in sorted(os.listdir(out)):
        print("    %-32s %8.1f KB" % (f, os.path.getsize(os.path.join(out, f)) / 1024))


if __name__ == "__main__":
    main()
