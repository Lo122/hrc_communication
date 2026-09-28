"""Shared train/eval engine, so train.py (single split) and loso.py (15 folds)
cannot drift apart.

Two target modes:

  hard   -- the original configuration: softmax cross-entropy on `task_id`,
            scalar progress. Reproduces the first benchmark. With bg_mask=True
            it additionally excludes background frames, isolating that bug.
  vector -- per-lane BCE on `task_id_plateau_vector`, per-lane progress masked to
            active lanes, plus an explicit background head.

Metrics are reported two ways. The multi-label view (per-lane AP / F1) matches
what the vector model actually predicts. The background-masked argmax macro-F1 is
the number comparable across ALL arms, including the original 0.416.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import (f1_score, balanced_accuracy_score, confusion_matrix,
                             average_precision_score, precision_score)
from data import TASK_NAMES, N_TASKS, ALL_TASK_NAMES, task_names


def focal_bce(logits, target, pos_weight, gamma, lane_weight=None):
    """BCE with the focal modulation (1-p_t)^gamma.

    Plain BCE keeps paying attention to examples it already gets right. The
    measured failure here is the opposite of hard-example mining: the model is
    CONFIDENTLY WRONG about Lift on 73% of the frames where Lift is absent, and
    those confident negatives are exactly what focal down-weights once they are
    learned, letting the remaining gradient go to the genuinely ambiguous frames.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")
    if gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        bce = (1 - p_t).clamp(min=1e-6) ** gamma * bce
    if lane_weight is not None:
        # Per-class emphasis. Unlike pos_weight, this scales the WHOLE lane --
        # both missing the class and falsely asserting it cost more -- so it
        # sharpens that class's decision boundary rather than just biasing it
        # toward saying yes.
        bce = bce * lane_weight
    return bce.mean()


def compute_losses(out, yb, mode, w, pos_w, bce_m, dev, focal_gamma=0.0,
                   bg_weight=0.2, lane_weight=None):
    """-> scalar loss. `w` is class weights (hard) or per-lane pos_weight (vector)."""
    mse = nn.functional.mse_loss
    if mode == "hard":
        loss = nn.functional.cross_entropy(out["task"], yb["task"], weight=w)
        loss = loss + 0.3 * mse(out["prog"].gather(
            1, yb["task"].unsqueeze(1)).squeeze(1), yb["prog"])
    else:
        # "vector" = plateau (flat 1.0 across the span: presence)
        # "peak"   = asymmetric_peak (1.0 only at the span midpoint: centrality)
        # Both are soft targets in [0,1]; BCE accepts them directly. They are NOT
        # distributions -- lanes accumulate independently and rows sum 0.00-3.30 --
        # so cross-entropy/KL would be wrong here.
        tgt = yb["plateau"] if mode == "vector" else yb["peak"]
        loss = focal_bce(out["task"], tgt, pos_w, focal_gamma, lane_weight)
        # Progress only where a lane is actually active: an inactive lane's
        # progress is 0 by construction and would otherwise swamp the loss.
        active = (yb["plateau"] >= 0.5).float()
        if active.sum() > 0:
            loss = loss + 0.3 * (((out["prog"] - yb["prog_vec"]) ** 2 * active).sum()
                                 / active.sum())
        # Idle frames are 18.2% of the new corpus, so this head is not a minor
        # auxiliary task. Measured: raising the weight 0.2 -> 0.5 gave +0.012
        # macro-F1 and +0.05 idle F1 on a single split.
        loss = loss + bg_weight * nn.functional.binary_cross_entropy_with_logits(
            out["bg"], yb["bg"])
    return loss + 0.3 * bce_m(out["mistake"], yb["mistake"].float())


@torch.no_grad()
def predict(model, X, dev, bs=512):
    model.eval()
    acc = {}
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs].to(dev))
        for k, v in o.items():
            acc.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def evaluate(model, X, y, dev, mode, names=None):
    """Both metric views. `argmax_*` keys are comparable across all arms."""
    o = predict(model, X, dev)
    task_logit = o["task"].numpy()
    plateau = y["plateau"].numpy()
    bg_true = y["bg"].numpy().astype(bool)
    C = plateau.shape[1]                  # 6 with Lift dropped, 7 without
    if names is None:
        names = (TASK_NAMES if C == len(TASK_NAMES)
                 else task_names(C < len(ALL_TASK_NAMES)))
    res = {}

    # --- multi-label view (what the vector model predicts) ---
    pos = (plateau >= 0.5)
    if mode in ("vector", "peak"):
        prob = 1.0 / (1.0 + np.exp(-task_logit))
        aps, f1s, precs = [], [], []
        per_prec = {}
        for k in range(C):
            if pos[:, k].any() and not pos[:, k].all():
                aps.append(average_precision_score(pos[:, k], prob[:, k]))
                f1s.append(f1_score(pos[:, k], prob[:, k] >= 0.5, zero_division=0))
                # PRECISION is the metric that was missing. Every other number
                # here -- macro-F1, recall, AP -- is recall-weighted and does not
                # penalise a false positive, which is why the model's habit of
                # asserting absent classes went unnoticed for five LOSO runs.
                pr = precision_score(pos[:, k], prob[:, k] >= 0.5, zero_division=0)
                precs.append(pr)
                per_prec[names[k]] = float(pr)
        res["lane_ap"] = float(np.mean(aps)) if aps else float("nan")
        res["lane_f1"] = float(np.mean(f1s)) if f1s else float("nan")
        res["lane_precision"] = float(np.mean(precs)) if precs else float("nan")
        res["per_class_precision"] = per_prec
        res["bg_f1"] = float(f1_score(bg_true, o["bg"].numpy() > 0, zero_division=0))

    # --- single-label view, background-masked: comparable everywhere ---
    keep = ~bg_true
    pred = task_logit.argmax(1)[keep]
    # Ground truth = strongest plateau lane, ties broken by the peak curve.
    # The plateau saturates at exactly 1.0, so 18% of windows tie at the max; a
    # plain argmax resolves all of them toward the lowest index and wipes out
    # Place (9.6% of windows -> 0.5%). Adding a small multiple of the peak score
    # -- which has a unique max by construction -- orders tied lanes by which
    # span's midpoint is nearest, without disturbing untied rows.
    true = (plateau + 1e-3 * y["peak"].numpy()).argmax(1)[keep]
    res["macro_f1"] = float(f1_score(true, pred, average="macro", zero_division=0))
    res["balanced_acc"] = float(balanced_accuracy_score(true, pred))
    res["accuracy"] = float((pred == true).mean())
    cm = confusion_matrix(true, pred, labels=range(C), normalize="true")
    res["per_class_recall"] = {n: float(r) for n, r in zip(names, cm.diagonal())}

    res["mistake_f1"] = float(f1_score(y["mistake"].numpy(),
                                       (o["mistake"].numpy() > 0).astype(int),
                                       zero_division=0))
    # progress MAE on active lanes only, in 0-100 units
    act = pos & keep[:, None]
    res["progress_mae"] = (float(np.abs(o["prog"].numpy()[act]
                                        - y["prog_vec"].numpy()[act]).mean() * 100)
                           if act.any() else float("nan"))
    return res


def train_one(model, Xtr, ytr, Xva, yva, dev, mode, epochs=12, bs=256, lr=1e-3,
              verbose=True, pos_weight_cap=None, focal_gamma=0.0, names=None,
              bg_weight=0.2, class_weight=None, keep_last=False):
    """Train, keeping the best epoch by val macro-F1. -> (model, best_val_f1)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    C = ytr["plateau"].shape[1]
    cnt = np.bincount(ytr["task"].numpy(), minlength=C).astype(np.float32)[:C]
    w = torch.tensor(cnt.sum() / (C * np.maximum(cnt, 1))).to(dev)
    # pos_weight balances each lane against its own rarity. Derive it from the
    # SAME curve being trained on: the peak curve is soft almost everywhere
    # (only 1% of its entries reach 1.0 vs plateau's 12%), so reusing plateau's
    # counts would systematically under-weight the positives a peak run sees.
    src = ytr["plateau"] if mode != "peak" else ytr["peak"]
    p = (src.numpy() >= 0.5)
    raw_pw = (len(p) - p.sum(0)) / np.maximum(p.sum(0), 1)
    # Cap it. pos_weight > 1 tells BCE to prefer false positives over false
    # negatives; uncapped these reach 20.05 (Pull), 7.36 (Align), 3.97 (Lift),
    # so wholesale over-prediction is the loss doing exactly what it was told.
    # That was right while the objective was per-class recall and wrong now that
    # the objective is a trigger a human has to answer.
    if pos_weight_cap is not None:
        raw_pw = np.minimum(raw_pw, pos_weight_cap)
    pos_w = torch.tensor(raw_pw.astype(np.float32)).to(dev)
    mr = ytr["mistake"].numpy()
    bce_m = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        float((mr == 0).sum()) / max(float((mr == 1).sum()), 1.0)).to(dev))

    lane_w = None
    if class_weight:
        C = ytr["plateau"].shape[1]
        lw = np.ones(C, dtype=np.float32)
        for name, v in class_weight.items():
            if names is None or name not in names:
                raise ValueError("class_weight: unknown class %r (have %s)" % (name, names))
            lw[names.index(name)] = v
        lane_w = torch.tensor(lw).to(dev)
        if verbose:
            print("  [loss] lane weights", dict(zip(names, lw.round(2).tolist())))
    best, best_state, n = -1.0, None, len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            yb = {k: v[j].to(dev) for k, v in ytr.items()}
            loss = compute_losses(model(Xtr[j].to(dev)), yb, mode, w, pos_w, bce_m,
                                  dev, focal_gamma, bg_weight, lane_w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()) * len(j)
        sched.step()
        v = evaluate(model, Xva, yva, dev, mode, names)
        if verbose:
            print("  ep%02d loss %.4f  val macro-F1 %.3f  bal-acc %.3f"
                  % (ep + 1, tot / n, v["macro_f1"], v["balanced_acc"]))
        if v["macro_f1"] > best:
            best = v["macro_f1"]
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
            # Recorded so deploy.py can retrain on every subject for exactly the
            # number of epochs validation selected.
            model.best_epoch = ep + 1
    if keep_last:
        # Final all-subject retrain: the "validation" set is inside the training
        # data, so selecting on it would be meaningless. The epoch count itself
        # was already chosen on genuinely held-out subjects.
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_state
