"""Figures for the round-1 report (annotation team's feature/idle question)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 200,
    "axes.edgecolor": "#AAAEB3", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#E4E7EA", "grid.linewidth": 0.7, "axes.axisbelow": True,
})
NAVY, BLUE, ORANGE, GREY, GREEN, RED = "#1A3A5C", "#4A8DBF", "#E08A3C", "#9AA3AB", "#4C9A6A", "#C0504D"
OUT = "figs_round1"; os.makedirs(OUT, exist_ok=True)

def runs(tag):
    return {r["test_uid"]: r for r in json.load(open("bench/results/loso_%s.json" % tag))["runs"]}

ARMS = [("N1_full_bg", "N1\nall 16 panels\n(251)"), ("N3_red_bg", "N3\n6 panels\n(89)"),
        ("N5_redvel_bg", "N5\n6 panels + vel\n(137)")]
R = {t: runs(t) for t, _ in ARMS}
U = sorted(R["N1_full_bg"])

# ---- Fig 1: feature sets, macro-F1 ----
fig, ax = plt.subplots(figsize=(5.6, 3.1))
m = [np.mean([R[t][u]["macro_f1"] for u in U]) for t, _ in ARMS]
s = [np.std([R[t][u]["macro_f1"] for u in U]) for t, _ in ARMS]
bars = ax.bar(range(3), m, yerr=s, capsize=3, width=0.58, color=[NAVY, ORANGE, BLUE],
              error_kw={"ecolor": "#5A6470", "elinewidth": 1})
for i, v in enumerate(m):
    ax.text(i, v + s[i] + 0.012, "%.3f" % v, ha="center", fontsize=8.5,
            fontweight="bold" if i == 0 else "normal")
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in ARMS], fontsize=7.8)
ax.set_ylabel("macro-F1, 15-fold LOSO (mean ± sd)"); ax.set_ylim(0, 0.68)
ax.set_title("Feature set comparison (idle → bg head)", fontsize=10, color=NAVY, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "/fig1_feature_sets.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: paired per-subject differences ----
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), sharey=True)
for ax, (a, b, ttl) in zip(axes, [("N1_full_bg", "N3_red_bg", "N3 − N1  (6 panels vs all)"),
                                  ("N3_red_bg", "N5_redvel_bg", "N5 − N3  (adding velocity x/y/z)")]):
    d = np.array([R[b][u]["macro_f1"] - R[a][u]["macro_f1"] for u in U])
    ax.bar(range(15), d, color=[GREEN if x > 0 else RED for x in d], width=0.65)
    ax.axhline(0, color="#5A6470", lw=0.9)
    ax.axhline(d.mean(), color=NAVY, ls="--", lw=1.1)
    wins = int((d > 0).sum())
    ax.set_title("%s\nmean %+.3f, better on %d/15" % (ttl, d.mean(), wins),
                 fontsize=8.8, color=NAVY)
    ax.set_xticks(range(15)); ax.set_xticklabels([str(u) for u in U], fontsize=6.5)
    ax.set_xlabel("held-out subject (uid)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Δ macro-F1")
fig.tight_layout(); fig.savefig(OUT + "/fig2_paired.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: per-class recall ----
cls = list(R["N1_full_bg"][U[0]]["per_class_recall"])
fig, ax = plt.subplots(figsize=(7.4, 2.9))
x = np.arange(len(cls)); w = 0.26
for k, ((t, lab), col) in enumerate(zip(ARMS, [NAVY, ORANGE, BLUE])):
    v = [np.mean([R[t][u]["per_class_recall"][c] for u in U]) for c in cls]
    ax.bar(x + (k - 1) * w, v, w, color=col, label=lab.replace("\n", " "))
ax.set_xticks(x); ax.set_xticklabels(cls, fontsize=8)
ax.set_ylabel("recall (15-fold mean)"); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
ax.set_title("Per-class recall by feature set", fontsize=10, color=NAVY, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "/fig3_perclass.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: fair idle comparison ----
F = json.load(open("bench/results/fair_idle_round1.json"))
names = F["names"]; P = {t: np.array(v) for t, v in F["per_fold_f1"].items()}
fig, ax = plt.subplots(figsize=(7.4, 2.9))
pairs = [("N3_red_bg", "N3 bg head", ORANGE), ("N4_red_idle", "N4 idle as class", "#F2C49B"),
         ("N5_redvel_bg", "N5 bg head", BLUE), ("N6_redvel_idle", "N6 idle as class", "#A9CBE3")]
x = np.arange(len(names)); w = 0.2
for k, (t, lab, col) in enumerate(pairs):
    ax.bar(x + (k - 1.5) * w, P[t].mean(0), w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("F1, same 7-way decision"); ax.set_ylim(0, 0.9)
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left")
ax.set_title("Idle handling: bg head vs 7th class, scored identically", fontsize=10,
             color=NAVY, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "/fig4_idle_fair.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 5: panel importance heatmap ----
I = json.load(open("bench/results/feat_importance_N1_full_bg.json"))
labs = I["labels"]; drop = I["drop"]
order = sorted(drop, key=lambda k: -drop[k][labs.index("MACRO")])
M = np.array([drop[k] for k in order])
fig, ax = plt.subplots(figsize=(7.4, 5.0))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.2, aspect="auto")
ax.set_xticks(range(len(labs))); ax.set_xticklabels([l[:10] for l in labs], rotation=35, ha="right", fontsize=7.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.3,
                color="white" if v > 0.12 else "#2B3137")
ax.set_title("Panel importance: F1 lost when the panel is shuffled (N1, 15 folds)",
             fontsize=9.5, color=NAVY, fontweight="bold")
for s_ in ax.spines.values(): s_.set_visible(False)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
fig.tight_layout(); fig.savefig(OUT + "/fig5_panel_importance.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 6: body-part importance ----
J = json.load(open("bench/results/feat_joint_importance_N1_full_bg.json"))
groups = ["ARMS (shoulder,elbow,wrist)", "WRISTS only", "TORSO+HEAD (spine..head)",
          "LOWER BODY (hips,knees,ankles)", "LEGS (knees,ankles)"]
jl = J["labels"]; mi = jl.index("MACRO")
fig, ax = plt.subplots(figsize=(6.2, 2.6))
vals = [J["drop"][g][mi] for g in groups]
sds = [np.std([f[mi] for f in J["per_fold"][g]]) for g in groups]
ax.barh(range(len(groups)), vals, xerr=sds, color=[NAVY, BLUE, BLUE, ORANGE, ORANGE],
        height=0.6, error_kw={"ecolor": "#5A6470", "elinewidth": 1})
ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=8); ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + sds[i] + 0.006, i, "%.3f" % v, va="center", fontsize=8)
ax.set_xlabel("macro-F1 lost when shuffled (± sd over folds)")
ax.set_title("Body-part importance (N1)", fontsize=10, color=NAVY, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "/fig6_body_parts.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 7: confusion matrices N1 vs N5 ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
for ax, (t, ttl) in zip(axes, [("N1_full_bg", "N1 — all 16 panels"), ("N5_redvel_bg", "N5 — 6 panels + velocity")]):
    E = json.load(open("bench/eval_round1/%s/evaluation.json" % t))
    n = E["names"]; M = np.array(E["row_normalised"])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=0.85)
    ax.set_xticks(range(len(n))); ax.set_yticks(range(len(n)))
    ax.set_xticklabels([s_[:8] for s_ in n], rotation=40, ha="right", fontsize=7.5)
    ax.set_yticklabels([s_[:12] for s_ in n], fontsize=7.5)
    for i in range(len(n)):
        for j in range(len(n)):
            if M[i, j] >= 0.01:
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.45 else "#2B3137",
                        fontweight="bold" if i == j else "normal")
    ax.set_title(ttl, fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xlabel("predicted"); ax.grid(False)
    for s_ in ax.spines.values(): s_.set_visible(False)
axes[0].set_ylabel("true")
fig.tight_layout(); fig.savefig(OUT + "/fig7_confusion.png", bbox_inches="tight"); plt.close(fig)

print("wrote", len(os.listdir(OUT)), "figures to", OUT)
