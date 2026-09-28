"""Figures for the round-2 report (relabelled corpus, 10 fps arms, export)."""
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
OUT = "figs_round2"; os.makedirs(OUT, exist_ok=True)

def runs(tag):
    return {r["test_uid"]: r for r in json.load(open("bench/results/loso_%s.json" % tag))["runs"]}

ARMS = [("B0_f30a_base", "B0\nbaseline\n30 fps 4 s"), ("F1_selected", "F1\n7 panels\n(121)"),
        ("C1_screw_clamp", "C1\nScrew/Clamp\n2x"), ("L1_lstm", "L1\nLSTM"),
        ("S1_10fps_4s", "S1\n10 fps\n4 s"), ("S2_10fps_8s", "S2\n10 fps\n8 s"),
        ("S3_10fps_8s_bg05", "S3\n10 fps 8 s\nbg 0.5")]
COL = [NAVY, GREY, GREY, GREY, BLUE, BLUE, ORANGE]
R = {t: runs(t) for t, _ in ARMS}
U = sorted(R["B0_f30a_base"])

# ---- Fig 1: all arms, macro-F1 ----
fig, ax = plt.subplots(figsize=(7.0, 3.1))
m = [np.mean([R[t][u]["macro_f1"] for u in U]) for t, _ in ARMS]
s = [np.std([R[t][u]["macro_f1"] for u in U]) for t, _ in ARMS]
ax.bar(range(len(ARMS)), m, yerr=s, capsize=3, width=0.6, color=COL,
       error_kw={"ecolor": "#5A6470", "elinewidth": 1})
for i, v in enumerate(m):
    ax.text(i, v + s[i] + 0.012, "%.3f" % v, ha="center", fontsize=8.5,
            fontweight="bold" if ARMS[i][0].startswith("S3") else "normal")
ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l in ARMS], fontsize=7.6)
ax.set_ylabel("macro-F1, 15-fold LOSO (mean ± sd)"); ax.set_ylim(0, 0.68)
ax.set_title("Round 2 arms (relabelled corpus, 7 task classes)", fontsize=10, color=NAVY, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "/fig1_arms.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: paired per-subject differences ----
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=True)
for ax, (a, b, ttl) in zip(axes, [("B0_f30a_base", "S1_10fps_4s", "S1 − B0  (10 fps vs 30 fps)"),
                                  ("S1_10fps_4s", "S2_10fps_8s", "S2 − S1  (8 s vs 4 s window)"),
                                  ("S2_10fps_8s", "S3_10fps_8s_bg05", "S3 − S2  (bg weight 0.5)")]):
    d = np.array([R[b][u]["macro_f1"] - R[a][u]["macro_f1"] for u in U])
    ax.bar(range(15), d, color=[GREEN if x > 0 else RED for x in d], width=0.65)
    ax.axhline(0, color="#5A6470", lw=0.9)
    ax.axhline(d.mean(), color=NAVY, ls="--", lw=1.1)
    ax.set_title("%s\nmean %+.3f, better on %d/15" % (ttl, d.mean(), int((d > 0).sum())),
                 fontsize=8.5, color=NAVY)
    ax.set_xticks(range(15)); ax.set_xticklabels([str(u) for u in U], fontsize=6.5)
    ax.set_xlabel("held-out subject (uid)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Δ macro-F1")
fig.tight_layout(); fig.savefig(OUT + "/fig2_paired.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: per-class recall, B0 / S1 / S3 ----
SEL = [("B0_f30a_base", "B0 (30 fps, 4 s)", NAVY), ("S1_10fps_4s", "S1 (10 fps, 4 s)", BLUE),
       ("S3_10fps_8s_bg05", "S3 (10 fps, 8 s, bg 0.5)", ORANGE)]
cls = list(R["B0_f30a_base"][U[0]]["per_class_recall"])
fig, ax = plt.subplots(figsize=(7.4, 2.9))
x = np.arange(len(cls)); w = 0.26
for k, (t, lab, col) in enumerate(SEL):
    v = [np.mean([R[t][u]["per_class_recall"][c] for u in U]) for c in cls]
    ax.bar(x + (k - 1) * w, v, w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(cls, fontsize=8)
ax.set_ylabel("recall (15-fold mean)"); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
ax.set_title("Per-class recall", fontsize=10, color=NAVY, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "/fig3_perclass.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: confusion matrices B0 vs S3 ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
for ax, (t, ttl) in zip(axes, [("B0_f30a_base", "B0 — 30 fps, 4 s"),
                               ("S3_10fps_8s_bg05", "S3 — 10 fps, 8 s (exported)")]):
    E = json.load(open("bench/eval_round2/%s/evaluation.json" % t))
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
fig.tight_layout(); fig.savefig(OUT + "/fig4_confusion.png", bbox_inches="tight"); plt.close(fig)

print("wrote", len(os.listdir(OUT)), "figures to", OUT)
