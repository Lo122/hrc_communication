# -*- coding: utf-8 -*-
"""Build the round-2 report as a .docx."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

NAVY = RGBColor(0x1A, 0x3A, 0x5C)
BLUE = RGBColor(0x2C, 0x5F, 0x8A)
GREY = RGBColor(0x6B, 0x77, 0x83)
DARK = RGBColor(0x44, 0x51, 0x5E)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(0.9)
sec.left_margin = sec.right_margin = Inches(1.0)

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15


def shade(cell, hexcolor):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolor)
    cell._tc.get_or_add_tcPr().append(el)


def para_shade(p, hexcolor):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolor)
    p._p.get_or_add_pPr().append(el)


def bottom_border(p, color="1A3A5C", size="6"):
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), size)
    b.set(qn("w:space"), "4")
    b.set(qn("w:color"), color)
    bdr.append(b)
    pPr.append(bdr)


def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = NAVY
    bottom_border(p)


def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = BLUE


def rich(parts, after=6, bullet=False):
    p = doc.add_paragraph(style="List Bullet" if bullet else None)
    p.paragraph_format.space_after = Pt(after)
    for part in parts:
        if isinstance(part, str):
            p.add_run(part)
        else:
            txt, o = part
            r = p.add_run(txt)
            r.bold = o.get("b", False)
            r.italic = o.get("i", False)
            if o.get("m"):
                r.font.name = "Consolas"
                r.font.size = Pt(9)


def body(text, after=6):
    rich([text], after=after)


def code(lines):
    for i, ln in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8 if i == len(lines) - 1 else 0)
        p.paragraph_format.left_indent = Inches(0.12)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(ln if ln else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
        para_shade(p, "F4F5F7")


def table(headers, rows, widths, mono=True, highlight=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htxt in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        if i:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(htxt)
        r.bold = True
        r.font.size = Pt(9)
        shade(hdr[i], "E8EAED")
    for row in rows:
        cells = t.add_row().cells
        hi = highlight(row) if highlight else False
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            if i:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(9)
            r.bold = hi
            if mono and i:
                r.font.name = "Consolas"
            if hi:
                shade(cells[i], "FFF6E0")
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(path, width=6.3, caption=None):
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(caption)
        r.font.size = Pt(8.5); r.italic = True; r.font.color.rgb = GREY


import json
import numpy as np

F = "figs_round2/"
E = json.load(open("bench/eval_round2/S3_10fps_8s_bg05/evaluation.json"))
CN, CM = E["names"], np.array(E["row_normalised"])
off = sorted(((CM[i, j], CN[i], CN[j]) for i in range(len(CN)) for j in range(len(CN)) if i != j),
             reverse=True)[:4]

# ---------------- Title ----------------
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
r = p.add_run("Round 2 — Relabelled Corpus, Live Frame Rate, Export")
r.bold = True; r.font.size = Pt(19); r.font.color.rgb = NAVY
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
r = p.add_run("Labels from LSTM_HRC commit f30a14d (Lift kept), first deployable model")
r.font.size = Pt(12); r.font.color.rgb = DARK
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(14)
r = p.add_run("ITECH M.Sc. Thesis · Recognition Layer · 28 September 2026 · "
              "7 arms × 15-fold leave-one-subject-out")
r.font.size = Pt(9.5); r.font.color.rgb = GREY
bottom_border(p, color="C3C8CE", size="4")

h1("Summary")
rich(["Round 1 found that its corpus labelled 41.6% of real lifting frames as idle. This round "
      "retrains on the same features relabelled with the annotation team’s current labeller "
      "(Lift kept as a class), tests the changes queued in Round 1, and exports the first model "
      "that can run in the live loop."])
for parts in [
    [("Training at the live rate (10 fps) costs nothing: ", {"b": 1}),
     "S1 vs B0 +0.007, better on 11 of 15 subjects (p = 0.17). The frame-rate mismatch "
     "identified in Round 1 can be removed without an accuracy penalty."],
    [("A longer window helps: ", {"b": 1}),
     "8 s instead of 4 s at 10 fps, +0.018 (S1 → S2, 10 / 15, p = 0.035). S2 is +0.025 over "
     "the baseline, better on 13 of 15 subjects (p = 0.0003)."],
    [("Raising the idle-head weight to 0.5 gives no measurable gain: ", {"b": 1}),
     "S2 → S3 +0.009, 7 / 15, p = 0.45; idle F1 unchanged (0.623 → 0.620)."],
    [("No effect: ", {"b": 1}),
     "the data-driven 7-panel set (F1, −0.002), Screw/Clamp loss weighting (C1, +0.007) and "
     "the LSTM (L1, −0.008). None is significant."],
    [("Exported: S3, macro-F1 0.499 ± 0.086. ", {"b": 1}),
     "Chosen by a rule fixed before the runs: highest LOSO score among the 10 fps arms. S2 is "
     "statistically equivalent; the choice between them does not matter for accuracy."],
]:
    rich(parts, bullet=True)

h1("1. Setup")
body("Corpus: data_f30a14d/ — the Round 1 features (93 takes, 15 subjects), labels regenerated "
     "with LSTM_HRC commit f30a14d. Lift is trimmed to the part before Place/Align and kept as a "
     "class; idle (“No Related Task”) is every frame no annotation covers and goes to the "
     "separate bg head. Seven task classes.")
table(["Arm", "Change from B0", "Rate", "Window"],
      [["B0", "baseline: all 16 panels (251 dims), GRU", "30 fps", "4 s (120)"],
       ["F1", "7 panels chosen by importance (121 dims)", "30 fps", "4 s (120)"],
       ["C1", "Screw and Clamp Coupling weighted 2× in the loss", "30 fps", "4 s (120)"],
       ["L1", "LSTM instead of GRU", "30 fps", "4 s (120)"],
       ["S1", "every 3rd frame", "10 fps", "4 s (40)"],
       ["S2", "every 3rd frame, 10 epochs", "10 fps", "8 s (80)"],
       ["S3", "S2 + idle-head loss weight 0.2 → 0.5", "10 fps", "8 s (80)"]],
      [0.6, 3.4, 0.9, 1.3], mono=False)
body("Common to all: peak-vector targets, mirror augmentation, pos_weight cap 2.0, best epoch "
     "chosen on two validation subjects, 15-fold leave-one-subject-out. Macro-F1 is over the "
     "seven task classes on windows whose true label is a task.")
rich([("Not comparable with Round 1. ", {"b": 1}),
      "Round 1 scored six classes on different labels. B0 (0.465) is the baseline for this "
      "round; Round 1’s 0.499 is not."])

h1("2. Results")
figure(F + "fig1_arms.png", 6.2,
       "Figure 1. Macro-F1, mean ± sd across 15 held-out subjects. Orange: exported model.")
table(["Arm", "macro-F1", "bal. acc.", "idle F1", "lane AP", "progress MAE"],
      [["B0  baseline", "0.465 ± 0.074", "0.479", "0.612", "0.509", "24.2"],
       ["F1  7 panels", "0.463 ± 0.067", "0.482", "0.607", "0.516", "24.1"],
       ["C1  Screw/Clamp 2×", "0.472 ± 0.075", "0.487", "0.616", "0.511", "24.3"],
       ["L1  LSTM", "0.456 ± 0.072", "0.470", "0.604", "0.501", "24.4"],
       ["S1  10 fps, 4 s", "0.472 ± 0.075", "0.485", "0.605", "0.517", "24.0"],
       ["S2  10 fps, 8 s", "0.490 ± 0.081", "0.502", "0.623", "0.541", "24.1"],
       ["S3  S2 + bg 0.5", "0.499 ± 0.086", "0.514", "0.620", "0.559", "24.1"]],
      [1.7, 1.2, 0.8, 0.8, 0.8, 1.0], highlight=lambda r: r[0].startswith("S3"))
table(["Comparison", "Δ macro-F1", "Subjects improved", "Wilcoxon p"],
      [["B0 → F1  (7 panels)", "−0.002", "4 / 15", "0.60"],
       ["B0 → C1  (class weights)", "+0.007", "8 / 15", "0.45"],
       ["B0 → L1  (LSTM)", "−0.008", "6 / 15", "0.19"],
       ["B0 → S1  (10 fps)", "+0.007", "11 / 15", "0.17"],
       ["S1 → S2  (8 s window)", "+0.018", "10 / 15", "0.035"],
       ["S2 → S3  (bg weight 0.5)", "+0.009", "7 / 15", "0.45"],
       ["B0 → S2", "+0.025", "13 / 15", "0.0003"],
       ["B0 → S3", "+0.034", "12 / 15", "0.0034"]],
      [2.6, 1.1, 1.4, 1.1])
figure(F + "fig2_paired.png", 6.6,
       "Figure 2. Per-subject differences along the 10 fps path. Same 15 held-out subjects in "
       "every arm, so the comparison is paired.")
rich([("Reading. ", {"b": 1}),
      "Only the window length reliably moves the score. At 10 fps an 8 s window is 80 "
      "recurrent steps — fewer than B0’s 120 — so the longer context comes at no extra "
      "runtime cost."])

h2("Per class")
figure(F + "fig3_perclass.png", 6.4, "Figure 3. Recall per class, 15-fold mean.")
table(["Class", "B0", "S1", "S2", "S3"],
      [["Pull Cables", "0.29", "0.30", "0.31", "0.28"], ["Lift", "0.48", "0.50", "0.45", "0.50"],
       ["Place", "0.45", "0.47", "0.46", "0.47"], ["Align", "0.62", "0.62", "0.62", "0.66"],
       ["Screw", "0.82", "0.82", "0.85", "0.86"],
       ["Connect Cables", "0.31", "0.31", "0.38", "0.37"],
       ["Clamp Coupling", "0.40", "0.39", "0.44", "0.45"]],
      [2.0, 0.9, 0.9, 0.9, 0.9])
body("The 8 s window’s gain is concentrated on Connect Cables (+0.07) and Clamp Coupling "
     "(+0.05) — the chest-height tasks Round 1 found the model confuses. Longer context "
     "helps separate them; loss weighting (C1) did not. Pull Cables remains the weakest class "
     "(0.28–0.33 in every arm).")
figure(F + "fig4_confusion.png", 6.6,
       "Figure 4. Confusion matrices pooled over all 15 held-out subjects, row-normalised.")
rich(["Largest confusions in S3 (true → predicted): "
      + "; ".join("%s → %s %.2f" % (a, b, v) for v, a, b in off) + "."])
rich([("Lift is now learnable: ", {"b": 1}),
      "recall 0.45–0.50 in every arm. In Round 1 the class did not exist and its frames were "
      "labelled idle."])

h1("3. Exported model")
body("The export rule was fixed before the runs: the live loop runs at 10 Hz, so only S1–S3 "
     "are deployable, and the one with the highest LOSO macro-F1 is exported. That is S3. The "
     "deployed weights are not taken from any single fold:")
for parts in [
    ["Stage 1: train on 13 subjects, choose the epoch count on subjects 2 and 7 → 4 epochs "
     "(validation macro-F1 0.448)."],
    ["Stage 2: retrain on all 15 subjects for 4 epochs (208,598 windows)."],
]:
    rich(parts, bullet=True)
table(["File (models/S3_10fps_8s_bg05/)", "Contents"],
      [["model_weights.pth", "GRU state_dict, 251 inputs, 7 task classes"],
       ["standardization.npz", "mean and std per input column, with column names"],
       ["standardization_by_panel.npz", "same, grouped by panel (layout norm_feat_rlt.py reads)"],
       ["feature_selection.json", "panels, joints, column order, transforms, window 80 / stride 3"],
       ["config.json", "classes, heads, training settings, trigger, expected accuracy"],
       ["model_bundle.pt", "all of the above in one file"],
       ["README.md", "how to feed the model and read its outputs"]],
      [2.6, 3.8], mono=False)
body("The export pipeline was checked before the runs on a throwaway configuration: the model "
     "input rebuilt from a raw take using only the exported files matched the training pipeline "
     "exactly (max difference 0), and model outputs were identical.")

h2("Before it can run live")
for parts in [
    [("Trigger threshold. ", {"b": 1}),
     "The trigger formula was tuned on earlier models; config.json marks it unvalidated with no "
     "threshold. The event-level trigger evaluation must be re-run on this model."],
    [("Runtime loader. ", {"b": 1}),
     "recognition_manager.py still loads the 2-output AssistLSTM. The new model has four heads "
     "(task, progress, mistake, idle) and multi-label task scores (independent sigmoids, not "
     "softmax)."],
    [("Feed rate. ", {"b": 1}),
     "The model expects one sample every 100 ms and an 80-sample (8 s) window."],
]:
    rich(parts, bullet=True)

h1("4. Recommendations")
for parts in [
    [("Deploy at 10 fps with an 8 s window. ", {"b": 1}),
     "Matches the live loop, and is the only change this round that reliably improved accuracy."],
    [("Keep all 16 panels and the GRU. ", {"b": 1}),
     "The 7-panel set and the LSTM are no better. If a smaller input is needed, the 7-panel "
     "set costs nothing measurable (−0.002)."],
    [("Drop class weighting and the idle-weight change as levers. ", {"b": 1}),
     "Neither moved the score beyond noise."],
    [("Next: test longer windows (12–16 s at 10 fps), ", {"b": 1}),
     "since the gain from 4 s to 8 s shows no sign of flattening yet; and regenerate the "
     "dataset at 10 fps so velocity smoothing and the MotionBERT clip match the live loop "
     "(Round 1, Section 6)."],
    [("Add participants. ", {"b": 1}),
     "Subject-to-subject spread (sd 0.086) is still larger than every effect measured here."],
]:
    rich(parts, bullet=True)

h1("Appendix: files")
table(["Path", "Contents"],
      [["bench/results/loso_<arm>.json", "per-fold metrics and run configuration"],
       ["bench/results/folds/<arm>_uid*.pth", "trained weights, one per held-out subject"],
       ["bench/eval_round2/<arm>/", "metrics.csv, confusion_matrix.csv, per_class.csv"],
       ["bench/logs/<arm>.log, deploy.log", "training and export logs"],
       ["models/S3_10fps_8s_bg05/", "exported model"],
       ["archive_round2/", "all of the above plus source code, figures and MANIFEST.json"]],
      [3.0, 3.4], mono=False)

doc.save("Round2_Report.docx")
print("saved Round2_Report.docx")


