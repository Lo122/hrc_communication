# -*- coding: utf-8 -*-
"""Build the recognition training results report as a .docx."""
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


F = "figs_round1/"

# ---------------- Title ----------------
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
r = p.add_run("Round 1 — Feature Sets and Idle Handling")
r.bold = True; r.font.size = Pt(19); r.font.color.rgb = NAVY
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
r = p.add_run("Re-annotated corpus (Lift removed, “No Related Task” labelled)")
r.font.size = Pt(12); r.font.color.rgb = DARK
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(14)
r = p.add_run("ITECH M.Sc. Thesis · Recognition Layer · 27 September 2026 · "
              "5 arms × 15-fold leave-one-subject-out")
r.font.size = Pt(9.5); r.font.color.rgb = GREY
bottom_border(p, color="C3C8CE", size="4")

h1("Summary")
rich(["This round answers two questions from the annotation team, on their re-annotated "
      "corpus: ", ("can the model use a reduced feature set", {"b": 1}), ", and ",
      ("should idle time be a separate output or a seventh class", {"b": 1}), "."])
for parts in [
    [("The reduced six-panel set costs accuracy: ", {"b": 1}),
     "macro-F1 0.499 → 0.464, worse on 13 of 15 subjects (p = 0.0015)."],
    [("Adding velocity x/y/z recovers almost none of it: ", {"b": 1}),
     "+0.006 (p = 0.49). The six-panel-plus-velocity set is still 0.028 below the full set "
     "(p = 0.0026)."],
    [("The reason is measurable: ", {"b": 1}),
     "the per-axis velocity panels are among the least-used inputs, while the speed and "
     "acceleration magnitudes that the reduced set omits are among the most-used."],
    [("Idle as a separate head vs idle as a seventh class makes no difference ", {"b": 1}),
     "once both are scored identically (−0.011, p = 0.12; −0.003, p = 0.93). An "
     "apparent large gap in the first evaluation was a scoring artefact, corrected here."],
    [("Idle F1 (0.62) is not comparable with the old corpus (0.51): ", {"b": 1}),
     "idle is derived, not annotated, and the two corpora define it differently. "
     "See Section 7 — this corpus also mislabels part of the lifting motion as idle."],
]:
    rich(parts, bullet=True)

h1("1. Setup")
body("Corpus: hrc_communication/other — 93 takes, 15 subjects, 3 camera views. The "
     "annotation team removed Lift (lane 1 is empty in every take) and added an explicit "
     "“No Related Task” label covering 18.2% of frames. The six task classes are "
     "numerically identical to the previous corpus.")
table(["Arm", "Feature panels", "Dims", "Idle handling"],
      [["N1", "all 16", "251", "separate bg head"],
       ["N3", "6 (position x/y/z, elevation, joint angles, distance)", "89", "separate bg head"],
       ["N4", "same 6", "89", "7th class"],
       ["N5", "same 6 + velocity x/y/z", "137", "separate bg head"],
       ["N6", "same 6 + velocity x/y/z", "137", "7th class"]],
      [0.6, 3.5, 0.6, 1.6], mono=False)
body("All arms: GRU (hidden 128), 120-frame windows at 30 fps, peak-vector targets, mirror "
     "augmentation (one copy per recording), pos_weight capped at 2.0, 12 epochs with the "
     "best epoch chosen on two validation subjects, 15-fold leave-one-subject-out. N2 "
     "(all panels, idle as class) was dropped at the team’s request.")

h1("2. Feature sets")
figure(F + "fig1_feature_sets.png", 4.6,
       "Figure 1. Macro-F1 over the six task classes, mean ± sd across 15 held-out subjects.")
table(["Arm", "macro-F1", "idle F1", "progress MAE"],
      [["N1  all 16 panels", "0.499 ± 0.089", "0.621", "23.5"],
       ["N3  6 panels", "0.464 ± 0.093", "0.612", "24.2"],
       ["N5  6 panels + velocity", "0.471 ± 0.093", "0.622", "23.7"]],
      [2.4, 1.5, 1.1, 1.3], highlight=lambda r: r[0].startswith("N1"))
figure(F + "fig2_paired.png", 6.5,
       "Figure 2. Per-subject differences. Every arm holds out the same 15 subjects, so the "
       "comparison is paired.")
table(["Comparison", "Δ macro-F1", "Subjects improved", "Wilcoxon p"],
      [["N1 → N3  (reduce to 6 panels)", "−0.035", "2 / 15", "0.0015"],
       ["N3 → N5  (add velocity x/y/z)", "+0.006", "9 / 15", "0.49"],
       ["N1 → N5  (6 panels + velocity vs all)", "−0.028", "3 / 15", "0.0026"]],
      [2.9, 1.1, 1.3, 1.1])
rich([("Reading. ", {"b": 1}),
      "The reduced set is reliably worse, not a noise effect: 13 of 15 subjects lose. "
      "Adding the per-axis velocity panels does not close the gap."])

h2("Per class")
figure(F + "fig3_perclass.png", 6.4, "Figure 3. Recall per class, 15-fold mean.")
table(["Class", "N1", "N3", "N5"],
      [["Pull Cables", "0.48", "0.47", "0.47"], ["Place", "0.50", "0.46", "0.50"],
       ["Align", "0.60", "0.60", "0.58"], ["Screw", "0.81", "0.75", "0.79"],
       ["Connect Cables", "0.30", "0.27", "0.25"], ["Clamp Coupling", "0.38", "0.37", "0.38"]],
      [2.2, 1.0, 1.0, 1.0])
body("The losses from reducing the feature set concentrate on Screw (−0.06) and Place "
     "(−0.04), the two classes defined by motion rather than posture. Velocity partly "
     "restores both (Screw 0.79, Place 0.50) but costs Connect (0.25).")
figure(F + "fig7_confusion.png", 6.6,
       "Figure 4. Confusion matrices pooled over all 15 held-out subjects, row-normalised.")
body("The dominant error is unchanged across feature sets: Connect Cables, Clamp Coupling "
     "and Screw are confused with one another. All three are performed standing at the panel "
     "with the hands near chest height.")

h1("3. Why: which features the model uses")
body("Each panel of the trained N1 models was shuffled across windows in turn, and the loss "
     "in F1 measured per class. A panel whose shuffling costs nothing is not being used.")
figure(F + "fig5_panel_importance.png", 6.2,
       "Figure 5. F1 lost when each panel is shuffled. Rows sorted by macro-F1 loss.")
rich([("Magnitudes matter, per-axis components barely do. ", {"b": 1}),
      "joint_speed (0.097) and joint_acceleration (0.057) are among the seven most-used "
      "panels. The six per-axis velocity and acceleration panels — 96 of the 251 input "
      "columns — each cost at most 0.028. This is why N5 does not recover N3’s loss: "
      "it adds the least-used panels while the reduced set still lacks the most-used ones."])
table(["Motion", "Most-used features", "Interpretation"],
      [["Screw", "speed, joint angles, height (z)", "repetitive hand motion, fixed posture"],
       ["Clamp", "height (z), azimuth, depth (y)", "where the hands are"],
       ["Place", "speed, acceleration, joint angles", "a burst of motion upward"],
       ["Align", "joint angles (0.34), distance", "arms-raised posture"],
       ["Connect", "azimuth, speed, depth (y)", "hand direction relative to the body"],
       ["Idle", "height (z), distance", "arms down, at rest"]],
      [1.0, 2.6, 2.7], mono=False)

h2("Body parts")
figure(F + "fig6_body_parts.png", 5.4,
       "Figure 6. Macro-F1 lost when every feature of a body part is shuffled together.")
rich(["The arms dominate (0.276), but ", ("the lower body is not negligible", {"b": 1}),
      ": shuffling hips, knees and ankles costs 0.087, about three-quarters of the torso and "
      "head. The legs matter most for Place (0.130), Clamp (0.113) and Connect (0.095) — "
      "the push uses the legs, and stance distinguishes the chest-height tasks. The lower "
      "body is therefore kept."])
body("Individually each leg joint costs only 0.016–0.028: they are redundant with each "
     "other but not with the upper body.")

h1("4. Idle handling")
rich(["A first evaluation showed the seventh-class arms far behind (N4 0.404 vs N3 0.464). ",
      ("That gap was a scoring artefact.", {"b": 1}),
      " The bg-head arms excluded idle frames from the task score and were never penalised "
      "when the bg head wrongly called a task frame idle; the seventh-class arms were. Both "
      "were therefore converted to the same seven-way decision and scored on every window."])
figure(F + "fig4_idle_fair.png", 6.4,
       "Figure 7. Seven-way F1 with identical scoring for both idle strategies.")
table(["Comparison", "Δ six-task F1", "7th-class arm better", "Wilcoxon p"],
      [["N3 (head) → N4 (class)", "−0.011", "4 / 15", "0.12"],
       ["N5 (head) → N6 (class)", "−0.003", "8 / 15", "0.93"]],
      [2.4, 1.3, 1.5, 1.1])
body("Neither difference is significant. Idle F1 is 0.61–0.62 for the head and 0.59 "
     "for the class. The separate head is kept because it is marginally ahead and lets the "
     "runtime tune the idle threshold independently of the task decision.")
rich([("Pull Cables is weak in the seven-way view (F1 0.14–0.20). ", {"b": 1}),
      "It is most often confused with idle: pulling cables is a low-motion, arms-down "
      "posture similar to standing at rest."])

h1("5. Recommendations")
for parts in [
    [("Keep all 16 panels for now. ", {"b": 1}),
     "The six-panel set loses 0.035 reliably."],
    [("If a smaller input is required, reduce by importance, not by panel type. ", {"b": 1}),
     "A seven-panel set chosen from Figure 5 (121 dims, including speed and acceleration "
     "magnitudes) is queued tonight as F1."],
    [("Keep the lower body. ", {"b": 1}), "It carries 0.087 macro-F1, concentrated on the weak classes."],
    [("Keep idle as a separate head. ", {"b": 1}), "Equivalent accuracy, more control at runtime."],
    [("Add participants. ", {"b": 1}),
     "Measured earlier: each additional real subject is worth about +0.011, and the curve has "
     "not saturated. No feature choice in this round moved the score by more than that of "
     "three subjects."],
]:
    rich(parts, bullet=True)

h1("6. Next: tonight’s runs, and a deployment mismatch")
rich([("The training data is 30 fps; the live loop feeds the model at about 10 Hz. ", {"b": 1}),
      "The recurrent model counts samples, not seconds, so a 120-sample window covers 4 s in "
      "training but 12 s live, and each step is three times longer. Every number in this "
      "report is at 30 fps. A --stride option now trains at the live rate; S1 retrains N1 at "
      "10 fps to measure the cost."])
table(["Arm", "Tests"],
      [["F1", "data-driven 7-panel set, 121 dims"],
       ["C1", "Screw and Clamp Coupling weighted 2× in the loss"],
       ["S1", "N1 retrained at 10 fps, 4 s window (40 samples)"],
       ["S2", "10 fps, 8 s window (80 samples)"],
       ["S3", "S2 with the idle-head loss weight raised 0.2 → 0.5"],
       ["L1", "LSTM with N1’s settings (runtime and team code use LSTM)"]],
      [0.8, 5.6], mono=False)
body("Two further settings are defined in frames rather than seconds and also differ live: "
     "velocity smoothing (0.3 s offline, 0.9 s live) and the MotionBERT clip (2.7 s offline, "
     "8.1 s live). Fixing these requires regenerating the dataset at 10 fps.")

h1("7. Correction: how the idle label is made, and a labelling problem")
rich(["An earlier draft described “No Related Task” as a human annotation and credited "
      "it for the rise in idle F1. ", ("That was wrong.", {"b": 1}),
      " Idle is derived: every frame no ELAN annotation covers is scored as idle, with the "
      "smoothing ramp placed inside the gap. The 0.51 and 0.62 figures are measured against "
      "two different definitions of idle and cannot be compared."])
rich(["More importantly, the corpus used in this round (hrc_communication/other) was built by "
      "a labeller that ", ("removed Lift entirely", {"b": 1}),
      ". The annotation team’s current labeller (LSTM_HRC commit f30a14d) instead trims "
      "each Lift span to the part before Place or Align begins and keeps it as a class. "
      "Comparing the two on the same annotations:"])
table(["", "commit f30a14d", "other/ (this round)"],
      [["Lift frames", "6.8%", "0.0%"], ["Idle frames", "15.2%", "18.1%"],
       ["Real lifting frames labelled idle", "—", "41.6%"]],
      [2.8, 1.6, 1.8])
body("In the data this round trained on, a large share of picking up and raising the panel "
     "is labelled “nobody is working” — immediately before Place and Align, "
     "which is when the robot most needs to anticipate. The feature-set comparison is still "
     "valid, because all arms shared the same labels. The idle-related numbers are not "
     "reliable. The corpus has been relabelled with f30a14d (data_f30a14d/, features "
     "unchanged) and all further runs use it, starting with a new baseline, B0.")

h1("Appendix A: incidents during this round")
for parts in [
    [("Memory exhaustion. ", {"b": 1}),
     "The first N1 run started on the old data path, which materialised every overlapping "
     "window (28.4 GB requested on a 31.4 GB machine). It slowed from ~5 min to ~3–4 h "
     "per fold at fold 11. Windows are now gathered per batch (1.8 GB, bit-identical output) "
     "and N1 was rerun."],
    [("Crash after training. ", {"b": 1}),
     "N1 finished all 15 folds, then crashed printing its summary (a stale class name). The "
     "results were rebuilt from the 15 saved fold checkpoints and match the training log "
     "exactly (0.499 ± 0.089)."],
    [("Scoring artefact. ", {"b": 1}), "Described in Section 4; corrected before reporting."],
]:
    rich(parts, bullet=True)

h1("Appendix B: files")
table(["Path", "Contents"],
      [["bench/results/loso_N*.json", "per-fold metrics and run configuration"],
       ["bench/results/folds/N*_uid*.pth", "trained weights, one per held-out subject"],
       ["bench/eval_round1/<arm>/", "metrics.csv, confusion_matrix.csv, per_class.csv"],
       ["bench/results/fair_idle_round1.json", "seven-way comparison, per fold"],
       ["bench/results/feat_importance_*.json", "panel and joint importance, per fold"],
       ["archive_round1_colleague/", "all of the above plus source code and MANIFEST.json"]],
      [2.9, 3.5], mono=False)

doc.save("Round1_Report.docx")
print("saved Round1_Report.docx")
