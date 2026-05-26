r"""
Update manuscript_ieee_final.docx to match main.tex: refresh the Comparative-Evaluation
opening paragraph and insert the 8-model benchmark table + fig_benchmark, matching the
LaTeX content. Surgical edit (keeps the rest of the carefully-formatted doc intact).
"""
import copy
from pathlib import Path
import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(r"D:\Naveen")
DOCX = ROOT / "manuscript_ieee_final.docx"
PNG = ROOT / "figures_svg" / "fig_benchmark.png"

NEW_PARA = (
    "Baseline models and added learners. The full model benchmark below evaluates eight models "
    "on identical leakage-safe splits, spanning classical feature-based classifiers, a tabular "
    "foundation model, and a sequence deep network. On detection the physics-feature classifiers "
    "cluster high under both protocols (macro-F1 0.95–0.98); the tabular foundation model "
    "TabPFN-2.5 is the strongest and the most load-robust, scoring macro-F1 0.979 in-distribution "
    "and 0.979 cross-load (LOLO), whereas gradient boosting falls from 0.976 to 0.948 across loads. "
    "For severity (within-one-level accuracy, the operational ordinal metric), TabPFN-2.5 is "
    "sharpest in-distribution (0.997, with the lowest mean absolute error of 0.033 level) and "
    "degrades gracefully across loads (0.969), while the MLP attains the best cross-load value "
    "(0.982); tree ensembles, by contrast, extrapolate poorly on absolute severity at unseen loads. "
    "Phase identification is load-robust for nearly all models (LOLO macro-F1 ≥ 0.96), reflecting "
    "the load-invariant negative-sequence-angle signature. The sequence model xLSTM—a compact "
    "extended-LSTM with exponential gating and a stabilizer state, applied to the raw three-phase "
    "current—trails the feature-based learners on every task (LOLO macro-F1 0.890 detection, "
    "within-one 0.908 severity, macro-F1 0.961 phase), yet it improves on the earlier raw-signal "
    "deep baselines (1-D ResNet and PCM-Net, 0.898 detection). This reaffirms the data-efficiency of "
    "physically engineered features over raw-waveform deep nets in this data regime. The feature "
    "pipeline is also lightweight: feature extraction plus XGBoost inference costs ≈0.34 ms per "
    "window on CPU."
)

TABLE_CAP = ("Full model benchmark on identical leakage-safe splits: StratifiedGroupKFold (GK, "
             "in-distribution) and Leave-One-Load-Out (LOLO, cross-load). Detection and phase report "
             "macro-F1; severity reports within-one-level accuracy. Best per column in bold. "
             "TabPFN-2.5 and xLSTM are added in this work.")
FIG_CAP = ("Model benchmark across the three diagnostic tasks under both protocols (GroupKFold vs. "
           "LOLO). Hatched bars denote the two models added in this work (TabPFN-2.5, xLSTM). "
           "Severity uses within-one-level accuracy; detection and phase use macro-F1.")

# columns: Det GK, Det LOLO, Sev GK, Sev LOLO, Phase GK, Phase LOLO
ROWS = [
    ("Logistic Regression", [0.978, 0.978, 0.997, 0.971, 0.970, 0.920]),
    ("SVM-RBF",             [0.978, 0.977, 0.991, 0.975, 0.984, 0.967]),
    ("k-NN",                [0.977, 0.977, 0.937, 0.942, 0.982, 0.981]),
    ("MLP",                 [0.978, 0.978, 0.997, 0.982, 0.956, 0.981]),
    ("Random Forest",       [0.978, 0.979, 0.985, 0.688, 0.965, 0.997]),
    ("XGBoost",             [0.976, 0.948, 0.990, 0.943, 0.976, 0.997]),
    ("TabPFN-2.5",          [0.979, 0.979, 0.997, 0.969, 0.968, 0.991]),
    ("xLSTM",               [0.957, 0.890, 0.982, 0.908, 0.921, 0.961]),
]
col_max = [max(r[1][j] for r in ROWS) for j in range(6)]  # best per column (bold)


def set_cell(cell, text, bold=False, size=8, center=True):
    cell.text = ""
    p = cell.paragraphs[0]
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold


def main():
    DOCX.with_suffix(".prebenchmark.bak.docx").write_bytes(DOCX.read_bytes())  # backup
    d = docx.Document(str(DOCX))

    # locate the Comparative-Evaluation opening paragraph
    anchor = None
    for p in d.paragraphs:
        if p.text.strip().startswith("Baseline models. On detection"):
            anchor = p
            break
    if anchor is None:
        raise SystemExit("could not find Comparative-Evaluation opening paragraph")

    # 1) refresh its text
    for r in list(anchor.runs):
        r.text = ""
    anchor.runs[0].text = NEW_PARA if anchor.runs else anchor.add_run(NEW_PARA)
    if not anchor.runs:
        anchor.add_run(NEW_PARA)

    styles = {s.name for s in d.styles}
    tcap_style = "Table Caption" if "Table Caption" in styles else None
    icap_style = "Image Caption" if "Image Caption" in styles else None
    cfig_style = "Captioned Figure" if "Captioned Figure" in styles else None

    cur = anchor._p  # XML anchor; insert each new element after `cur`, then advance

    # 2) table caption
    cap = d.add_paragraph(TABLE_CAP, style=tcap_style) if tcap_style else d.add_paragraph(TABLE_CAP)
    cur.addnext(cap._p); cur = cap._p

    # 3) the benchmark table (grouped header)
    tbl = d.add_table(rows=2 + len(ROWS), cols=7)
    try:
        tbl.style = "Table"
    except Exception:
        pass
    r0 = tbl.rows[0].cells
    r1 = tbl.rows[1].cells
    set_cell(r0[0], "")
    g1 = r0[1].merge(r0[2]); set_cell(g1, "Detection (macro-F1)", bold=True)
    g2 = r0[3].merge(r0[4]); set_cell(g2, "Severity (within-1)", bold=True)
    g3 = r0[5].merge(r0[6]); set_cell(g3, "Phase (macro-F1)", bold=True)
    for i, lab in enumerate(["Model", "GK", "LOLO", "GK", "LOLO", "GK", "LOLO"]):
        set_cell(r1[i], lab, bold=True, center=(i != 0))
    for ri, (name, vals) in enumerate(ROWS, start=2):
        cells = tbl.rows[ri].cells
        set_cell(cells[0], name, center=False)
        for j, v in enumerate(vals):
            set_cell(cells[j + 1], f"{v:.3f}", bold=(abs(v - col_max[j]) < 1e-9))
    cur.addnext(tbl._tbl); cur = tbl._tbl

    # 4) figure image
    figp = d.add_paragraph(style=cfig_style) if cfig_style else d.add_paragraph()
    figp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    figp.add_run().add_picture(str(PNG), width=Inches(3.3))
    cur.addnext(figp._p); cur = figp._p

    # 5) figure caption
    fcap = d.add_paragraph(FIG_CAP, style=icap_style) if icap_style else d.add_paragraph(FIG_CAP)
    cur.addnext(fcap._p); cur = fcap._p

    d.save(str(DOCX))
    # report
    d2 = docx.Document(str(DOCX))
    print(f"saved {DOCX.name}: paragraphs={len(d2.paragraphs)} tables={len(d2.tables)} "
          f"images={len(d2.inline_shapes)}")


if __name__ == "__main__":
    main()
