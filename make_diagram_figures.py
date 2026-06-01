r"""
Generates three publication-quality block-diagram figures for the manuscript:

  manuscript/figs/fig_pipeline_flowchart.{pdf,svg}   -- end-to-end diagnosis pipeline
  manuscript/figs/fig_ml_dl_framework.{pdf,svg}      -- ML cascade + DL alternative
  manuscript/figs/fig_jetson_sweep.{pdf,svg}         -- measured Jetson 6-case sweep

Run: python make_diagram_figures.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manuscript" / "figs"
OUT.mkdir(parents=True, exist_ok=True)


# ============ helper drawing primitives ============
def box(ax, x, y, w, h, text, fc="white", ec="black", fontsize=9,
        rounding=0.06, bold=False, fontweight=None):
    fw = "bold" if bold else (fontweight or "normal")
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                           linewidth=1.0, facecolor=fc, edgecolor=ec)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, wrap=True)
    return (x, y, x + w, y + h)


def arrow(ax, p0, p1, text=None, color="black", style="-|>",
          rad=0.0, lw=1.0, fontsize=8):
    ar = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10,
                         color=color, linewidth=lw,
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(ar)
    if text:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2 + 0.08
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=fontsize,
                style="italic", color=color)


# ============ FIGURE 1: PIPELINE FLOWCHART ============
def fig_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13)
    ax.set_aspect("equal"); ax.axis("off")

    # banner
    box(ax, 0.5, 12.1, 9.0, 0.7,
        "End-to-End Diagnosis Pipeline: Three-Phase Stator Current $\\rightarrow$ Verdict",
        fc="#e6e6e6", bold=True, fontsize=10)

    # Step S1 -- Acquisition
    box(ax, 0.6, 10.7, 8.8, 1.1,
        "S1.  ACQUISITION & INDEXING\n"
        "Three-phase current $i_a, i_b, i_c$ at $f_s$  •  operating-point label  •  region bounds via RMS envelope",
        fc="#dbeafe", fontsize=9)

    # Step S2 -- Windowing
    box(ax, 0.6, 9.3, 8.8, 1.1,
        "S2.  WINDOWING + LABELLING\n"
        "$W=2000$ samples (4 cycles, 80 ms), 75% overlap; pre-fault region $\\rightarrow$ matched-load HEALTHY",
        fc="#dbeafe", fontsize=9)

    # Step S3 -- Normalisation
    box(ax, 0.6, 7.9, 8.8, 1.1,
        "S3.  PER-RECORDING NORMALIZATION\n"
        "Divide all 3 phases by single RMS scale (preserves inter-phase imbalance, removes load amplitude)",
        fc="#dbeafe", fontsize=9)

    # Step S4-S6 -- Feature extraction (group)
    box(ax, 0.6, 4.7, 8.8, 3.0,
        "", fc="#fff7e6", fontsize=9)
    ax.text(5, 7.4, "S4–S6.  FEATURE EXTRACTION  (96 features per window)",
            ha="center", fontweight="bold", fontsize=9.5)
    box(ax, 0.9, 6.0, 2.8, 1.2,
        "Symmetrical comp.\n$|I_0|, |I_1|, |I_2|$,\n"
        "$|I_2|/|I_1|$,  $\\angle I_2$",
        fc="#fffbe0", fontsize=8)
    box(ax, 3.9, 6.0, 2.8, 1.2,
        "Physics + spectral\nEPVA SF,  Park ellipse,\nTHD, 3rd harmonic",
        fc="#fffbe0", fontsize=8)
    box(ax, 6.9, 6.0, 2.4, 1.2,
        "Per-phase stats\nRMS / skew / kurt;\n"
        "inter-phase diffs",
        fc="#fffbe0", fontsize=8)
    box(ax, 0.9, 4.85, 4.0, 1.0,
        "Load-invariant $n_*$ magnitudes\n(magnitude $\\div$ within-window $|I_1|$)",
        fc="#fffbe0", fontsize=8)
    box(ax, 5.1, 4.85, 4.2, 1.0,
        "Biomedical (24): HRV cycle-to-cycle,\nHjorth, fractal dim., entropies",
        fc="#fffbe0", fontsize=8)

    # Stage 0
    box(ax, 0.6, 3.3, 8.8, 1.0,
        "STAGE 0.  ONSET TRIGGER   $\\rho(k) = |I_2(k)|/|I_1(k)| > \\overline{\\rho} + 6\\sigma$  for 3 cycles",
        fc="#dcfce7", fontsize=9, bold=True)

    # Decision gate Stage 1
    box(ax, 1.6, 1.7, 6.8, 1.2,
        "STAGE 1.  DETECTION  (XGBoost, full 96 features)\n"
        "Healthy vs Faulty  •  calibrated probability  •  SHAP audit passes",
        fc="#fee2e2", fontsize=9, bold=True)

    # If Faulty, two parallel branches
    box(ax, 0.6, 0.1, 4.2, 1.3,
        "STAGE 2.  SEVERITY  (ordinal)\n"
        "Load-invariant feats $+$ measured load_pct\n"
        "$\\rightarrow$ $r \\in \\{0..6\\}$  ($\\{0.3..5\\}\\%$ shorted turns)",
        fc="#fce7f3", fontsize=9)
    box(ax, 5.2, 0.1, 4.2, 1.3,
        "STAGE 3.  FAULTED PHASE  (3-class)\n"
        "$\\sin(\\angle I_2)$, $\\cos(\\angle I_2)$ dominant feats\n"
        "$\\rightarrow$ Phase $\\in \\{A, B, C\\}$",
        fc="#fce7f3", fontsize=9)

    # arrows top-to-bottom
    for y0, y1 in [(10.7, 10.4), (9.3, 9.0), (7.9, 7.6), (4.7, 4.3),
                   (3.3, 2.9)]:
        arrow(ax, (5, y0), (5, y1))
    arrow(ax, (5, 1.7), (2.7, 1.4), text="if FAULTY")
    arrow(ax, (5, 1.7), (7.3, 1.4))
    arrow(ax, (5, 1.7), (5, -0.3), text="if HEALTHY $\\rightarrow$ STOP",
          rad=0.0, color="gray")
    # frame
    fig.suptitle("Stator Inter-Turn Fault Diagnosis Pipeline (Track A, deployed)", y=0.985,
                 fontsize=11, fontweight="bold")
    out_pdf = OUT / "fig_pipeline_flowchart.pdf"
    out_svg = OUT / "fig_pipeline_flowchart.svg"
    fig.savefig(out_pdf); fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ============ FIGURE 2: ML / DL FRAMEWORK ============
def fig_ml_dl_framework():
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5)
    ax.set_aspect("equal"); ax.axis("off")

    # banner
    box(ax, 0.3, 7.7, 9.4, 0.65,
        "Two Diagnosis Tracks: Physics-Feature Cascade  (Track A, deployed)  vs.  PCM-Net  (Track B, deep)",
        fc="#e6e6e6", bold=True, fontsize=9.5)

    # Common input
    box(ax, 3.5, 6.6, 3.0, 0.7,
        "INPUT: $(i_a, i_b, i_c)$ window  $(3, W)$",
        fc="#dbeafe", fontsize=9, bold=True)

    # ============ Left column: TRACK A (XGBoost cascade, deployed) ============
    # title
    ax.text(2.4, 6.0, "TRACK A — Physics features + XGBoost cascade",
            ha="center", fontweight="bold", fontsize=9.5, color="#1f4e79")
    # feat extract
    box(ax, 0.3, 4.9, 4.2, 0.85,
        "Feature extractor (NumPy/SciPy)\n96 features: physics + statistical + biomedical",
        fc="#fff7e6", fontsize=8.5)

    # cascade boxes
    box(ax, 0.4, 3.8, 4.0, 0.6, "Stage 0 — Onset trigger  ($|I_2|/|I_1|$ CUSUM)",
        fc="#dcfce7", fontsize=8.5)
    box(ax, 0.4, 2.95, 4.0, 0.7, "Stage 1 — Detection  (XGBoost binary)",
        fc="#fee2e2", fontsize=9, bold=True)
    box(ax, 0.4, 2.05, 1.85, 0.7, "Stage 2 — Severity\n(XGB ordinal)",
        fc="#fce7f3", fontsize=8.5)
    box(ax, 2.55, 2.05, 1.85, 0.7, "Stage 3 — Phase\n(XGB 3-class)",
        fc="#fce7f3", fontsize=8.5)
    box(ax, 0.4, 1.0, 4.0, 0.7,
        "Per-recording aggregation\n(median sev, majority-vote phase)",
        fc="#e5e7eb", fontsize=8.5)
    box(ax, 0.4, 0.05, 4.0, 0.7,
        "Verdict:  $\\{$state, severity \\%, faulted phase$\\}$",
        fc="#dbeafe", fontsize=9, bold=True)

    # arrows TRACK A
    for y0, y1 in [(4.9, 4.4), (3.8, 3.65), (2.95, 2.75), (1.0, 0.75)]:
        arrow(ax, (2.4, y0), (2.4, y1))
    # detection -> sev / phase
    arrow(ax, (2.4, 2.95), (1.32, 2.75), text="if F")
    arrow(ax, (2.4, 2.95), (3.48, 2.75))
    arrow(ax, (1.32, 2.05), (2.4, 1.7))
    arrow(ax, (3.48, 2.05), (2.4, 1.7))

    # ============ Right column: TRACK B (PCM-Net) ============
    ax.text(7.6, 6.0, "TRACK B — PCM-Net deep model  (alternative)",
            ha="center", fontweight="bold", fontsize=9.5, color="#7c3aed")
    # stem
    box(ax, 5.5, 4.9, 4.2, 0.85,
        "Multi-scale 1-D conv stem  (3 phase tokens)",
        fc="#ede9fe", fontsize=8.5)
    # phase coupling
    box(ax, 5.5, 3.95, 4.2, 0.75,
        "Phase-coupling module (graph attn over $\\{a, b, c\\}$)",
        fc="#ede9fe", fontsize=8.5)
    # backbone
    box(ax, 5.5, 3.05, 4.2, 0.75,
        "Dilated-temporal backbone  (Mamba surrogate)",
        fc="#ede9fe", fontsize=8.5)
    # FiLM
    box(ax, 5.5, 2.10, 4.2, 0.75,
        "FiLM load conditioning $\\;\\;\\gamma, \\beta = f(\\text{load\\_pct})$",
        fc="#fef9c3", fontsize=8.5, bold=True)
    # heads
    box(ax, 5.5, 1.10, 1.3, 0.75, "Detect\nhead", fc="#fee2e2", fontsize=8)
    box(ax, 6.95, 1.10, 1.3, 0.75, "Severity\n(KAN)", fc="#fce7f3", fontsize=8)
    box(ax, 8.4, 1.10, 1.3, 0.75, "Phase\nhead", fc="#fce7f3", fontsize=8)
    # PINN reg
    box(ax, 5.5, 0.05, 4.2, 0.75,
        "PINN regularizer  $\\|i_a + i_b + i_c\\|^2$ + neg-seq vs sev",
        fc="#fef9c3", fontsize=8)
    # arrows TRACK B
    arrow(ax, (6.5, 6.6), (7.6, 5.75))      # input to stem
    for y0, y1 in [(4.9, 4.7), (3.95, 3.8), (3.05, 2.85), (2.10, 1.85)]:
        arrow(ax, (7.6, y0), (7.6, y1))

    # input feed arrows
    arrow(ax, (4.0, 6.6), (2.4, 5.75))   # input to feature extractor

    out_pdf = OUT / "fig_ml_dl_framework.pdf"
    out_svg = OUT / "fig_ml_dl_framework.svg"
    fig.savefig(out_pdf); fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ============ FIGURE 3: JETSON 6-CASE SWEEP ============
def fig_jetson_sweep():
    # Cases run + verdicts (from receiver log).
    cases = ["HLT", "B30", "A30", "C30", "A40", "B10"]
    truth_state = ["Healthy", "Faulty", "Faulty", "Faulty", "Faulty", "Faulty"]
    truth_sev   = [0, 30, 30, 30, 40, 10]
    truth_phase = ["—", "B", "A", "C", "A", "B"]
    pred_state  = ["FAULTY (FP)", "FAULTY", "FAULTY", "FAULTY", "FAULTY", "FAULTY"]
    pred_sev    = [40, 30, 40, 40, 40, 40]
    pred_phase  = ["A", "B", "A", "C", "A", "B"]

    detect_ok = [s == ("Faulty" if "FAULTY" in p and "FP" not in p else "Healthy")
                 for s, p in zip(truth_state, pred_state)]
    detect_ok = [True if (t == "Healthy" and "FP" not in p) or
                          (t == "Faulty" and "FP" not in p)
                 else False
                 for t, p in zip(truth_state, pred_state)]
    sev_within1 = [abs(t - p) <= 10 if t != 0 else (p == 0)
                   for t, p in zip(truth_sev, pred_sev)]
    sev_exact = [t == p for t, p in zip(truth_sev, pred_sev)]
    phase_ok = [(t == p) or (t == "—" and "FP" in pred_state[i])
                for i, (t, p) in enumerate(zip(truth_phase, pred_phase))]
    phase_ok[0] = False  # the FP doesn't get a phase credit

    fig = plt.figure(figsize=(7.3, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], width_ratios=[1.0, 1.0],
                          hspace=0.55, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])

    # Left: per-case grid (truth vs prediction)
    n = len(cases)
    x = np.arange(n)
    bar_h = 0.35
    truth_y = np.zeros(n) + 1
    pred_y = np.zeros(n)

    # Cell background by correctness
    for i in range(n):
        det_color = "#86efac" if detect_ok[i] else "#fca5a5"
        sev_color = "#86efac" if sev_exact[i] else ("#fde68a" if sev_within1[i] else "#fca5a5")
        ph_color = "#86efac" if phase_ok[i] else "#fca5a5"
        ax.add_patch(Rectangle((i - 0.45, 0.2), 0.9, 0.35, facecolor=det_color, edgecolor="black", lw=0.6))
        ax.add_patch(Rectangle((i - 0.45, 0.55), 0.9, 0.35, facecolor=sev_color, edgecolor="black", lw=0.6))
        ax.add_patch(Rectangle((i - 0.45, 0.9), 0.9, 0.35, facecolor=ph_color, edgecolor="black", lw=0.6))
        # text labels
        det_lbl = ("FAULTY" if "FAULTY" in pred_state[i] else "HEALTHY")
        if "FP" in pred_state[i]: det_lbl += "\n(FP)"
        ax.text(i, 0.375, det_lbl, ha="center", va="center", fontsize=7.5)
        ax.text(i, 0.725, f"{pred_sev[i]}%", ha="center", va="center", fontsize=8)
        ax.text(i, 1.075, pred_phase[i], ha="center", va="center", fontsize=8, fontweight="bold")
        # truth labels above
        ax.text(i, 1.55, f"{truth_state[i]}\nsev={truth_sev[i]}\nphase={truth_phase[i]}",
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1f2937")

    ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=9)
    ax.set_yticks([1.55, 1.075, 0.725, 0.375])
    ax.set_yticklabels(["Truth", "Pred\nphase", "Pred\nsev", "Pred\nstate"], fontsize=8)
    ax.set_ylim(0.05, 1.95)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_title("Per-recording verdicts from the Jetson Orin Nano\n(green = correct, yellow = within 1 severity level, red = wrong)",
                 fontsize=9.5, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)

    # bottom-left: aggregate bar
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ["Detection\n5/6", "Phase\n5/5 faulted", "Severity (within-1)\n4/5"]
    vals = [5/6, 5/5, 4/5]
    ax2.barh(metrics, vals, color=["#1f4e79", "#1f4e79", "#1f4e79"])
    for i, v in enumerate(vals):
        ax2.text(v - 0.02, i, f"{v:.2f}", ha="right", va="center", color="white",
                 fontsize=9, fontweight="bold")
    ax2.set_xlim(0, 1.05); ax2.set_xlabel("Accuracy", fontsize=8.5)
    ax2.set_title("Aggregate per-recording", fontsize=9, pad=4)
    ax2.tick_params(axis="both", labelsize=8)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    # bottom-right: latency bars
    ax3 = fig.add_subplot(gs[1, 1])
    modes = ["CPU\n(sklearn)", "GPU\n(needs build)", "Accelerated CPU\n(Booster API)"]
    totals = [26.2, np.nan, 23.7]
    colors = ["#94a3b8", "#fca5a5", "#1f4e79"]
    bars = ax3.bar(modes, totals, color=colors, edgecolor="black", linewidth=0.6)
    ax3.axhline(500, color="gray", linestyle=":", lw=1)
    ax3.text(2.4, 460, "window-fill (500 ms)", ha="right", fontsize=7.5,
             color="gray", style="italic")
    for b, v in zip(bars, totals):
        if not np.isnan(v):
            ax3.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}", ha="center",
                     va="bottom", fontsize=8.5, fontweight="bold")
        else:
            ax3.text(b.get_x() + b.get_width()/2, 2, "n/a", ha="center",
                     va="bottom", fontsize=8.5, color="#7f1d1d")
    ax3.set_yscale("log")
    ax3.set_ylim(1, 700)
    ax3.set_ylabel("ms per window", fontsize=8.5)
    ax3.set_title("Per-window latency on Orin Nano", fontsize=9, pad=4)
    ax3.tick_params(axis="x", labelsize=8)
    ax3.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"): ax3.spines[s].set_visible(False)

    fig.suptitle("Real-Time Edge Validation on a Jetson Orin Nano (Super) — 5 kHz UART, 0 bytes dropped",
                 fontsize=10.5, fontweight="bold", y=1.02)

    out_pdf = OUT / "fig_jetson_sweep.pdf"
    out_svg = OUT / "fig_jetson_sweep.svg"
    fig.savefig(out_pdf); fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    fig_pipeline_flowchart()
    fig_ml_dl_framework()
    fig_jetson_sweep()
    print("done.")
