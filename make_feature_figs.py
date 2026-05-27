r"""
Feature-importance and feature-reduction figures (editable SVG -> figures_svg/ + results_svg/,
PDF -> manuscript/figs/). Reads stored artifacts only (no heavy recompute):
  - stage1_metrics.json   : Stage-1 detection SHAP top-15 (mean|SHAP|)
  - ablate_bio_results.json: top biomedical features by gain (severity)
  - lir_selection.json    : LIR-mRMR vs mRMR feature reduction (within-1 + load-drift vs k)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 120})
ROOT = Path(r"D:\Naveen"); DS = ROOT / "dataset"
OUTS = [ROOT / "figures_svg", ROOT / "results_svg"]; [o.mkdir(exist_ok=True) for o in OUTS]
PDF = ROOT / "manuscript" / "figs"; PDF.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    for o in OUTS:
        fig.savefig(o / (name + ".svg"), format="svg", bbox_inches="tight")
    fig.savefig(PDF / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(ROOT / "results_svg" / (name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved", name)


def main():
    # ---------- feature importance (2 panels) ----------
    shap = json.load(open(DS / "stage1_metrics.json")).get("shap_top15", [])[:12][::-1]
    names = [s[0] for s in shap]; vals = [s[1] for s in shap]
    bio = json.load(open(DS / "ablate_bio_results.json"))["severity"]["top_bio_importance"][:8][::-1]
    bnames = [b[0] for b in bio]; bvals = [b[1] for b in bio]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].barh(names, vals, color="#4C78A8")
    ax[0].set_xlabel("mean |SHAP|"); ax[0].set_title("(a) Detection feature importance (SHAP, top-12)")
    ax[1].barh(bnames, bvals, color="#59A14F")
    ax[1].set_xlabel("XGBoost gain"); ax[1].set_title("(b) Top biomedical features (severity)")
    save(fig, "fig_feature_importance")

    # ---------- feature reduction (LIR-mRMR vs mRMR) ----------
    perk = json.load(open(DS / "lir_selection.json"))["per_k"]
    ks = sorted(int(k) for k in perk)
    mr = [perk[str(k)]["mrmr_within1"] for k in ks]; lr = [perk[str(k)]["lir_within1"] for k in ks]
    md = [perk[str(k)]["mrmr_meandrift"] for k in ks]; ld = [perk[str(k)]["lir_meandrift"] for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ks, mr, "o--", color="#888", label="mRMR")
    ax[0].plot(ks, lr, "o-", color="#E15759", label="LIR-mRMR (proposed)")
    ax[0].set_xlabel("number of selected features $k$"); ax[0].set_ylabel("severity within-1 (LOLO)")
    ax[0].set_title("(a) Cross-load accuracy vs feature-set size"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_xticks(ks)
    ax[1].plot(ks, md, "o--", color="#888", label="mRMR")
    ax[1].plot(ks, ld, "o-", color="#E15759", label="LIR-mRMR (proposed)")
    ax[1].set_xlabel("number of selected features $k$"); ax[1].set_ylabel("mean cross-load feature drift")
    ax[1].set_title("(b) Selected-feature load-drift (lower=better)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_xticks(ks)
    save(fig, "fig_feature_reduction")
    print("done")


if __name__ == "__main__":
    main()
