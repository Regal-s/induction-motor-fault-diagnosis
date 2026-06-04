r"""
Assemble the model comparison from dataset/benchmark_results.json into:
  - dataset/benchmark_comparison.md   (markdown tables, per task x protocol)
  - figures_svg/fig_benchmark.svg/.pdf (grouped-bar comparison, new models highlighted)

New models (TabPFN-2.5, xLSTM) are marked with *.  Previously-reported deep baselines
(ResNet-1D, PCM-Net) ran on partial / single-split protocols and are listed separately
(NOT identical splits) so the main table stays apples-to-apples.

Run:  python benchmark_report.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"svg.fonttype": "none", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 120})

ROOT = Path(r"D:\Naveen"); DS = ROOT / "dataset"; OUT = ROOT / "figures_svg"
NEW = {"TabPFN-2.5", "xLSTM", "ESN", "NG-RC", "ROCKET",
       "PatchTST", "iTransformer", "TimesNet", "TTM", "Chronos-Bolt"}
PRIMARY = {"detection": "macro_f1", "severity": "within1", "phase": "macro_f1"}
PRIMARY_LABEL = {"detection": "macro-F1", "severity": "within-1", "phase": "macro-F1"}
ORDER = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest", "XGBoost", "TabPFN-2.5",
         "xLSTM", "ESN", "NG-RC", "ROCKET",
         "PatchTST", "iTransformer", "TimesNet", "TTM", "Chronos-Bolt"]


def load():
    return json.load(open(DS / "benchmark_results.json"))["results"]


def md_table(res, task):
    cols = (["acc", "macro_f1", "within1", "mae", "qwk"] if task == "severity"
            else ["acc", "macro_f1"])
    head = "| Model | Protocol | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    rows = [head, sep]
    for m in [x for x in ORDER if x in res]:
        for proto in ("groupkfold", "lolo"):
            d = res[m].get(task, {}).get(proto)
            if not d or "error" in d:
                continue
            tag = f"**{m}***" if m in NEW else m
            vals = " | ".join(f"{d.get(c, float('nan')):.3f}" for c in cols)
            rows.append(f"| {tag} | {proto} | {vals} |")
    return "\n".join(rows)


def main():
    res = load()
    models = [m for m in ORDER if m in res]
    print("models:", models)

    # ---------------- markdown ----------------
    lines = ["# Model benchmark — TabPFN-2.5 & xLSTM vs prior models",
             "",
             "Identical leakage-safe splits (StratifiedGroupKFold + Leave-One-Load-Out) on the "
             "72 physics features (feature models) / decimated 3-phase windows (xLSTM). "
             "Severity = 7-level ordinal classification. New models marked *.",
             ""]
    for task in ("detection", "severity", "phase"):
        lines += [f"## {task.capitalize()}", "", md_table(res, task), ""]

    # previously-reported deep baselines (partial protocols) ---------------------
    extra = []
    try:
        dl = json.load(open(DS / "dl_metrics.json"))
        extra.append(("ResNet-1D", "detection/GK", f"macro-F1 {dl['stage1_grouped']['macro_f1']:.3f}"))
        extra.append(("ResNet-1D", "severity/GK", f"within-1 {dl['stage2_grouped']['within1']:.3f}"))
        extra.append(("ResNet-1D", "phase/GK", f"macro-F1 {dl['stage3_grouped']['macro_f1']:.3f}"))
        extra.append(("ResNet-1D", "detection/LOLO(load20 only)", f"macro-F1 {dl['stage1_lolo20']['macro_f1']:.3f}"))
    except Exception:
        pass
    try:
        pc = json.load(open(DS / "pcmnet_results.json"))
        extra.append(("PCM-Net", "detection/GK", f"macro-F1 {pc['group_film=True']['detect_f1']:.3f}"))
        extra.append(("PCM-Net", "phase/GK", f"macro-F1 {pc['group_film=False']['phase_f1']:.3f}"))
        extra.append(("PCM-Net", "severity/LOLO", f"within-1 {pc['lolo_severity_within1_film=True']:.3f}"))
    except Exception:
        pass
    if extra:
        lines += ["## Previously-reported deep baselines (partial / single-split protocols — NOT identical)",
                  "", "| Model | Task/Protocol | Metric |", "|---|---|---|"]
        lines += [f"| {a} | {b} | {c} |" for a, b, c in extra]
        lines += [""]
    (DS / "benchmark_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote dataset/benchmark_comparison.md")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, task in zip(axes, ("detection", "severity", "phase")):
        key = PRIMARY[task]
        gk = [res[m].get(task, {}).get("groupkfold", {}).get(key, np.nan) for m in models]
        lo = [res[m].get(task, {}).get("lolo", {}).get(key, np.nan) for m in models]
        x = np.arange(len(models)); w = 0.4
        b1 = ax.bar(x - w/2, gk, w, label="GroupKFold", color="#4C78A8")
        b2 = ax.bar(x + w/2, lo, w, label="LOLO", color="#F58518")
        # highlight new models with a hatch
        for i, m in enumerate(models):
            if m in NEW:
                b1[i].set_hatch("//"); b2[i].set_hatch("//")
                b1[i].set_edgecolor("k"); b2[i].set_edgecolor("k")
        ax.set_xticks(x, [m + ("*" if m in NEW else "") for m in models], rotation=45, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title(f"{task.capitalize()} ({PRIMARY_LABEL[task]})")
        ax.grid(axis="y", alpha=0.3)
        if task == "detection":
            ax.set_ylabel("score"); ax.legend(loc="lower left")
    fig.suptitle("Model benchmark (identical splits; * = newly added)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_benchmark.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / "fig_benchmark.png", dpi=200, bbox_inches="tight")  # raster for Word
    (ROOT / "manuscript" / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "manuscript" / "figs" / "fig_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures_svg/fig_benchmark.svg (+ png + pdf)")


if __name__ == "__main__":
    main()
