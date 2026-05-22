r"""Make a results-summary figure for the README: per-stage + end-to-end performance,
in-distribution (GroupKFold) vs cross-load (Leave-One-Load-Out). -> dataset/results_summary.png"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = Path(r"D:\Naveen") / "dataset"
labels = ["Detection\n(macro-F1)", "Severity\n(within-1)", "Phase ID\n(macro-F1)", "End-to-end\n(within-1)"]
gk = [0.977, 0.997, 0.972, 0.949]
lolo = [0.948, 0.946, 0.997, 0.916]

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.6))
b1 = ax.bar(x - w / 2, gk, w, label="In-distribution (GroupKFold)", color="#4C78A8")
b2 = ax.bar(x + w / 2, lolo, w, label="Cross-load (Leave-One-Load-Out)", color="#F58518")
for b in (b1, b2):
    for r in b:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.003,
                f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, labels)
ax.set_ylim(0.80, 1.01)
ax.set_ylabel("score")
ax.set_title("Stator inter-turn fault diagnosis — performance by stage\n"
             "(load-aware severity; physics features + XGBoost)")
ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(DS / "results_summary.png", dpi=140)
print("saved", DS / "results_summary.png")
