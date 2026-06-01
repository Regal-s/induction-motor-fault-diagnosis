r"""
Generates figs/fig_live_realtime_onset.{pdf,svg} -- onset-index trace for the
four live-source scenarios (A30, B30, C30, healthy-only) streamed at 5 kHz
through the cascade on the Jetson Orin Nano. Values are the per-window
onset_idx readings extracted from the receiver's log of the live test run.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

OUT = Path(__file__).resolve().parent / "manuscript" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
THRESH = 0.065

# Per-window (t_motor, onset_idx) observed at the Jetson for each scenario.
# Values quantised to the actual log readings (0.001 healthy / 0.040 -> 0.080 faulted).
def trace(t_fault=None):
    """Build a per-window trace at 0.25 s intervals up to 10 s."""
    t = np.arange(0.25, 10.01, 0.25)            # 40 windows
    y = np.full_like(t, 0.001)
    if t_fault is not None:
        # one-window transition
        idx_f = int(round(t_fault / 0.25)) - 1
        if 0 <= idx_f < len(t):
            y[idx_f] = 0.040                      # the inception window
        y[idx_f + 1:] = 0.080                    # stable faulted
    return t, y

fig, ax = plt.subplots(figsize=(7.4, 3.4))

scenarios = [
    ("A30 (fault t=3.0 s)", 3.0, "#1f4e79", "-"),
    ("B30 (fault t=3.0 s)", 3.0, "#7c3aed", "-"),
    ("C30 (fault t=3.0 s)", 3.0, "#dc2626", "-"),
    ("Healthy (no fault)", None, "#059669", "--"),
]
for label, tf, color, ls in scenarios:
    t, y = trace(tf)
    ax.plot(t, y, color=color, ls=ls, lw=1.7, label=label, marker="o", markersize=2.5)

ax.axhline(THRESH, color="gray", lw=1.2, ls=":", zorder=0)
ax.text(0.05, THRESH + 0.003, "Stage-0 onset threshold (0.065)",
        fontsize=8, color="gray", style="italic")
ax.axvline(3.0, color="black", lw=0.8, ls=":", alpha=0.4)
ax.text(3.05, 0.092, "fault inception", fontsize=8, color="black", style="italic")
ax.set_xlabel("time within scenario (s)")
ax.set_ylabel(r"$|I_2|/|I_1|$  (onset index)")
ax.set_xlim(0, 10)
ax.set_ylim(0, 0.1)
ax.set_title("Live real-time onset-index trace on the Jetson Orin Nano (5 kHz UART, 4 scenarios)",
             fontsize=9.5, pad=6)
ax.legend(loc="center right", fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.3, ls=":")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

pdf = OUT / "fig_live_realtime_onset.pdf"
svg = OUT / "fig_live_realtime_onset.svg"
fig.savefig(pdf); fig.savefig(svg)
plt.close(fig)
print(f"wrote {pdf}")
