"""Annotated phase-A current for the TT-fault case, marking motor start,
load change and fault inception/clearing."""
import numpy as np
import matplotlib.pyplot as plt

FS, N = 25000.0, 500
F = (r"D:\Naveen\TT fault Data\5%\Phase_a_5%\5%_20%"
     r"\TTfault_25k_phase_a_5%_loading_20%_02.out")

d = np.loadtxt(F)
t, ia = d[:, 0], d[:, 1]
rms = np.sqrt((ia[:(len(ia)//N)*N].reshape(-1, N)**2).mean(1))
tc = (np.arange(len(rms))+0.5)*N/FS

events = {"Motor start ~2.0 s": 2.0,
          "Load change ~4.0 s": 4.0,
          "TT fault inception ~7.19 s": 7.19,
          "Fault cleared ~7.99 s": 7.99}
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax1.plot(t, ia, lw=0.3, color="0.3")
ax1.set_ylabel("Phase-A current (inst.)")
ax1.set_title("TT fault 5%, phase A, 20% load  -  instantaneous current")
ax2.plot(tc, rms, lw=1.2, color="tab:purple")
ax2.set_ylabel("Phase-A current (per-cycle RMS)")
ax2.set_xlabel("Time (s)")
for ax in (ax1, ax2):
    for (lbl, x), c in zip(events.items(), colors):
        ax.axvline(x, color=c, ls="--", lw=1.2, label=lbl)
    ax.grid(True, alpha=0.3)
ax2.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("events_annotated.png", dpi=130)
print("saved events_annotated.png")
