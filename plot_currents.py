"""
Plot three-phase stator currents (cols 2,3,4 of a *_02.out file) for an
induction-motor case: Ia_ABCIM_s phases 1/2/3.

  col 1 = time (s, 40 us step, 25 kHz)
  col 2 = Ia_ABCIM_s:1  (phase A)
  col 3 = Ia_ABCIM_s:2  (phase B)
  col 4 = Ia_ABCIM_s:3  (phase C)

Usage:
  python plot_currents.py                 # default healthy vs 5% TT fault demo
  python plot_currents.py path_to_02.out  # plot a single case
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

FS = 25_000.0  # sampling frequency [Hz]


def load_currents(out02_path):
    """Return (t, ia, ib, ic) from a *_02.out file. Skips the blank header
    line automatically (np.loadtxt ignores blank lines)."""
    data = np.loadtxt(out02_path)
    t = data[:, 0]
    ia, ib, ic = data[:, 1], data[:, 2], data[:, 3]
    return t, ia, ib, ic


def plot_compare(healthy_path, faulty_path, healthy_label, faulty_label,
                 out_png="currents_healthy_vs_faulty.png"):
    th, ha, hb, hc = load_currents(healthy_path)
    tf, fa, fb, fc = load_currents(faulty_path)

    phases = ["Phase A  (Ia_ABCIM_s:1)",
              "Phase B  (Ia_ABCIM_s:2)",
              "Phase C  (Ia_ABCIM_s:3)"]
    healthy = [ha, hb, hc]
    faulty = [fa, fb, fc]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, name, h, f in zip(axes, phases, healthy, faulty):
        ax.plot(th, h, lw=0.4, color="tab:green", label=healthy_label)
        ax.plot(tf, f, lw=0.4, color="tab:red", alpha=0.8, label=faulty_label)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Three-phase stator current: healthy vs turn-to-turn fault",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


def plot_single(out02_path, out_png="currents_single.png"):
    t, ia, ib, ic = load_currents(out02_path)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, ia, lw=0.4, label="Phase A")
    ax.plot(t, ib, lw=0.4, label="Phase B")
    ax.plot(t, ic, lw=0.4, label="Phase C")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stator current (Ia_ABCIM_s)")
    ax.set_title(out02_path)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        plot_single(sys.argv[1])
    else:
        base = r"D:\Naveen"
        healthy = base + r"\Healthy case data\5% Loading\Normal5Plaoding_02.out"
        faulty = (base + r"\TT fault Data\5%\Phase_a_5%\5%_20%"
                         r"\TTfault_25k_phase_a_5%_loading_20%_02.out")
        plot_compare(healthy, faulty,
                     healthy_label="Healthy (5% load)",
                     faulty_label="TT fault 5%, phase A (20% load)")
