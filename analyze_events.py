"""Detect motor-start, load-change and fault-inception instants from the
per-cycle RMS envelope of the three-phase stator current (cols 2,3,4 of *_02.out)."""
import numpy as np

FS = 25_000.0
N_CYC = 500            # samples per 50 Hz cycle at 25 kHz


def load_currents(p):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def cycle_rms(x, n=N_CYC):
    m = (len(x) // n) * n
    r = x[:m].reshape(-1, n)
    return np.sqrt((r ** 2).mean(axis=1))          # one RMS value per cycle


def step_times(rms, thresh_frac=0.15):
    """Return cycle indices where |d RMS| jumps more than thresh_frac of the
    running steady level."""
    d = np.diff(rms)
    base = np.maximum(rms[:-1], 1e-6)
    rel = np.abs(d) / base
    return np.where(rel > thresh_frac)[0]


def analyse(path, label):
    t, ia, ib, ic = load_currents(path)
    dur = t[-1]
    print(f"\n=== {label} ===")
    print(f"  file length: {len(t)} samples  ->  {dur:.4f} s")
    ima = cycle_rms(ia)
    tc = (np.arange(len(ima)) + 0.5) * N_CYC / FS   # cycle centre times

    # 1) motor start = first cycle where phase-A RMS exceeds a small floor
    floor = 0.02
    started = np.where(ima > floor)[0]
    t_start = tc[started[0]] if len(started) else None

    # 2) significant steps in the envelope (after start), grouped
    idx = step_times(ima)
    idx = idx[tc[idx] > (t_start + 0.3 if t_start else 0)]   # ignore start transient
    # group consecutive cycle indices into events
    events = []
    for i in idx:
        if not events or i - events[-1][-1] > 5:
            events.append([i])
        else:
            events[-1].append(i)
    print(f"  motor start  : t ~= {t_start:.3f} s")
    for grp in events:
        c0 = grp[0]
        before = ima[max(c0 - 3, 0)]
        after = ima[min(grp[-1] + 3, len(ima) - 1)]
        print(f"  step event   : t ~= {tc[c0]:.3f} s   RMS {before:.3f} -> {after:.3f}")

    # steady levels in a few windows for context
    def win_rms(a, b):
        s, e = int(a * FS), int(b * FS)
        return np.sqrt((ia[s:e] ** 2).mean())
    print(f"  RMS windows  : 2.8-2.9s={win_rms(2.8,2.9):.3f}  "
          f"3.5-3.6s={win_rms(3.5,3.6):.3f}  "
          f"5.0-5.1s={win_rms(5.0,5.1):.3f}  "
          f"7.5-7.6s={win_rms(7.5,7.6):.3f}  "
          f"9.0-9.1s={win_rms(9.0,9.1):.3f}")


if __name__ == "__main__":
    base = r"D:\Naveen"
    analyse(base + r"\Healthy case data\5% Loading\Normal5Plaoding_02.out",
            "Healthy 5% load")
    analyse(base + r"\TT fault Data\5%\Phase_a_5%\5%_20%"
                   r"\TTfault_25k_phase_a_5%_loading_20%_02.out",
            "TT fault 5%, phase A, 20% load")
