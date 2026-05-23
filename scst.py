r"""
M2 — Sequence-Component Stockwell Tensor (SCST), a novel input representation.

For each 3-phase current window we (1) form the instantaneous symmetrical-component streams
(positive/negative/zero) via the analytic (Hilbert) signal and the Fortescue transform, and
(2) apply a band-limited Stockwell transform to each stream, stacking the three |S| maps into a
3-channel time-frequency tensor. The negative-sequence channel carries the load-invariant
inter-turn signature, and the S-transform's frequency-dependent resolution tracks the 2f0 EPVA
line and sidebands. A raw-phase Stockwell tensor (S-transform of i_a,i_b,i_c) is generated as a
control for the ablation.

Outputs (in dataset/scst/):
  scst_X.npy        (N,3,F,T) float32   sequence-component Stockwell tensors
  rawst_X.npy       (N,3,F,T) float32   control: raw-phase Stockwell tensors
  scst_labels.csv   N rows (subset of the sim windows, stratified, with labels)
  scst_examples.png example tensors (healthy vs severities)

Run:  python scst.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Naveen")
DS = ROOT / "dataset"
OUT = DS / "scst"
A = np.exp(2j * np.pi / 3)
NF, NT, KBAND = 32, 48, 6          # freq rows, time cols, band = KBAND * f0
SUBSET_PER_GROUP = 16              # windows sampled per operating point (keep compute bounded)


def inst_seq_components(win):
    """win (3,W) real -> (i0,i1,i2) complex (W,) instantaneous symmetrical components."""
    z = hilbert(win, axis=1)                      # analytic per phase (3,W) complex
    za, zb, zc = z[0], z[1], z[2]
    i0 = (za + zb + zc) / 3.0
    i1 = (za + A * zb + A**2 * zc) / 3.0
    i2 = (za + A**2 * zb + A * zc) / 3.0
    return i0, i1, i2


def stockwell_band(x, fs, f0, nf=NF, nt=NT, kband=KBAND):
    """Band-limited Stockwell transform magnitude of complex signal x (W,) -> (nf, nt)."""
    N = len(x)
    X = np.fft.fft(x)
    df = fs / N
    max_idx = max(2, int(round(kband * f0 / df)))
    idxs = np.unique(np.linspace(1, min(max_idx, N // 2 - 1), nf).astype(int))
    m = np.arange(N)
    rows = np.empty((len(idxs), N))
    for r, n in enumerate(idxs):
        g = np.exp(-2.0 * np.pi**2 * (m**2) / float(n**2))   # frequency-domain Gaussian
        rows[r] = np.abs(np.fft.ifft(np.roll(X, -int(n)) * g))
    # pad to nf rows if some idx collapsed, then downsample time to nt
    if rows.shape[0] < nf:
        rows = np.vstack([rows, np.repeat(rows[-1:], nf - rows.shape[0], axis=0)])
    t_edges = np.linspace(0, N, nt + 1).astype(int)
    img = np.stack([rows[:, t_edges[k]:t_edges[k + 1]].mean(1) for k in range(nt)], axis=1)
    return img.astype(np.float32)


def build_scst(win, fs, f0):
    i0, i1, i2 = inst_seq_components(win)
    return np.stack([stockwell_band(i1, fs, f0),   # positive
                     stockwell_band(i2, fs, f0),   # negative (fault channel)
                     stockwell_band(i0, fs, f0)], 0)


def build_rawst(win, fs, f0):
    return np.stack([stockwell_band(win[p].astype(complex), fs, f0) for p in range(3)], 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(DS / "windows.npy", mmap_mode="r")
    lab = pd.read_csv(DS / "window_labels.csv", low_memory=False)
    fs, f0 = 25000.0, 50.0

    # stratified subset by operating point (group_id), balanced & bounded
    rng = np.random.RandomState(0)
    idx = []
    for g, sub in lab.groupby("group_id"):
        take = sub.sample(min(SUBSET_PER_GROUP, len(sub)), random_state=0).index.to_numpy()
        idx.append(take)
    idx = np.sort(np.concatenate(idx))
    sub = lab.loc[idx].reset_index(drop=True)
    print(f"SCST subset: {len(idx)} windows from {lab['group_id'].nunique()} groups")

    scst = np.empty((len(idx), 3, NF, NT), np.float32)
    raw = np.empty((len(idx), 3, NF, NT), np.float32)
    for j, i in enumerate(idx):
        w = np.asarray(X[i])
        scst[j] = build_scst(w, fs, f0)
        raw[j] = build_rawst(w, fs, f0)
        if (j + 1) % 200 == 0:
            print(f"  {j+1}/{len(idx)}")
    np.save(OUT / "scst_X.npy", scst)
    np.save(OUT / "rawst_X.npy", raw)
    sub.to_csv(OUT / "scst_labels.csv", index=False)
    print(f"saved {OUT/'scst_X.npy'} {scst.shape}, control {raw.shape}, labels {len(sub)}")

    # example figure: negative-sequence channel for healthy + a few severities (phase A)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    picks = [("healthy", sub[(sub.label == 0)]),
             ("TT 0.3% A", sub[(sub.severity_pct == 0.3) & (sub.phase == "A")]),
             ("TT 1% A", sub[(sub.severity_pct == 1.0) & (sub.phase == "A")]),
             ("TT 5% A", sub[(sub.severity_pct == 5.0) & (sub.phase == "A")])]
    for ax, (title, rows) in zip(axes, picks):
        if len(rows):
            ax.imshow(scst[rows.index[0], 1], aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title, fontsize=9); ax.set_xlabel("time"); ax.set_ylabel("freq")
    fig.suptitle("SCST negative-sequence channel (|S| of i2) — healthy vs severity")
    fig.tight_layout(); fig.savefig(OUT / "scst_examples.png", dpi=130)
    print(f"saved {OUT/'scst_examples.png'}")


if __name__ == "__main__":
    main()
