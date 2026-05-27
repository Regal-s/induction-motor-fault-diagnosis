r"""
Biomedical anomaly-detection features for 3-phase stator current (NOVEL feature engineering).

Imports signal-complexity / irregularity / variability descriptors from ECG/EEG/HRV anomaly
detection into stator inter-turn fault diagnosis (motor-fault literature uses physics/spectral
features only -> this is a genuine cross-domain gap). See papers/BIOMEDICAL_FEATURES_RESEARCH.md.

Design rule (critical): complexity is computed on the RESIDUAL (current minus its fitted
fundamental) and on the Park-vector modulus -- NOT raw current, whose clean fundamental is trivially
predictable. Scale-free members (fractal dimension, entropies, SD1/SD2 ratio) aid load-invariance.

Feature blocks (per window):
  HRV-on-current (the headline import): cycle-to-cycle waveform distance + Poincare SD1/SD2 on the
    per-cycle RMS series of the Park modulus (each fundamental cycle = a "heartbeat").
  Complexity scalars on Park modulus and on the mean phase residual: Hjorth mobility/complexity,
    permutation entropy, spectral entropy, Higuchi/Katz/Petrosian fractal dimension, dispersion
    entropy, sample entropy.

build_bio_frame(X, fs, f0) -> DataFrame aligned row-for-row with the input windows X (N,3,T).
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import antropy as ant
try:
    import EntropyHub as eh
    _HAS_EH = True
except Exception:
    _HAS_EH = False


def _fundamental_residual(x, fs, f0):
    """Subtract the single-frequency f0 component (projection onto sin/cos) -> residual."""
    n = len(x); t = np.arange(n) / fs
    c = np.cos(2 * np.pi * f0 * t); s = np.sin(2 * np.pi * f0 * t)
    # least-squares amplitude of cos/sin (orthogonal over integer cycles ~)
    a = 2.0 * (x * c).mean(); b = 2.0 * (x * s).mean()
    return x - (a * c + b * s)


def _park_modulus(win):
    ia, ib, ic = win
    al = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
    be = (1.0 / np.sqrt(3.0)) * (ib - ic)
    return np.sqrt(al * al + be * be)


def _cycle_split(sig, spc):
    """Split sig into ~n whole-fundamental-cycle segments (spc = fs/f0 samples per cycle, may be float)."""
    n = max(2, int(round(len(sig) / spc)))
    edges = np.linspace(0, len(sig), n + 1).astype(int)
    return [sig[edges[i]:edges[i + 1]] for i in range(n)]


def _hrv_block(sig, spc, prefix):
    """HRV-style cycle-to-cycle variability of a 1-D stream (the current 'heartbeat' analog)."""
    cyc = _cycle_split(sig, spc)
    rms = np.array([np.sqrt((c ** 2).mean() + 1e-12) for c in cyc])
    d = np.diff(rms)
    sd1 = float(np.std(d / np.sqrt(2))) if len(d) else 0.0
    sd2 = float(np.sqrt(max(2 * np.var(rms) - 0.5 * np.var(d), 0.0)))
    sdnn = float(rms.std()); cv = float(rms.std() / (rms.mean() + 1e-12))
    rmssd = float(np.sqrt((d ** 2).mean())) if len(d) else 0.0
    # cycle-to-cycle waveform distance to the median cycle (resampled to common length)
    L = max(4, min(len(c) for c in cyc))
    M = np.stack([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(c)), c) for c in cyc])
    med = np.median(M, axis=0)
    wd = np.sqrt(((M - med) ** 2).mean(axis=1))
    scale = np.sqrt((med ** 2).mean()) + 1e-12
    return {f"{prefix}_hrv_sd1": sd1, f"{prefix}_hrv_sd2": sd2,
            f"{prefix}_hrv_sd1sd2": float(sd1 / (sd2 + 1e-9)), f"{prefix}_hrv_sdnn_cv": cv,
            f"{prefix}_hrv_rmssd": rmssd,
            f"{prefix}_hrv_wavedist_mean": float(wd.mean() / scale),
            f"{prefix}_hrv_wavedist_std": float(wd.std() / scale)}


def _complexity_block(x, fs, prefix, kmax, with_spectral, with_sampen):
    """Complexity scalars for a 1-D signal."""
    x = np.asarray(x, dtype=float)
    out = {}
    try:
        mob, comp = ant.hjorth_params(x)
        out[f"{prefix}_hjorth_mob"] = float(mob); out[f"{prefix}_hjorth_comp"] = float(comp)
    except Exception:
        out[f"{prefix}_hjorth_mob"] = out[f"{prefix}_hjorth_comp"] = np.nan
    for nm, fn in [("perm_ent", lambda: ant.perm_entropy(x, order=4, normalize=True)),
                   ("higuchi", lambda: ant.higuchi_fd(x, kmax=kmax)),
                   ("katz", lambda: ant.katz_fd(x)),
                   ("petrosian", lambda: ant.petrosian_fd(x))]:
        try:
            out[f"{prefix}_{nm}"] = float(fn())
        except Exception:
            out[f"{prefix}_{nm}"] = np.nan
    if with_spectral:
        try:
            out[f"{prefix}_spec_ent"] = float(ant.spectral_entropy(x, sf=fs, method="welch", normalize=True))
        except Exception:
            out[f"{prefix}_spec_ent"] = np.nan
    if with_sampen:
        try:
            out[f"{prefix}_sampen"] = float(ant.sample_entropy(x, order=2))
        except Exception:
            out[f"{prefix}_sampen"] = np.nan
    if _HAS_EH:
        try:
            d, _ = eh.DispEn(x, m=3, tau=1, c=6)
            out[f"{prefix}_dispen"] = float(d)
        except Exception:
            out[f"{prefix}_dispen"] = np.nan
    return out


def window_bio_features(win, fs, f0):
    """All biomedical features for one (3, T) window."""
    spc = fs / f0
    kmax = int(min(10, max(3, win.shape[1] // 10)))
    pmod = _park_modulus(win)
    res = np.mean([_fundamental_residual(win[k], fs, f0) for k in range(3)], axis=0)  # mean phase residual
    feat = {}
    feat.update(_hrv_block(pmod, spc, "pmod"))
    feat.update(_complexity_block(pmod, fs, "pmod", kmax, with_spectral=True, with_sampen=True))
    feat.update(_complexity_block(res, fs, "res", kmax, with_spectral=False, with_sampen=True))
    return feat


def build_bio_frame(X, fs=25000.0, f0=50.0, verbose=False, n_jobs=-1):
    """Compute biomedical features for all windows X:(N,3,T). Returns a DataFrame (row-aligned)."""
    import pandas as pd
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    try:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, batch_size=64)(
            delayed(window_bio_features)(X[i], fs, f0) for i in range(N))
    except Exception:
        rows = [window_bio_features(X[i], fs, f0) for i in range(N)]
    df = pd.DataFrame(rows)
    if verbose:
        print(f"bio features: {df.shape[1]} cols x {N} windows")
    return df


if __name__ == "__main__":
    import sys, time, pandas as pd
    from pathlib import Path
    DS = Path(r"D:\Naveen") / "dataset"
    fs, f0 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (25000.0, 50.0)
    Xw = np.load(DS / "windows.npy", mmap_mode="r")
    t = time.time()
    fr = build_bio_frame(np.asarray(Xw), fs=fs, f0=f0, verbose=True)
    lab = pd.read_parquet(DS / "features.parquet")[["window_id"]] if "window_id" in pd.read_parquet(DS / "features.parquet").columns else None
    if lab is not None and len(lab) == len(fr):
        fr.insert(0, "window_id", lab["window_id"].to_numpy())
    fr.to_parquet(DS / "features_bio.parquet", index=False)
    print(f"saved dataset/features_bio.parquet {fr.shape} in {time.time()-t:.0f}s")
