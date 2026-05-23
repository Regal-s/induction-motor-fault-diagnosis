r"""
Phase 1 / M3 — Engineered (Track-A) features per window, fully labelled.

Reads dataset/windows.npy (N,3,2000) + dataset/window_labels.csv, computes physics-
based + statistical features per window, then a 2nd pass for the per-case COMPENSATED
negative-sequence current (subtract each case's pre-fault healthy baseline I2).
Stores dataset/features.parquet (fallback .csv) with ALL label columns attached.

Window: 2000 samples @ 25 kHz, 50 Hz fundamental -> df=12.5 Hz, exact bins:
  50 Hz=bin4, 100=8, 150=12, 200=16, 250=20, 300=24, 350=28.

Feature groups:
  Tier1: EPVA severity factor; sequence comps |I1|,|I2|,|I0|, ratios, angles; (pass2) compensated I2.
  Tier2: 3rd-harmonic ratio, THD; PVA ellipse (eccentricity/axis-ratio/tilt).
  Tier3: per-phase RMS/peak/crest/std/skew/kurt + inter-phase imbalance; Park-modulus stats.

Run:  python features.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Naveen")
DS = ROOT / "dataset"
FS, F0, W = 25000.0, 50.0, 2000
B1, B2, B3, B5, B7 = 4, 8, 12, 20, 28   # FFT bins for 50/100/150/250/350 Hz
A = np.exp(2j * np.pi / 3)              # Fortescue operator
EPS = 1e-12
CHUNK = 3000


def chunk_features(X: np.ndarray, fs: float = FS, f0: float = F0) -> dict:
    """X: (n,3,W) normalised current windows -> dict of (n,) feature arrays
    (+ I2_real/I2_imag for the later compensation pass). FFT harmonic bins are derived
    from (fs, f0, W) so the SAME extractor serves the 25 kHz/50 Hz simulation and the
    1 kHz/60 Hz experimental data."""
    ia, ib, ic = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    n, W = X.shape[0], X.shape[2]
    bn = lambda h: int(round(h * f0 * W / fs))    # FFT bin of the h-th harmonic
    B1, B2, B3 = bn(1), bn(2), bn(3)
    nyq = W // 2

    # --- fundamental phasors (single-bin DFT at f0) & symmetrical components ---
    F = np.fft.rfft(X, axis=2)                    # (n,3,W/2+1)
    P = (2.0 / W) * F[:, :, B1]                   # complex (n,3) phasors
    Pa, Pb, Pc = P[:, 0], P[:, 1], P[:, 2]
    I1 = (Pa + A * Pb + A**2 * Pc) / 3.0
    I2 = (Pa + A**2 * Pb + A * Pc) / 3.0
    I0 = (Pa + Pb + Pc) / 3.0
    I1a, I2a, I0a = np.abs(I1), np.abs(I2), np.abs(I0)

    # --- EPVA: Park (Clarke) modulus and its 2f0 component ---
    alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
    beta = (ib - ic) / np.sqrt(3.0)
    mod = np.sqrt(alpha**2 + beta**2)             # (n,W)
    Fmod = np.fft.rfft(mod, axis=1)
    park_dc = mod.mean(axis=1)
    park_2f = 2.0 * np.abs(Fmod[:, B2]) / W
    epva_sf = park_2f / (park_dc + EPS)

    # --- harmonics per phase ---
    mag = lambda b: 2.0 * np.abs(F[:, :, b]) / W  # (n,3)
    h1, h3 = mag(B1), mag(B3)
    h3ratio = h3 / (h1 + EPS)                     # (n,3)
    # THD over harmonics 2..15 (clamped below Nyquist for low-fs experimental data)
    harm_bins = [bn(h) for h in range(2, 16) if bn(h) < nyq]
    thd_num = np.sqrt(np.sum((2.0 * np.abs(F[:, :, harm_bins]) / W) ** 2, axis=2))
    thd = thd_num / (h1 + EPS)                    # (n,3)

    # --- PVA ellipse (2x2 covariance eigen-decomposition) ---
    ma, mb = alpha.mean(1), beta.mean(1)
    cxx = (alpha**2).mean(1) - ma**2
    cyy = (beta**2).mean(1) - mb**2
    cxy = (alpha * beta).mean(1) - ma * mb
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    disc = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0.0))
    l1 = tr / 2 + disc        # major eigenvalue
    l2 = np.maximum(tr / 2 - disc, 0.0)
    pva_major, pva_minor = np.sqrt(l1 + EPS), np.sqrt(l2 + EPS)
    pva_axratio = pva_minor / (pva_major + EPS)
    pva_ecc = np.sqrt(np.maximum(1.0 - l2 / (l1 + EPS), 0.0))
    pva_tilt = np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy))

    # --- per-phase time-domain stats ---
    rms = np.sqrt((X**2).mean(2))                 # (n,3)
    peak = np.abs(X).max(2)
    crest = peak / (rms + EPS)
    mu = X.mean(2, keepdims=True)
    sd = X.std(2) + EPS
    xc = X - mu
    skew = (xc**3).mean(2) / sd**3
    kurt = (xc**4).mean(2) / sd**4 - 3.0

    # inter-phase imbalance
    rms_maxmin = rms.max(1) - rms.min(1)
    rms_std = rms.std(1)
    argmax_phase = rms.argmax(1)                  # 0=A,1=B,2=C

    d = dict(
        epva_sf=epva_sf, park_dc=park_dc, park_2f=park_2f,
        park_mod_mean=mod.mean(1), park_mod_std=mod.std(1),
        I1_abs=I1a, I2_abs=I2a, I0_abs=I0a,
        I2_I1_ratio=I2a / (I1a + EPS), I0_I1_ratio=I0a / (I1a + EPS),
        I1_angle_deg=np.degrees(np.angle(I1)),
        I2_angle_deg=np.degrees(np.angle(I2)),
        I2_angle_sin=np.sin(np.angle(I2)), I2_angle_cos=np.cos(np.angle(I2)),
        I2_rel_angle_deg=np.degrees(np.angle(I2 / (I1 + EPS))),
        I2_rel_angle_sin=np.sin(np.angle(I2 / (I1 + EPS))),
        I2_rel_angle_cos=np.cos(np.angle(I2 / (I1 + EPS))),
        pva_major=pva_major, pva_minor=pva_minor, pva_ecc=pva_ecc,
        pva_axratio=pva_axratio, pva_tilt_deg=pva_tilt,
        h3ratio_mean=h3ratio.mean(1), thd_mean=thd.mean(1),
        rms_imbalance_maxmin=rms_maxmin, rms_imbalance_std=rms_std,
        rms_diff_ab=np.abs(rms[:, 0] - rms[:, 1]),
        rms_diff_bc=np.abs(rms[:, 1] - rms[:, 2]),
        rms_diff_ca=np.abs(rms[:, 2] - rms[:, 0]),
        argmax_rms_phase=argmax_phase,
        I2_real=I2.real, I2_imag=I2.imag,   # for compensation pass
    )
    for j, ph in enumerate("abc"):
        d[f"ph{ph}_rms"] = rms[:, j]
        d[f"ph{ph}_peak"] = peak[:, j]
        d[f"ph{ph}_crest"] = crest[:, j]
        d[f"ph{ph}_std"] = sd[:, j]
        d[f"ph{ph}_skew"] = skew[:, j]
        d[f"ph{ph}_kurt"] = kurt[:, j]
        d[f"ph{ph}_h3ratio"] = h3ratio[:, j]
        d[f"ph{ph}_thd"] = thd[:, j]
    return d


def build_feature_frame(X, lab, verbose=False, fs=FS, f0=F0):
    """X (N,3,W) + label df (needs case_id,label) -> feature-only DataFrame (incl
    compensated I2c and |I1|-normalised n_* features), aligned to X rows, finite.
    fs/f0 select the harmonic bins (sim: 25 kHz/50 Hz; experimental: 1 kHz/60 Hz)."""
    n = X.shape[0]
    parts = []
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        parts.append(pd.DataFrame(chunk_features(np.asarray(X[s:e]), fs=fs, f0=f0)))
        if verbose:
            print(f"  features {e}/{n}")
    feat = pd.concat(parts, ignore_index=True)

    # pass 2: per-case compensated negative-sequence (baseline = mean I2 over healthy windows)
    feat["case_id"] = lab["case_id"].values
    feat["_label"] = lab["label"].values
    base = feat[feat["_label"] == 0].groupby("case_id")[["I2_real", "I2_imag"]].mean()
    base.columns = ["b_re", "b_im"]
    feat = feat.merge(base, on="case_id", how="left")
    feat["b_re"] = feat["b_re"].fillna(0.0)
    feat["b_im"] = feat["b_im"].fillna(0.0)
    I2c = (feat["I2_real"] - feat["b_re"]) + 1j * (feat["I2_imag"] - feat["b_im"])
    feat["I2c_abs"] = np.abs(I2c)
    feat["I2c_I1_ratio"] = feat["I2c_abs"] / (feat["I1_abs"] + EPS)
    feat["I2c_angle_deg"] = np.degrees(np.angle(I2c))
    feat["I2c_angle_sin"] = np.sin(np.angle(I2c))
    feat["I2c_angle_cos"] = np.cos(np.angle(I2c))

    # |I1|-normalised magnitude features (load-invariant magnitude SHAPE)
    I1n = feat["I1_abs"].to_numpy() + EPS
    for src in ["park_dc", "park_2f", "park_mod_mean", "park_mod_std", "pva_minor",
                "pva_major", "I0_abs", "I2_abs", "rms_imbalance_std",
                "rms_imbalance_maxmin", "rms_diff_ab", "rms_diff_bc", "rms_diff_ca"]:
        feat[f"n_{src}"] = feat[src].to_numpy() / I1n
    feat.drop(columns=["I2_real", "I2_imag", "b_re", "b_im", "_label", "case_id"], inplace=True)
    feat = feat.reset_index(drop=True)
    arr = feat.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        feat[list(feat.columns)] = np.nan_to_num(arr)
    return feat


def main():
    X = np.load(DS / "windows.npy", mmap_mode="r")
    lab = pd.read_csv(DS / "window_labels.csv")
    assert len(lab) == X.shape[0], "labels/windows length mismatch"

    feat = build_feature_frame(X, lab, verbose=True)
    feature_cols = list(feat.columns)
    out = pd.concat([lab.reset_index(drop=True), feat], axis=1)

    try:
        path = DS / "features.parquet"
        out.to_parquet(path, index=False)
    except Exception as ex:
        path = DS / "features.csv"
        out.to_csv(path, index=False)
        print(f"(parquet unavailable: {ex}; wrote CSV)")
    print(f"\nSaved {path}  shape={out.shape}  ({len(feature_cols)} features)")

    # ---- quick discriminative sanity ----
    g = out.groupby("state")
    print("\nhealthy vs faulty (mean):")
    for c in ["I2_I1_ratio", "I2c_I1_ratio", "epva_sf", "rms_imbalance_std", "pva_ecc"]:
        m = g[c].mean()
        print(f"  {c:18} healthy={m.get('healthy', float('nan')):.4f}  faulty={m.get('faulty', float('nan')):.4f}")
    print("\nfaulty: compensated NSC angle by faulted phase via circular mean "
          "(should differ ~120° apart):")
    fa = out[out.label == 1]
    for ph, gp in fa.groupby("phase"):
        cm = np.degrees(np.arctan2(gp["I2c_angle_sin"].mean(), gp["I2c_angle_cos"].mean()))
        R = np.hypot(gp["I2c_angle_sin"].mean(), gp["I2c_angle_cos"].mean())
        print(f"  phase {ph}: circular-mean angle={cm:7.1f}°  concentration R={R:.3f}  n={len(gp)}")
    print("\nfaulty: |I2c|/|I1| mean by severity (should rise with severity):")
    print(fa.groupby("severity_pct")["I2c_I1_ratio"].mean().to_string())
    print(f"\nfeature columns: {feature_cols}")


if __name__ == "__main__":
    main()
