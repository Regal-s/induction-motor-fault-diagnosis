r"""Compute physics features on the experimental ibarram windows using the SAME extractor
(parameterized for 1 kHz / 60 Hz), and sanity-check that the inter-turn physics holds on
real data. -> experimental/ibarram/features.parquet"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Naveen")
sys.path.insert(0, str(ROOT))
from features import build_feature_frame   # noqa: E402

OUT = ROOT / "experimental" / "ibarram"
FS, F0 = 1000.0, 60.0

X = np.load(OUT / "windows.npy")
lab = pd.read_csv(OUT / "window_labels.csv")
feat = build_feature_frame(X, lab, verbose=True, fs=FS, f0=F0)
out = pd.concat([lab.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
out.to_parquet(OUT / "features.parquet", index=False)
print(f"\nSaved {OUT/'features.parquet'}  shape={out.shape}")

# --- physics sanity on REAL data ---
g = out.groupby("state")
print("\nhealthy vs faulty (mean):")
for c in ["I2_I1_ratio", "epva_sf", "rms_imbalance_std", "pva_ecc"]:
    m = g[c].mean()
    print(f"  {c:18} healthy={m.get('healthy', float('nan')):.4f}  faulty={m.get('faulty', float('nan')):.4f}")

print("\nfaulty: |I2|/|I1| mean by severity (should RISE with severity on real data):")
print(out[out.label == 1].groupby("severity_pct")["I2_I1_ratio"].mean().to_string())

print("\nfaulty: negative-seq angle by faulted phase (circular mean, should differ ~120 deg):")
fa = out[out.label == 1]
for ph, gp in fa.groupby("phase"):
    cm = np.degrees(np.arctan2(np.sin(np.radians(gp["I2_angle_deg"])).mean(),
                               np.cos(np.radians(gp["I2_angle_deg"])).mean()))
    print(f"  phase {ph}: {cm:7.1f} deg  (n={len(gp)})")
