r"""
Phase 4 — Conformalised Quantile Regression (CQR) for Stage-2 severity.

Fixes the under-calibrated naive quantile intervals (Phase 3.3, coverage 0.63).
For each GroupKFold test fold: split the train groups into fit/calibration, fit q0.1/q0.9
quantile models on fit, compute conformity scores on calibration, and widen the interval by
the finite-sample-corrected (1-alpha) quantile of those scores. Target coverage = 80 %.

Run:  python conformal_uncertainty.py  -> dataset/conformal_uncertainty.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from splits import stratified_group_folds
from feature_sets import severity_features

DS = Path(r"D:\Naveen") / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV)}
ALPHA = 0.2  # 80% interval


def qmodel(a):
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=a, n_estimators=400,
                        max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        n_jobs=-1, tree_method="hist", random_state=42)


def main():
    df = pd.read_parquet(DS / "features.parquet")
    df = df[df.label == 1].reset_index(drop=True)
    fs = severity_features(df.columns)
    X = df[fs].to_numpy(np.float32)
    yr = df["severity_pct"].map(RANK).to_numpy(int)
    groups = df["group_id"].to_numpy()

    lo_oof = np.full(len(df), np.nan); hi_oof = np.full(len(df), np.nan)
    naive_cov, naive_w = [], []
    rng = np.random.default_rng(0)
    for tr, te in stratified_group_folds(df, y_col="severity_pct"):
        # split train groups -> fit / calibration (~25% groups for calibration)
        g = np.unique(groups[tr])
        cal_g = set(rng.choice(g, size=max(1, int(0.25 * len(g))), replace=False))
        is_cal = np.array([gg in cal_g for gg in groups[tr]])
        fit, cal = tr[~is_cal], tr[is_cal]
        qlo = qmodel(ALPHA / 2).fit(X[fit], yr[fit])
        qhi = qmodel(1 - ALPHA / 2).fit(X[fit], yr[fit])
        lo_c, hi_c = qlo.predict(X[cal]), qhi.predict(X[cal])
        E = np.maximum(lo_c - yr[cal], yr[cal] - hi_c)            # conformity scores
        n = len(cal)
        k = min(n, int(np.ceil((n + 1) * (1 - ALPHA))))
        Q = np.sort(E)[k - 1]                                     # conformal correction
        lo_t, hi_t = qlo.predict(X[te]), qhi.predict(X[te])
        lo_oof[te], hi_oof[te] = lo_t - Q, hi_t + Q
        # naive (no correction) for comparison
        naive_cov.append(((yr[te] >= lo_t) & (yr[te] <= hi_t)).mean())
        naive_w.append((hi_t - lo_t).mean())

    cov = float(((yr >= lo_oof) & (yr <= hi_oof)).mean())
    width = float((hi_oof - lo_oof).mean())
    res = dict(target_coverage=1 - ALPHA,
               conformal_coverage=cov, conformal_mean_width_levels=width,
               naive_coverage=float(np.mean(naive_cov)), naive_mean_width=float(np.mean(naive_w)))
    print(f"CQR (target {1-ALPHA:.0%}): coverage={cov:.3f}  mean width={width:.2f} levels")
    print(f"  (naive quantile: coverage={res['naive_coverage']:.3f}, width={res['naive_mean_width']:.2f})")
    with open(DS / "conformal_uncertainty.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {DS/'conformal_uncertainty.json'}")


if __name__ == "__main__":
    main()
