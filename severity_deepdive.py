r"""
Phase 3c — Severity cross-load deep-dive.

(1) Feature stability across loads: for each feature on faulty windows, compute a
    severity Fisher ratio (between-severity / within-severity variance) and a load-drift
    score (spread of per-load means). Good cross-load severity features have HIGH Fisher
    + LOW load-drift. Confirms ratios (|I2|/|I1|, EPVA) are load-stable while raw
    magnitudes drift.
(2) Load-aware severity: add measured load (load_pct) to the severity model and re-test
    GroupKFold + LOLO within-1 (load is measurable in deployment).

Run:  python severity_deepdive.py  -> dataset/severity_deepdive.json + .png
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splits import stratified_group_folds, leave_one_load_out
from feature_sets import severity_features, LABEL_COLS

DS = Path(r"D:\Naveen") / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV)}
LOADS = ["NL", "20", "40", "60", "80", "100"]


def fisher_and_drift(df, feats):
    """Per feature: severity Fisher ratio (pooled) and load-drift (std of per-load means
    normalised by overall std)."""
    rows = []
    for f in feats:
        x = df[f].to_numpy(float)
        ov_var = x.var() + 1e-12
        # between-severity variance / within-severity variance
        gb = df.groupby("severity_pct")[f]
        means = gb.mean(); counts = gb.count(); within = gb.var(ddof=0).fillna(0)
        grand = x.mean()
        between = (counts * (means - grand) ** 2).sum() / counts.sum()
        within_v = (counts * within).sum() / counts.sum() + 1e-12
        fisher = between / within_v
        # load drift: std of per-load mean / overall std
        load_means = df.groupby("load_label")[f].mean()
        drift = load_means.std() / (np.sqrt(ov_var))
        rows.append((f, fisher, drift))
    return pd.DataFrame(rows, columns=["feature", "fisher", "load_drift"]).sort_values("fisher", ascending=False)


def sev_within1(df, feat_cols):
    X = df[feat_cols].to_numpy(np.float32)
    yr = df["severity_pct"].map(RANK).to_numpy(int)

    def reg():
        return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=42)
    oof = np.full(len(df), np.nan)
    for tr, te in stratified_group_folds(df, y_col="severity_pct"):
        oof[te] = reg().fit(X[tr], yr[tr]).predict(X[te])
    gk = float((np.abs(np.clip(np.round(oof), 0, 6) - yr) <= 1).mean())
    lt, lp, per = [], [], {}
    for L, tr, te in leave_one_load_out(df):
        pr = np.clip(np.round(reg().fit(X[tr], yr[tr]).predict(X[te])), 0, 6).astype(int)
        per[L] = float((np.abs(pr - yr[te]) <= 1).mean())
        lt.append(yr[te]); lp.append(pr)
    lt, lp = np.concatenate(lt), np.concatenate(lp)
    lolo = float((np.abs(lp - lt) <= 1).mean())
    return dict(gk_within1=gk, lolo_within1=lolo, per_load=per)


def main():
    df = pd.read_parquet(DS / "features.parquet")
    df = df[df.label == 1].reset_index(drop=True)
    sf = severity_features(df.columns)
    res = {}

    # (1) feature stability
    fd = fisher_and_drift(df, sf)
    print("Top severity-discriminative features (Fisher) and their cross-load drift:")
    print(fd.head(12).to_string(index=False))
    res["feature_stability_top12"] = fd.head(12).to_dict("records")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fd["load_drift"], fd["fisher"], s=20)
    for _, r in fd.head(10).iterrows():
        ax.annotate(r["feature"], (r["load_drift"], r["fisher"]), fontsize=7)
    ax.set_xlabel("load drift (lower=more load-stable)"); ax.set_ylabel("severity Fisher ratio (higher=better)")
    ax.set_title("Severity features: discriminability vs load-stability")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(DS / "severity_feature_stability.png", dpi=130)

    # (2) load-aware severity
    base = sev_within1(df, sf)
    aware = sev_within1(df, sf + ["load_pct"])
    res["severity_invariant"] = base
    res["severity_load_aware"] = aware
    print(f"\nSeverity within-1 — invariant feats : GK={base['gk_within1']:.3f}  LOLO={base['lolo_within1']:.3f}")
    print(f"Severity within-1 — +load_pct      : GK={aware['gk_within1']:.3f}  LOLO={aware['lolo_within1']:.3f}")
    print("  (load-aware helps in-distribution; LOLO tests UNSEEN loads so gains are limited — extrapolation.)")

    with open(DS / "severity_deepdive.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved severity_deepdive.json, severity_feature_stability.png")


if __name__ == "__main__":
    main()
