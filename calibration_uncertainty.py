r"""
Phase 3 — Calibration (Stage-1) + Uncertainty (Stage-2).

Calibration: on Stage-1 GroupKFold OOF probabilities -> Brier score, ECE (10 bins),
reliability diagram.
Uncertainty: Stage-2 quantile regression (q=0.1/0.5/0.9) under GroupKFold -> prediction
intervals; report empirical coverage of the 80% interval, mean width, and that intervals
widen where predictions are wrong (useful uncertainty).

Run:  python calibration_uncertainty.py  -> dataset/calibration_uncertainty.json + .png
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splits import stratified_group_folds
from feature_sets import severity_features

DS = Path(r"D:\Naveen") / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV)}


def ece(prob, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    rows = []
    for i in range(n_bins):
        m = (prob > bins[i]) & (prob <= bins[i + 1]) if i else (prob >= 0) & (prob <= bins[1])
        if m.sum() == 0:
            rows.append((np.nan, np.nan, 0)); continue
        conf, acc = prob[m].mean(), y[m].mean()
        e += abs(acc - conf) * m.sum() / len(prob)
        rows.append((conf, acc, int(m.sum())))
    return e, rows


def main():
    res = {}

    # ---------- Calibration: Stage-1 ----------
    s1 = pd.read_csv(DS / "stage1_oof_predictions.csv")
    prob = s1["prob_faulty"].to_numpy()
    y = s1["label"].to_numpy(int)
    brier = brier_score_loss(y, prob)
    e, rows = ece(prob, y)
    res["stage1_calibration"] = dict(brier=float(brier), ece=float(e))
    print(f"Stage-1 calibration: Brier={brier:.4f}  ECE={e:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    cx = [r[0] for r in rows]; cy = [r[1] for r in rows]
    ax.plot(cx, cy, "o-", label="model")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Stage-1 reliability (ECE={e:.3f}, Brier={brier:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(DS / "calibration_reliability.png", dpi=130)

    # ---------- Uncertainty: Stage-2 quantile regression ----------
    df = pd.read_parquet(DS / "features.parquet")
    df = df[df.label == 1].reset_index(drop=True)
    fs = severity_features(df.columns)
    X = df[fs].to_numpy(np.float32)
    yr = df["severity_pct"].map(RANK).to_numpy(int)

    def qmodel(alpha):
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                            n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                            tree_method="hist", random_state=42)

    q10 = np.full(len(df), np.nan); q50 = np.full(len(df), np.nan); q90 = np.full(len(df), np.nan)
    for tr, te in stratified_group_folds(df, y_col="severity_pct"):
        q10[te] = qmodel(0.1).fit(X[tr], yr[tr]).predict(X[te])
        q50[te] = qmodel(0.5).fit(X[tr], yr[tr]).predict(X[te])
        q90[te] = qmodel(0.9).fit(X[tr], yr[tr]).predict(X[te])
    lo, hi = np.minimum(q10, q90), np.maximum(q10, q90)
    coverage = float(((yr >= lo) & (yr <= hi)).mean())
    width = float((hi - lo).mean())
    pred = np.clip(np.round(q50), 0, 6).astype(int)
    err = np.abs(pred - yr)
    w_correct = float((hi - lo)[err == 0].mean())
    w_wrong = float((hi - lo)[err > 0].mean())
    res["stage2_uncertainty"] = dict(interval="80% (q0.1-q0.9)", empirical_coverage=coverage,
                                     mean_width_levels=width, width_when_correct=w_correct,
                                     width_when_wrong=w_wrong)
    print(f"\nStage-2 uncertainty (80% interval): coverage={coverage:.3f} (target 0.80)  "
          f"mean width={width:.2f} levels")
    print(f"  interval width: correct preds={w_correct:.2f}  wrong preds={w_wrong:.2f} "
          f"(wider = useful uncertainty)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["correct", "wrong"], [w_correct, w_wrong], color=["tab:green", "tab:red"])
    ax.set_ylabel("mean 80% interval width (levels)")
    ax.set_title(f"Stage-2 uncertainty width (coverage={coverage:.2f})")
    fig.tight_layout(); fig.savefig(DS / "uncertainty_width.png", dpi=130)

    with open(DS / "calibration_uncertainty.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved calibration_uncertainty.json, calibration_reliability.png, uncertainty_width.png")


if __name__ == "__main__":
    main()
