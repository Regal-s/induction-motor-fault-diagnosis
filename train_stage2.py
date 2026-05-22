r"""
Phase 2 / Stage-2 — Severity estimation (% shorted turns) as ORDINAL regression.

Faulty windows only (label==1). Targets are the 7 severity levels
{0.3,0.5,1,2,3,4,5} mapped to equal-spaced ranks 0..6; an XGBoost regressor predicts
a continuous rank which is rounded to the nearest level. This respects ordinality:
metrics are MAE-in-levels, quadratic-weighted kappa, exact & within-1 accuracy.

Evaluated under StratifiedGroupKFold (stratify=severity) and Leave-One-Load-Out (PRIMARY).
Saves OOF predictions WITH labels, metrics json, confusion figure.

Run:  python train_stage2.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splits import stratified_group_folds, leave_one_load_out
from feature_sets import LABEL_COLS, severity_features

ROOT = Path(r"D:\Naveen")
DS = ROOT / "dataset"
SEV_LEVELS = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV_LEVELS)}


def load_faulty():
    p = DS / "features.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(DS / "features.csv", low_memory=False)
    df = df[df["label"] == 1].reset_index(drop=True)
    feats = severity_features(df.columns)   # load-invariant set (Stage-2)
    return df, feats


def make_reg():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        objective="reg:squarederror", n_jobs=-1, random_state=42,
                        tree_method="hist")


def eval_ranks(true_rank, pred_cont, tag):
    pr = np.clip(np.round(pred_cont), 0, 6).astype(int)
    mae = mean_absolute_error(true_rank, pr)
    qwk = cohen_kappa_score(true_rank, pr, weights="quadratic", labels=list(range(7)))
    exact = float((pr == true_rank).mean())
    within1 = float((np.abs(pr - true_rank) <= 1).mean())
    mae_cont = float(np.mean(np.abs(pred_cont - true_rank)))
    cm = confusion_matrix(true_rank, pr, labels=range(7))
    print(f"\n[{tag}]  MAE(levels)={mae:.3f}  MAE(cont)={mae_cont:.3f}  "
          f"QWK={qwk:.4f}  exact={exact:.3f}  within-1={within1:.3f}")
    return dict(mae_levels=float(mae), mae_cont=mae_cont, qwk=float(qwk),
                exact=exact, within1=within1, confusion=cm.tolist())


def main():
    df, feats = load_faulty()
    X = df[feats].to_numpy(np.float32)
    yr = df["severity_pct"].map(RANK).to_numpy(int)
    print(f"faulty windows={len(df)}  features={len(feats)}  severity levels={SEV_LEVELS}")
    results = {}

    # ---- StratifiedGroupKFold (stratify by severity) ----
    print("\n========== StratifiedGroupKFold (stratify=severity) ==========")
    oof = np.full(len(df), np.nan)
    for k, (tr, te) in enumerate(stratified_group_folds(df, y_col="severity_pct")):
        m = make_reg().fit(X[tr], yr[tr])
        oof[te] = m.predict(X[te])
        mae_k = mean_absolute_error(yr[te], np.clip(np.round(oof[te]), 0, 6))
        print(f"  fold {k}: n_test={len(te)}  MAE(levels)={mae_k:.3f}")
    results["groupkfold"] = eval_ranks(yr, oof, "GroupKFold OOF")

    # per-case aggregation (median predicted rank)
    cf = df[["case_id", "severity_pct"]].copy()
    cf["pred_cont"] = oof
    agg = cf.groupby("case_id").agg(true=("severity_pct", "first"),
                                    pred=("pred_cont", "median"))
    ct = agg["true"].map(RANK).to_numpy()
    cp = np.clip(np.round(agg["pred"]), 0, 6).astype(int)
    results["groupkfold_percase"] = dict(
        mae_levels=float(mean_absolute_error(ct, cp)),
        exact=float((cp == ct).mean()), within1=float((np.abs(cp - ct) <= 1).mean()))
    print(f"  per-case: MAE(levels)={results['groupkfold_percase']['mae_levels']:.3f}  "
          f"exact={results['groupkfold_percase']['exact']:.3f}  "
          f"within-1={results['groupkfold_percase']['within1']:.3f}")

    # ---- Leave-One-Load-Out (PRIMARY) ----
    print("\n========== Leave-One-Load-Out (PRIMARY) ==========")
    lt, lp, per_load = [], [], {}
    for L, tr, te in leave_one_load_out(df):
        m = make_reg().fit(X[tr], yr[tr])
        pc = m.predict(X[te])
        pr = np.clip(np.round(pc), 0, 6).astype(int)
        per_load[L] = dict(n_test=int(len(te)),
                           mae_levels=float(mean_absolute_error(yr[te], pr)),
                           within1=float((np.abs(pr - yr[te]) <= 1).mean()))
        lt.append(yr[te]); lp.append(pc)
        print(f"  hold-out load={L:>3}: n={len(te):5d}  MAE(levels)={per_load[L]['mae_levels']:.3f}  "
              f"within-1={per_load[L]['within1']:.3f}")
    results["lolo"] = eval_ranks(np.concatenate(lt), np.concatenate(lp), "LOLO pooled")
    results["lolo"]["per_load"] = per_load
    results["lolo"]["mae_mean_over_loads"] = float(np.mean([v["mae_levels"] for v in per_load.values()]))
    print(f"  LOLO mean MAE(levels) over loads = {results['lolo']['mae_mean_over_loads']:.3f}")

    # ---- SHAP ----
    print("\n========== SHAP (top features) ==========")
    try:
        import shap
        tr, te = next(stratified_group_folds(df, y_col="severity_pct"))
        m = make_reg().fit(X[tr], yr[tr])
        idx = np.random.RandomState(0).choice(te, min(2000, len(te)), replace=False)
        imp = np.abs(shap.TreeExplainer(m).shap_values(X[idx])).mean(0)
    except Exception as ex:
        print(f"  SHAP failed ({ex}); using gain importance")
        tr, te = next(stratified_group_folds(df, y_col="severity_pct"))
        imp = make_reg().fit(X[tr], yr[tr]).feature_importances_
    for i in np.argsort(imp)[::-1][:12]:
        print(f"    {feats[i]:22} {imp[i]:.4f}")
    results["shap_top12"] = [(feats[i], float(imp[i])) for i in np.argsort(imp)[::-1][:12]]

    # ---- save ----
    out = df[list(LABEL_COLS & set(df.columns))].copy()
    out["true_rank"] = yr
    out["pred_rank_cont"] = oof
    out["pred_rank"] = np.clip(np.round(oof), 0, 6).astype(int)
    out["pred_severity_pct"] = [SEV_LEVELS[r] for r in out["pred_rank"]]
    out.to_csv(DS / "stage2_oof_predictions.csv", index=False)

    cm = np.array(results["groupkfold"]["confusion"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Stage-2 severity (GroupKFold OOF)")
    ax.set_xticks(range(7), [str(s) for s in SEV_LEVELS]); ax.set_xlabel("pred %")
    ax.set_yticks(range(7), [str(s) for s in SEV_LEVELS]); ax.set_ylabel("true %")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=8,
                color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(DS / "stage2_confusion.png", dpi=130)
    with open(DS / "stage2_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved stage2_oof_predictions.csv, stage2_metrics.json, stage2_confusion.png")


if __name__ == "__main__":
    main()
