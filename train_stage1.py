r"""
Phase 1 / M4 — Stage-1 detector: Healthy vs Faulty (XGBoost on Track-A features).

Evaluates under TWO leakage-safe protocols:
  (1) StratifiedGroupKFold (group=group_id)  -> in-distribution headline
  (2) Leave-One-Load-Out (LOLO)              -> PRIMARY generalisation test

Reports macro-F1 / accuracy / ROC-AUC, confusion matrices, per-load breakdown,
and a SHAP feature-importance audit (top features should be physical, not load proxies).
Saves out-of-fold predictions WITH labels (project convention).

Run:  python train_stage1.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_recall_fscore_support)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splits import stratified_group_folds, leave_one_load_out

ROOT = Path(r"D:\Naveen")
DS = ROOT / "dataset"

LABEL_COLS = {"window_id", "case_id", "group_id", "state", "label", "severity_pct",
              "phase", "load_label", "load_pct", "inception_s", "variant", "region",
              "src_start", "scale"}


def load_features():
    p = DS / "features.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(DS / "features.csv")
    feats = [c for c in df.columns if c not in LABEL_COLS]
    return df, feats


def make_model(y):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=n_neg / max(n_pos, 1), n_jobs=-1, random_state=42,
        tree_method="hist",
    )


def report(y_true, y_pred, tag):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    print(f"\n[{tag}]  acc={acc:.4f}  macro-F1={f1m:.4f}")
    print(f"  healthy: P={p[0]:.3f} R={r[0]:.3f} F1={f[0]:.3f} | "
          f"faulty: P={p[1]:.3f} R={r[1]:.3f} F1={f[1]:.3f}")
    print(f"  confusion [rows=true 0/1, cols=pred 0/1]:\n   {cm.tolist()}")
    return dict(acc=acc, macro_f1=f1m, confusion=cm.tolist())


def main():
    df, feats = load_features()
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    print(f"features={len(feats)}  windows={len(df)}  "
          f"(healthy={int((y==0).sum())}, faulty={int((y==1).sum())})")

    results = {}

    # ---------- Protocol 1: StratifiedGroupKFold ----------
    print("\n========== StratifiedGroupKFold (group=group_id) ==========")
    oof_prob = np.full(len(df), np.nan)
    fold_f1 = []
    for k, (tr, te) in enumerate(stratified_group_folds(df)):
        m = make_model(y[tr]).fit(X[tr], y[tr])
        prob = m.predict_proba(X[te])[:, 1]
        oof_prob[te] = prob
        fold_f1.append(f1_score(y[te], (prob >= 0.5).astype(int), average="macro"))
        print(f"  fold {k}: n_test={len(te)}  macro-F1={fold_f1[-1]:.4f}")
    pred = (oof_prob >= 0.5).astype(int)
    results["groupkfold"] = report(y, pred, "GroupKFold OOF")
    results["groupkfold"]["fold_f1_mean"] = float(np.mean(fold_f1))
    results["groupkfold"]["fold_f1_std"] = float(np.std(fold_f1))
    results["groupkfold"]["roc_auc"] = float(roc_auc_score(y, oof_prob))
    print(f"  fold macro-F1 = {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}  "
          f"| ROC-AUC(OOF)={results['groupkfold']['roc_auc']:.4f}")
    # save OOF predictions WITH labels
    oof = df[list(LABEL_COLS & set(df.columns))].copy()
    oof["prob_faulty"] = oof_prob
    oof["pred"] = pred
    oof.to_csv(DS / "stage1_oof_predictions.csv", index=False)

    # ---------- Protocol 2: Leave-One-Load-Out (PRIMARY) ----------
    print("\n========== Leave-One-Load-Out (PRIMARY) ==========")
    lolo_true, lolo_pred = [], []
    per_load = {}
    for L, tr, te in leave_one_load_out(df):
        m = make_model(y[tr]).fit(X[tr], y[tr])
        prob = m.predict_proba(X[te])[:, 1]
        pr = (prob >= 0.5).astype(int)
        f1m = f1_score(y[te], pr, average="macro")
        per_load[L] = dict(n_test=int(len(te)), macro_f1=float(f1m),
                           acc=float(accuracy_score(y[te], pr)),
                           faulty_recall=float(((pr == 1) & (y[te] == 1)).sum() / max((y[te] == 1).sum(), 1)),
                           healthy_recall=float(((pr == 0) & (y[te] == 0)).sum() / max((y[te] == 0).sum(), 1)))
        lolo_true.append(y[te]); lolo_pred.append(pr)
        print(f"  hold-out load={L:>3}: n_test={len(te):5d}  macro-F1={f1m:.4f}  "
              f"acc={per_load[L]['acc']:.4f}")
    lt, lp = np.concatenate(lolo_true), np.concatenate(lolo_pred)
    results["lolo"] = report(lt, lp, "LOLO pooled")
    results["lolo"]["per_load"] = per_load
    results["lolo"]["macro_f1_mean_over_loads"] = float(np.mean([v["macro_f1"] for v in per_load.values()]))
    print(f"  LOLO mean macro-F1 over loads = {results['lolo']['macro_f1_mean_over_loads']:.4f}")

    # ---------- SHAP leakage audit ----------
    print("\n========== SHAP feature importance (leakage audit) ==========")
    try:
        import shap
        tr, te = next(stratified_group_folds(df))
        m = make_model(y[tr]).fit(X[tr], y[tr])
        idx = np.random.RandomState(0).choice(te, size=min(2000, len(te)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(X[idx])
        imp = np.abs(sv).mean(0)
        top = np.argsort(imp)[::-1][:15]
        print("  top-15 features by mean|SHAP|:")
        for i in top:
            print(f"    {feats[i]:22} {imp[i]:.4f}")
        results["shap_top15"] = [(feats[i], float(imp[i])) for i in top]
    except Exception as ex:
        print(f"  SHAP failed ({ex}); using XGBoost gain importance:")
        tr, te = next(stratified_group_folds(df))
        m = make_model(y[tr]).fit(X[tr], y[tr])
        imp = m.feature_importances_
        top = np.argsort(imp)[::-1][:15]
        for i in top:
            print(f"    {feats[i]:22} {imp[i]:.4f}")
        results["shap_top15"] = [(feats[i], float(imp[i])) for i in top]

    # ---------- confusion figure ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, (name, cm) in zip(ax, [("GroupKFold", np.array(results["groupkfold"]["confusion"])),
                                  ("LOLO", np.array(results["lolo"]["confusion"]))]):
        im = a.imshow(cm, cmap="Blues")
        a.set_title(f"Stage-1 {name}")
        a.set_xticks([0, 1], ["pred H", "pred F"]); a.set_yticks([0, 1], ["true H", "true F"])
        for (r, c), v in np.ndenumerate(cm):
            a.text(c, r, str(v), ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(DS / "stage1_confusion.png", dpi=130)

    with open(DS / "stage1_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {DS/'stage1_oof_predictions.csv'}, {DS/'stage1_metrics.json'}, "
          f"{DS/'stage1_confusion.png'}")


if __name__ == "__main__":
    main()
