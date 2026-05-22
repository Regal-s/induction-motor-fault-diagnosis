r"""
Phase 2 / Stage-3 — Faulted-phase identification (A/B/C).

Faulty windows only (label==1). XGBoost 3-class on Track-A features (the wrap-free
sin/cos negative-sequence angle features carry the main localisation cue).
Evaluated under StratifiedGroupKFold (stratify=phase) and Leave-One-Load-Out (PRIMARY).
Reports per-phase F1 + confusion; per-case majority vote. SHAP audit. Saves labeled OOF.

Run:  python train_stage3.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support)
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
PHASES = ["A", "B", "C"]
PMAP = {"A": 0, "B": 1, "C": 2}


def load_faulty():
    p = DS / "features.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(DS / "features.csv", low_memory=False)
    df = df[df["label"] == 1].reset_index(drop=True)
    feats = [c for c in df.columns if c not in LABEL_COLS]
    return df, feats


def make_model():
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                         n_jobs=-1, random_state=42, tree_method="hist")


def report(y, pred, tag):
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    print(f"\n[{tag}]  acc={acc:.4f}  macro-F1={f1m:.4f}")
    for j, ph in enumerate(PHASES):
        print(f"  phase {ph}: P={p[j]:.3f} R={r[j]:.3f} F1={f[j]:.3f}")
    print(f"  confusion [rows=true A/B/C, cols=pred]:\n   {cm.tolist()}")
    return dict(acc=float(acc), macro_f1=float(f1m), confusion=cm.tolist(),
                per_phase_f1={ph: float(f[j]) for j, ph in enumerate(PHASES)})


def main():
    df, feats = load_faulty()
    X = df[feats].to_numpy(np.float32)
    y = df["phase"].map(PMAP).to_numpy(int)
    print(f"faulty windows={len(df)}  features={len(feats)}  "
          f"phase counts={dict(df['phase'].value_counts())}")
    results = {}

    # ---- StratifiedGroupKFold (stratify by phase) ----
    print("\n========== StratifiedGroupKFold (stratify=phase) ==========")
    oof = np.full(len(df), -1, int)
    fold_f1 = []
    for k, (tr, te) in enumerate(stratified_group_folds(df, y_col="phase")):
        m = make_model().fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        fold_f1.append(f1_score(y[te], oof[te], average="macro"))
        print(f"  fold {k}: n_test={len(te)}  macro-F1={fold_f1[-1]:.4f}")
    results["groupkfold"] = report(y, oof, "GroupKFold OOF")
    results["groupkfold"]["fold_f1_mean"] = float(np.mean(fold_f1))
    results["groupkfold"]["fold_f1_std"] = float(np.std(fold_f1))

    # per-case majority vote
    cf = df[["case_id", "phase"]].copy(); cf["pred"] = oof
    agg = cf.groupby("case_id").agg(true=("phase", "first"),
                                    pred=("pred", lambda s: s.value_counts().idxmax()))
    ct = agg["true"].map(PMAP).to_numpy(); cp = agg["pred"].to_numpy()
    results["groupkfold_percase"] = dict(acc=float((cp == ct).mean()),
                                         macro_f1=float(f1_score(ct, cp, average="macro")))
    print(f"  per-case majority vote: acc={results['groupkfold_percase']['acc']:.4f}  "
          f"macro-F1={results['groupkfold_percase']['macro_f1']:.4f}")

    # ---- Leave-One-Load-Out (PRIMARY) ----
    print("\n========== Leave-One-Load-Out (PRIMARY) ==========")
    lt, lp, per_load = [], [], {}
    for L, tr, te in leave_one_load_out(df):
        m = make_model().fit(X[tr], y[tr])
        pr = m.predict(X[te])
        per_load[L] = dict(n_test=int(len(te)), acc=float(accuracy_score(y[te], pr)),
                           macro_f1=float(f1_score(y[te], pr, average="macro")))
        lt.append(y[te]); lp.append(pr)
        print(f"  hold-out load={L:>3}: n={len(te):5d}  acc={per_load[L]['acc']:.4f}  "
              f"macro-F1={per_load[L]['macro_f1']:.4f}")
    results["lolo"] = report(np.concatenate(lt), np.concatenate(lp), "LOLO pooled")
    results["lolo"]["per_load"] = per_load
    results["lolo"]["macro_f1_mean_over_loads"] = float(np.mean([v["macro_f1"] for v in per_load.values()]))
    print(f"  LOLO mean macro-F1 over loads = {results['lolo']['macro_f1_mean_over_loads']:.4f}")

    # ---- SHAP ----
    print("\n========== SHAP (top features) ==========")
    try:
        import shap
        tr, te = next(stratified_group_folds(df, y_col="phase"))
        m = make_model().fit(X[tr], y[tr])
        idx = np.random.RandomState(0).choice(te, min(2000, len(te)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(X[idx])
        imp = np.abs(np.array(sv)).mean(axis=tuple(range(np.array(sv).ndim - 1)))
    except Exception as ex:
        print(f"  SHAP failed ({ex}); using gain importance")
        tr, te = next(stratified_group_folds(df, y_col="phase"))
        imp = make_model().fit(X[tr], y[tr]).feature_importances_
    for i in np.argsort(imp)[::-1][:12]:
        print(f"    {feats[i]:22} {imp[i]:.4f}")
    results["shap_top12"] = [(feats[i], float(imp[i])) for i in np.argsort(imp)[::-1][:12]]

    # ---- save ----
    out = df[list(LABEL_COLS & set(df.columns))].copy()
    out["true_phase"] = df["phase"].values
    out["pred_phase"] = [PHASES[p] for p in oof]
    out.to_csv(DS / "stage3_oof_predictions.csv", index=False)
    cm = np.array(results["groupkfold"]["confusion"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Greens")
    ax.set_title("Stage-3 phase (GroupKFold OOF)")
    ax.set_xticks(range(3), PHASES); ax.set_yticks(range(3), PHASES)
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(DS / "stage3_confusion.png", dpi=130)
    with open(DS / "stage3_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved stage3_oof_predictions.csv, stage3_metrics.json, stage3_confusion.png")


if __name__ == "__main__":
    main()
