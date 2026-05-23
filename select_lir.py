r"""
M3 — Load-Invariance-Regularized mRMR (LIR-mRMR) feature selection.

Greedy selection with criterion
    J(f) = Rel(f) - lambda * Redundancy(f, selected) - gamma * LoadInstability(f)
where Rel = ANOVA F-score w.r.t. severity, Redundancy = mean |Pearson corr| to already-selected
features, and LoadInstability = how much the feature's class-conditional mean drifts across load
(normalized). gamma=0 recovers plain mRMR. Demonstrated on the LOAD-DEPENDENT severity task
(load-blind features) under Leave-One-Load-Out: LIR-mRMR should select load-stable features and
generalize better cross-load with fewer features than mRMR.

Run:  python select_lir.py  -> dataset/lir_selection.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from feature_sets import all_features, LABEL_COLS  # noqa: E402
from splits import leave_one_load_out  # noqa: E402

SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]; RANK = {s: i for i, s in enumerate(SEV)}
LOADS = ["NL", "20", "40", "60", "80", "100"]


def minmax(v):
    v = np.asarray(v, float); r = v.max() - v.min()
    return (v - v.min()) / r if r > 0 else np.zeros_like(v)


def load_instability(df, feats):
    """Per feature: mean over severity classes of std-across-load of the class mean,
    normalized by the feature's global std. High = drifts with load."""
    inst = {}
    gstd = df[feats].std() + 1e-9
    for f in feats:
        per = df.groupby(["severity_pct", "load_label"])[f].mean().unstack()
        drift = per.std(axis=1).mean()        # avg over severities of std across loads
        inst[f] = drift / gstd[f]
    return np.array([inst[f] for f in feats])


def lir_mrmr(df, feats, k, lam=1.0, gamma=0.0):
    X = df[feats].to_numpy(float)
    yr = df["severity_pct"].map(RANK).to_numpy(int)
    rel = minmax(np.nan_to_num(f_classif(X, yr)[0]))
    inst = minmax(load_instability(df, feats))
    corr = np.abs(np.corrcoef(X, rowvar=False)); np.fill_diagonal(corr, 0)
    fi = {f: i for i, f in enumerate(feats)}
    selected, remaining = [], list(feats)
    # first pick: relevance minus load penalty
    score0 = rel - gamma * inst
    first = feats[int(np.argmax(score0))]
    selected.append(first); remaining.remove(first)
    while len(selected) < k and remaining:
        best, best_s = None, -1e9
        sel_idx = [fi[s] for s in selected]
        for f in remaining:
            i = fi[f]
            red = corr[i, sel_idx].mean()
            s = rel[i] - lam * red - gamma * inst[i]
            if s > best_s:
                best_s, best = s, f
        selected.append(best); remaining.remove(best)
    return selected, dict(zip(feats, inst))


def lolo_within1(df, feats):
    X = df[feats].to_numpy(np.float32); yr = df["severity_pct"].map(RANK).to_numpy(int)
    lt, lp = [], []
    for L, tr, te in leave_one_load_out(df):
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                         random_state=42).fit(X[tr], yr[tr])
        pr = np.clip(np.round(m.predict(X[te])), 0, 6).astype(int)
        lt.append(yr[te]); lp.append(pr)
    lt, lp = np.concatenate(lt), np.concatenate(lp)
    return float((np.abs(lp - lt) <= 1).mean())


def main():
    df = pd.read_parquet(ROOT / "dataset" / "features.parquet")
    df = df[df.label == 1].reset_index(drop=True)
    feats = all_features(df.columns)
    print(f"faulty windows={len(df)}  candidate features={len(feats)}")
    inst_all = dict(zip(feats, minmax(load_instability(df, feats))))

    res = {"per_k": {}}
    for k in [5, 10, 15, 20]:
        sel_mrmr, _ = lir_mrmr(df, feats, k, gamma=0.0)
        sel_lir, _ = lir_mrmr(df, feats, k, gamma=1.0)
        w_mrmr = lolo_within1(df, sel_mrmr)
        w_lir = lolo_within1(df, sel_lir)
        drift_mrmr = float(np.mean([inst_all[f] for f in sel_mrmr]))
        drift_lir = float(np.mean([inst_all[f] for f in sel_lir]))
        res["per_k"][k] = dict(
            mrmr_within1=round(w_mrmr, 4), lir_within1=round(w_lir, 4),
            mrmr_meandrift=round(drift_mrmr, 4), lir_meandrift=round(drift_lir, 4),
            mrmr_feats=sel_mrmr, lir_feats=sel_lir)
        print(f"k={k:2d} | LOLO within-1: mRMR={w_mrmr:.4f}  LIR-mRMR={w_lir:.4f} "
              f"| mean load-drift: mRMR={drift_mrmr:.3f}  LIR={drift_lir:.3f}")
    res["all_features_within1"] = round(lolo_within1(df, feats), 4)
    print(f"\nall {len(feats)} features LOLO within-1 = {res['all_features_within1']:.4f}")
    print("\nLIR-mRMR top-10 selected:", res["per_k"][10]["lir_feats"])
    print("mRMR     top-10 selected:", res["per_k"][10]["mrmr_feats"])
    json.dump(res, open(ROOT / "dataset" / "lir_selection.json", "w"), indent=2)
    print("\nsaved dataset/lir_selection.json")


if __name__ == "__main__":
    main()
