r"""
Stage-2 improvement experiment: does restricting to LOAD-INVARIANT (dimensionless)
features and/or COARSE severity recover cross-load (LOLO) performance?

Compares 4 configs:
  fine  severity (7 levels, ordinal regression) x {all features, invariant features}
  coarse severity (Incipient/Moderate/Severe, 3-class) x {all, invariant}
Headline = LOLO; we also print the worst (NL) fold.

Run:  python stage2_experiment.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from splits import stratified_group_folds, leave_one_load_out

DS = Path(r"D:\Naveen") / "dataset"
LABEL_COLS = {"window_id", "case_id", "group_id", "state", "label", "severity_pct",
              "phase", "load_label", "load_pct", "inception_s", "variant", "region",
              "src_start", "scale"}
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV)}
COARSE = {0.3: 0, 0.5: 0, 1.0: 0, 2.0: 1, 3.0: 1, 4.0: 2, 5.0: 2}  # Incipient/Moderate/Severe

# dimensionless / shape / angle features (load-invariant by construction)
INVARIANT = [
    "epva_sf", "I2_I1_ratio", "I0_I1_ratio", "I2c_I1_ratio",
    "pva_ecc", "pva_axratio", "pva_tilt_deg",
    "h3ratio_mean", "thd_mean",
    "pha_crest", "phb_crest", "phc_crest",
    "pha_skew", "phb_skew", "phc_skew",
    "pha_kurt", "phb_kurt", "phc_kurt",
    "pha_h3ratio", "phb_h3ratio", "phc_h3ratio",
    "pha_thd", "phb_thd", "phc_thd",
]


def load_faulty():
    p = DS / "features.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(DS / "features.csv", low_memory=False)
    return df[df.label == 1].reset_index(drop=True)


def reg(): return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                              random_state=42, tree_method="hist")


def clf(k): return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, num_class=k,
                                objective="multi:softprob", eval_metric="mlogloss",
                                n_jobs=-1, random_state=42, tree_method="hist")


def fine(df, X):
    yr = df.severity_pct.map(RANK).to_numpy(int)
    # groupkfold within-1
    oof = np.full(len(df), np.nan)
    for tr, te in stratified_group_folds(df, y_col="severity_pct"):
        oof[te] = reg().fit(X[tr], yr[tr]).predict(X[te])
    gk_w1 = (np.abs(np.clip(np.round(oof), 0, 6) - yr) <= 1).mean()
    gk_mae = mean_absolute_error(yr, np.clip(np.round(oof), 0, 6))
    # LOLO
    w1s, maes, perload = [], [], {}
    lt, lp = [], []
    for L, tr, te in leave_one_load_out(df):
        pc = reg().fit(X[tr], yr[tr]).predict(X[te])
        pr = np.clip(np.round(pc), 0, 6).astype(int)
        perload[L] = float((np.abs(pr - yr[te]) <= 1).mean())
        lt.append(yr[te]); lp.append(pr)
    lt, lp = np.concatenate(lt), np.concatenate(lp)
    return dict(gk_within1=gk_w1, gk_mae=gk_mae,
                lolo_within1=float((np.abs(lp - lt) <= 1).mean()),
                lolo_mae=float(mean_absolute_error(lt, lp)),
                lolo_NL_within1=perload.get("NL"))


def coarse(df, X):
    yc = df.severity_pct.map(COARSE).to_numpy(int)
    # stratify proxy: use severity_pct grouping ok
    oof = np.full(len(df), -1, int)
    for tr, te in stratified_group_folds(df, y_col="severity_pct"):
        oof[te] = clf(3).fit(X[tr], yc[tr]).predict(X[te])
    gk_f1 = f1_score(yc, oof, average="macro")
    lt, lp, perload = [], [], {}
    for L, tr, te in leave_one_load_out(df):
        pr = clf(3).fit(X[tr], yc[tr]).predict(X[te])
        perload[L] = float(f1_score(yc[te], pr, average="macro"))
        lt.append(yc[te]); lp.append(pr)
    lt, lp = np.concatenate(lt), np.concatenate(lp)
    return dict(gk_macroF1=gk_f1, lolo_macroF1=float(f1_score(lt, lp, average="macro")),
                lolo_acc=float(accuracy_score(lt, lp)), lolo_NL_f1=perload.get("NL"))


def main():
    df = load_faulty()
    all_feats = [c for c in df.columns if c not in LABEL_COLS]
    inv = [f for f in INVARIANT if f in all_feats]

    # |I1|-normalised magnitude features (load-invariant magnitude SHAPE)
    I1 = df["I1_abs"].to_numpy() + 1e-9
    for src in ["park_dc", "park_2f", "park_mod_mean", "pva_minor", "pva_major",
                "rms_imbalance_std", "rms_imbalance_maxmin",
                "rms_diff_ab", "rms_diff_bc", "rms_diff_ca", "I0_abs", "I2_abs"]:
        if src in df.columns:
            df[f"n_{src}"] = df[src].to_numpy() / I1
    norm_feats = [f"n_{s}" for s in ["park_dc", "park_2f", "park_mod_mean", "pva_minor",
                  "pva_major", "rms_imbalance_std", "rms_imbalance_maxmin",
                  "rms_diff_ab", "rms_diff_bc", "rms_diff_ca", "I0_abs", "I2_abs"] if f"n_{s}" in df.columns]
    inv_plus = inv + norm_feats
    print(f"faulty windows={len(df)}  all_feats={len(all_feats)}  "
          f"invariant_feats={len(inv)}  invariant+I1norm={len(inv_plus)}\n")

    Xall = df[all_feats].to_numpy(np.float32)
    Xinv = df[inv].to_numpy(np.float32)
    Xinvp = df[inv_plus].to_numpy(np.float32)

    print("===== FINE severity (7 levels, ordinal regression) =====")
    for name, Xs in [("all features", Xall), ("invariant features", Xinv),
                     ("invariant+I1norm", Xinvp)]:
        r = fine(df, Xs)
        print(f"  [{name:18}] GK within-1={r['gk_within1']:.3f} (MAE {r['gk_mae']:.3f}) | "
              f"LOLO within-1={r['lolo_within1']:.3f} (MAE {r['lolo_mae']:.3f}) | NL within-1={r['lolo_NL_within1']:.3f}")

    print("\n===== COARSE severity (Incipient/Moderate/Severe, 3-class) =====")
    for name, Xs in [("all features", Xall), ("invariant features", Xinv)]:
        r = coarse(df, Xs)
        print(f"  [{name:18}] GK macro-F1={r['gk_macroF1']:.3f} | "
              f"LOLO macro-F1={r['lolo_macroF1']:.3f} (acc {r['lolo_acc']:.3f}) | NL macro-F1={r['lolo_NL_f1']:.3f}")


if __name__ == "__main__":
    main()
