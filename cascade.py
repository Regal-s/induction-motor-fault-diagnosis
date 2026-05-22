r"""
Phase 2 — End-to-end CASCADE: Stage1 (detect) -> {Stage2 severity, Stage3 phase}.

For each CV fold, all three stages are trained on the fold's TRAIN windows
(Stages 2/3 on the faulty train windows only) and applied to TEST windows. Stages 2/3
are scored on Stage-1's PREDICTED-faulty windows (true end-to-end), plus a conditional
score on (true-faulty & pred-faulty). Per-case aggregation (majority/median) included.

Feature sets: Stage1 & Stage3 use ALL features; Stage2 uses the load-invariant set.
Protocols: StratifiedGroupKFold (in-distribution) + Leave-One-Load-Out (PRIMARY).

Run:  python cascade.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from splits import stratified_group_folds, leave_one_load_out
from feature_sets import LABEL_COLS, all_features, severity_features

DS = Path(r"D:\Naveen") / "dataset"
SEV_LEVELS = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV_LEVELS)}
PHASES = ["A", "B", "C"]
PMAP = {"A": 0, "B": 1, "C": 2}


def s1_model(y):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
                         scale_pos_weight=nneg / max(npos, 1), n_jobs=-1, random_state=42, tree_method="hist")


def s2_model():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1,
                        random_state=42, tree_method="hist")


def s3_model():
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, num_class=3, objective="multi:softprob",
                         eval_metric="mlogloss", n_jobs=-1, random_state=42, tree_method="hist")


def run_oof(df, Xall, Xsev, label, rank, phase, folds):
    """folds: iterable of (train_idx, test_idx). Returns OOF prediction arrays."""
    n = len(df)
    p_lab = np.full(n, -1, int)
    p_rank = np.full(n, -1, int)
    p_phase = np.full(n, -1, int)
    for tr, te in folds:
        f = label[tr] == 1
        s1 = s1_model(label[tr]).fit(Xall[tr], label[tr])
        s2 = s2_model().fit(Xsev[tr][f], rank[tr][f])
        s3 = s3_model().fit(Xall[tr][f], phase[tr][f])
        p_lab[te] = s1.predict(Xall[te])
        p_rank[te] = np.clip(np.round(s2.predict(Xsev[te])), 0, 6).astype(int)
        p_phase[te] = s3.predict(Xall[te])
    return p_lab, p_rank, p_phase


def evaluate(df, label, rank, phase, p_lab, p_rank, p_phase, tag):
    true_f = label == 1
    pred_f = p_lab == 1
    # detection
    det_acc = accuracy_score(label, p_lab)
    det_f1 = f1_score(label, p_lab, average="macro")
    # conditional on true-faulty AND pred-faulty
    cond = true_f & pred_f
    sev_w1 = float((np.abs(p_rank[cond] - rank[cond]) <= 1).mean()) if cond.any() else float("nan")
    sev_exact = float((p_rank[cond] == rank[cond]).mean()) if cond.any() else float("nan")
    ph_acc = float((p_phase[cond] == phase[cond]).mean()) if cond.any() else float("nan")
    # end-to-end window exact-match: healthy correct, or faulty + sev-exact + phase
    correct_exact = np.where(true_f,
                             pred_f & (p_rank == rank) & (p_phase == phase),
                             ~pred_f)
    correct_w1 = np.where(true_f,
                          pred_f & (np.abs(p_rank - rank) <= 1) & (p_phase == phase),
                          ~pred_f)
    res = dict(detect_acc=float(det_acc), detect_macroF1=float(det_f1),
               cond_severity_within1=sev_w1, cond_severity_exact=sev_exact,
               cond_phase_acc=ph_acc,
               endto_end_exact=float(correct_exact.mean()),
               endto_end_within1=float(correct_w1.mean()),
               faulty_detect_recall=float((pred_f & true_f).sum() / max(true_f.sum(), 1)))
    print(f"\n[{tag}] detect: acc={det_acc:.4f} F1={det_f1:.4f} | "
          f"cond(sev within1={sev_w1:.3f} exact={sev_exact:.3f}, phase acc={ph_acc:.3f})")
    print(f"   END-TO-END window: exact={res['endto_end_exact']:.4f}  "
          f"within1(sev)={res['endto_end_within1']:.4f}")
    return res


def per_case(df, label, rank, phase, p_lab, p_rank, p_phase, tag):
    g = pd.DataFrame(dict(case=df["case_id"].values, label=label, rank=rank, phase=phase,
                          plab=p_lab, prank=p_rank, pphase=p_phase))
    rows = []
    for case, gp in g.groupby("case"):
        true_faulty = (gp["label"] == 1).any()
        pred_faulty = gp["plab"].mean() >= 0.5
        fp = gp[gp["plab"] == 1]              # windows predicted faulty
        if true_faulty:
            tr_rank = int(gp.loc[gp["label"] == 1, "rank"].mode().iloc[0])
            tr_ph = int(gp.loc[gp["label"] == 1, "phase"].mode().iloc[0])
        else:
            tr_rank, tr_ph = -1, -1
        pr_rank = int(round(fp["prank"].median())) if len(fp) else -1
        pr_ph = int(fp["pphase"].mode().iloc[0]) if len(fp) else -1
        ok = (pred_faulty == true_faulty) and (
            (not true_faulty) or (pr_rank == tr_rank and pr_ph == tr_ph))
        ok_w1 = (pred_faulty == true_faulty) and (
            (not true_faulty) or (abs(pr_rank - tr_rank) <= 1 and pr_ph == tr_ph))
        rows.append((true_faulty, pred_faulty, ok, ok_w1))
    R = pd.DataFrame(rows, columns=["tf", "pf", "ok", "okw1"])
    res = dict(n_cases=len(R),
               detect_acc=float((R.tf == R.pf).mean()),
               exact=float(R.ok.mean()), within1=float(R.okw1.mean()))
    print(f"   PER-CASE: detect_acc={res['detect_acc']:.4f}  "
          f"exact={res['exact']:.4f}  within1={res['within1']:.4f}  (n={len(R)})")
    return res


def main():
    df = pd.read_parquet(DS / "features.parquet")
    fa, fs = all_features(df.columns), severity_features(df.columns)
    Xall = df[fa].to_numpy(np.float32)
    Xsev = df[fs].to_numpy(np.float32)
    label = df["label"].to_numpy(int)
    rank = df["severity_pct"].map(RANK).fillna(-1).to_numpy(int)
    phase = df["phase"].map(PMAP).fillna(-1).to_numpy(int)
    print(f"windows={len(df)}  all_feats={len(fa)}  severity_feats={len(fs)}")

    results = {}

    print("\n========== CASCADE — StratifiedGroupKFold ==========")
    pl, pr, pp = run_oof(df, Xall, Xsev, label, rank, phase, stratified_group_folds(df))
    results["groupkfold"] = evaluate(df, label, rank, phase, pl, pr, pp, "GroupKFold OOF")
    results["groupkfold"]["per_case"] = per_case(df, label, rank, phase, pl, pr, pp, "GK")
    # save OOF window predictions (labeled)
    out = df[list(LABEL_COLS & set(df.columns))].copy()
    out["pred_state"] = np.where(pl == 1, "faulty", "healthy")
    out["pred_severity_pct"] = [SEV_LEVELS[r] if r >= 0 else np.nan for r in pr]
    out["pred_phase"] = [PHASES[p] if p >= 0 else "" for p in pp]
    out.to_csv(DS / "cascade_oof_predictions.csv", index=False)

    print("\n========== CASCADE — Leave-One-Load-Out (PRIMARY) ==========")
    folds = [(tr, te) for _, tr, te in leave_one_load_out(df)]
    pl, pr, pp = run_oof(df, Xall, Xsev, label, rank, phase, folds)
    # LOLO only holds out TT loads -> healthy files at non-TT loadings are never tested;
    # evaluate ONLY over covered windows (those that appeared in some test fold).
    cov = pl != -1
    print(f"  (covered windows: {int(cov.sum())}/{len(df)}; "
          f"uncovered = healthy files at non-TT loads)")
    df_c = df.iloc[cov].reset_index(drop=True)
    results["lolo"] = evaluate(df_c, label[cov], rank[cov], phase[cov],
                               pl[cov], pr[cov], pp[cov], "LOLO pooled")
    results["lolo"]["per_case"] = per_case(df_c, label[cov], rank[cov], phase[cov],
                                           pl[cov], pr[cov], pp[cov], "LOLO")

    with open(DS / "cascade_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {DS/'cascade_oof_predictions.csv'}, {DS/'cascade_metrics.json'}")


if __name__ == "__main__":
    main()
