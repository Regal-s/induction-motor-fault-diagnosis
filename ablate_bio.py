r"""
Ablation: do the biomedical anomaly features (features_bio.py) improve diagnosis on top of the
72 physics features? Compares PHYSICS vs PHYSICS+BIOMEDICAL for detection / severity / phase under
identical leakage-safe splits (StratifiedGroupKFold + Leave-One-Load-Out), with XGBoost (deployed).
Also reports the top biomedical features by gain importance (evidence they are used, not noise).

Run:  python ablate_bio.py [--folds 10]
Outputs: dataset/ablate_bio_results.json, dataset/ablate_bio_table.md
"""
from __future__ import annotations
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, cohen_kappa_score
from xgboost import XGBClassifier, XGBRegressor
from splits import stratified_group_folds, leave_one_load_out
from feature_sets import all_features, severity_features

SEV_RANK = {0.3: 0, 0.5: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
PHASE_RANK = {"A": 0, "B": 1, "C": 2}


def score(task, yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    if task == "severity":
        yr = np.rint(yp).astype(int).clip(0, 6)
        return {"within1": float((np.abs(yt - yr) <= 1).mean()), "exact": float((yt == yr).mean()),
                "mae": float(mean_absolute_error(yt, yr)),
                "qwk": float(cohen_kappa_score(yt, yr, weights="quadratic", labels=list(range(7))))}
    return {"macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "acc": float(accuracy_score(yt, yp))}


def prim(task, yt, yp):
    return score(task, yt, yp)["within1" if task == "severity" else "macro_f1"]


def model(task):
    if task == "severity":
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, tree_method="hist", random_state=0)
    nc = 2 if task == "detection" else 3
    kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, n_jobs=-1, tree_method="hist", random_state=0)
    return (XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw) if nc == 2
            else XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", **kw))


def run(task, sub, cols, ycol, n_folds):
    X = sub[cols].to_numpy(np.float32); y = sub[ycol].to_numpy(int)
    out = {}
    oof = np.full(len(sub), np.nan)
    for tr, te in stratified_group_folds(sub, y_col=ycol, n_splits=n_folds):
        oof[te] = model(task).fit(X[tr], y[tr]).predict(X[te])
    m = ~np.isnan(oof)
    out["groupkfold"] = score(task, y[m], oof[m])
    yt_all, yp_all = [], []
    for L, tr, te in leave_one_load_out(sub):
        if not len(te):
            continue
        yp_all.append(model(task).fit(X[tr], y[tr]).predict(X[te])); yt_all.append(y[te])
    out["lolo"] = score(task, np.concatenate(yt_all), np.concatenate(yp_all))
    return out


def top_bio_importance(task, sub, phys, bio, ycol, k=10):
    X = sub[phys + bio].to_numpy(np.float32); y = sub[ycol].to_numpy(int)
    tr, te = next(stratified_group_folds(sub, y_col=ycol, n_splits=5))
    m = model(task).fit(X[tr], y[tr])
    imp = m.feature_importances_
    names = phys + bio
    bio_idx = [(names[i], float(imp[i])) for i in range(len(names)) if names[i] in bio]
    bio_idx.sort(key=lambda t: -t[1])
    rank_all = np.argsort(imp)[::-1]
    bio_in_top20 = [names[i] for i in rank_all[:20] if names[i] in bio]
    return bio_idx[:k], bio_in_top20


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=10); a = ap.parse_args()
    phys_df = pd.read_parquet(DS / "features.parquet")
    bio_df = pd.read_parquet(DS / "features_bio.parquet")
    if "window_id" in bio_df.columns and "window_id" in phys_df.columns:
        df = phys_df.merge(bio_df, on="window_id", how="inner")
    else:
        df = pd.concat([phys_df.reset_index(drop=True), bio_df.reset_index(drop=True)], axis=1)
    bio_cols = [c for c in bio_df.columns if c != "window_id"]
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    print(f"merged {len(df)} rows | physics + {len(bio_cols)} biomedical features")

    results = {}
    for task in ["detection", "severity", "phase"]:
        fonly = task != "detection"
        sub = df[df["label"] == 1].reset_index(drop=True) if fonly else df
        ycol = {"detection": "label", "severity": "_sevrank", "phase": "_phrank"}[task]
        if task == "severity":
            phys = [c for c in severity_features(sub.columns, load_aware=True) if c not in ("_sevrank", "_phrank")]
        else:
            phys = [c for c in all_features(sub.columns) if c not in bio_cols + ["_sevrank", "_phrank"]
                    and pd.api.types.is_numeric_dtype(sub[c])]
        bio = [c for c in bio_cols if pd.api.types.is_numeric_dtype(sub[c])]
        res_phys = run(task, sub, phys, ycol, a.folds)
        res_both = run(task, sub, phys + bio, ycol, a.folds)
        topbio, intop20 = top_bio_importance(task, sub, phys, bio, ycol)
        results[task] = {"physics": res_phys, "physics+bio": res_both,
                         "n_phys": len(phys), "n_bio": len(bio),
                         "top_bio_importance": topbio, "bio_in_top20": intop20}
        pk = "within1" if task == "severity" else "macro_f1"
        print(f"\n=== {task} (metric={pk}) ===")
        for proto in ["groupkfold", "lolo"]:
            p = res_phys[proto][pk]; b = res_both[proto][pk]
            print(f"  {proto:11s} physics={p:.3f}  physics+bio={b:.3f}  delta={b-p:+.3f}")
        print(f"  top bio feats: {[t[0] for t in topbio[:5]]}")
        (DS / "ablate_bio_results.json").write_text(json.dumps(results, indent=2))

    # markdown table
    L = ["# Biomedical-feature ablation (XGBoost, leakage-safe)", "",
         "Physics (existing) vs physics + biomedical anomaly features, identical splits.", ""]
    for task in results:
        pk = "within1" if task == "severity" else "macro-F1"
        r = results[task]
        L += [f"## {task.capitalize()} ({pk})", "",
              "| Protocol | Physics | Physics+Bio | Δ |", "|---|---|---|---|"]
        key = "within1" if task == "severity" else "macro_f1"
        for proto in ["groupkfold", "lolo"]:
            p = r["physics"][proto][key]; b = r["physics+bio"][proto][key]
            L.append(f"| {proto} | {p:.3f} | {b:.3f} | {b-p:+.3f} |")
        L += ["", f"Top biomedical features by gain: {', '.join(t[0] for t in r['top_bio_importance'][:6])}",
              f"Biomedical features in overall top-20: {', '.join(r['bio_in_top20']) or 'none'}", ""]
    (DS / "ablate_bio_table.md").write_text("\n".join(L), encoding="utf-8")
    print("\nsaved dataset/ablate_bio_results.json + ablate_bio_table.md")


if __name__ == "__main__":
    main()
