r"""
Improve faulted-phase F1 on the REAL (ibarram) hardware data.

Hypothesis: real-data phase failure is caused by (i) per-recording drift of the
ABSOLUTE negative-sequence angle and (ii) per-window noise. Fixes tested:
  A. per-recording majority vote (aggregate windows -> one decision per recording)
  B. reference-invariant features: relative angle (Delta-angle, already present) +
     reference-free per-phase asymmetry ("which phase deviates most" in THD/3rd-harm/
     RMS/...), DROPPING the absolute angle features that drift across recordings.

Compares, for XGBoost and RandomForest, on the phase task (faulty windows only):
  feature set in {baseline(all), invariant} x scoring in {per-window, per-recording}
under StratifiedGroupKFold (by recording) and Leave-One-Repetition-Out (LORO).

Run:  python phase_hw_improve.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
EXP = ROOT / "experimental" / "ibarram"
from splits import stratified_group_folds                 # noqa: E402
from feature_sets import all_features                     # noqa: E402

PHASE_RANK = {"A": 0, "B": 1, "C": 2}
ASYM_METRICS = ["rms", "thd", "h3ratio", "std", "crest", "skew", "kurt", "peak"]
ABS_ANGLE = {"I1_angle_deg", "I2_angle_deg", "I2_angle_sin", "I2_angle_cos"}  # drift across recordings


def engineer(df):
    """Reference-free per-phase asymmetry features: per-phase deviation from the 3-phase
    mean, plus which-phase-is-max / -min indicators (these directly encode the faulted phase
    without any angle reference)."""
    new = {}
    for m in ASYM_METRICS:
        cols = [f"pha_{m}", f"phb_{m}", f"phc_{m}"]
        if not all(c in df.columns for c in cols):
            continue
        M = df[cols].to_numpy(float)
        mean = M.mean(1, keepdims=True)
        dev = M - mean
        for k, ph in enumerate("abc"):
            new[f"dev_{ph}_{m}"] = dev[:, k]
        amax = M.argmax(1); amin = M.argmin(1)
        for k in range(3):                       # one-hot of argmax/argmin phase
            new[f"ismax_{'abc'[k]}_{m}"] = (amax == k).astype(float)
            new[f"ismin_{'abc'[k]}_{m}"] = (amin == k).astype(float)
        new[f"spread_{m}"] = M.max(1) - M.min(1)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1), list(new)


def invariant_cols(df, engineered):
    keep = []
    for c in all_features(df.columns):
        if c in ABS_ANGLE:                       # drop drifting absolute angle
            continue
        if c in ("_sevrank", "_phrank", "_rep", "dataset"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        keep.append(c)
    return sorted(set(keep + engineered))


def leave_one_rep_out(df):
    rep = df["_rep"].to_numpy()
    for r in sorted(np.unique(rep)):
        te = np.where(rep == r)[0]; tr = np.where(rep != r)[0]
        yield r, tr, te


def model(name, y):
    if name == "XGBoost":
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                             eval_metric="mlogloss", random_state=0)
    return RandomForestClassifier(400, class_weight="balanced", n_jobs=-1, random_state=0)


def per_record_vote(case_ids, y_true, y_pred):
    """Majority-vote predictions per recording; return record-level (true, pred)."""
    dfp = pd.DataFrame({"c": case_ids, "t": y_true, "p": y_pred})
    rt, rp = [], []
    for c, g in dfp.groupby("c"):
        rt.append(int(g["t"].mode().iloc[0]))
        rp.append(int(g["p"].mode().iloc[0]))
    return np.array(rt), np.array(rp)


def evaluate(df, cols, name):
    X = df[cols].to_numpy(np.float32); y = df["_phrank"].to_numpy(int)
    cid = df["case_id"].to_numpy()
    out = {}
    for proto, splititer in [("groupkfold", lambda: stratified_group_folds(df, y_col="_phrank", group_col="case_id")),
                             ("loro", lambda: ((r, tr, te) for r, tr, te in leave_one_rep_out(df)))]:
        oof = np.full(len(df), -1, dtype=int)
        for fold in splititer():
            tr, te = (fold[1], fold[2]) if len(fold) == 3 else fold
            m = model(name, y[tr]).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        mask = oof >= 0
        win_f1 = f1_score(y[mask], oof[mask], average="macro")
        rt, rp = per_record_vote(cid[mask], y[mask], oof[mask])
        rec_f1 = f1_score(rt, rp, average="macro")
        out[proto] = {"window_f1": float(win_f1), "record_f1": float(rec_f1),
                      "window_acc": float(accuracy_score(y[mask], oof[mask])),
                      "record_acc": float(accuracy_score(rt, rp))}
    return out


def main():
    df = pd.read_parquet(EXP / "features.parquet").reset_index(drop=True)
    df["_rep"] = df["case_id"].str.extract(r"Repetition(\d+)")[0].astype(int)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    df = df[df["label"] == 1].reset_index(drop=True)   # faulty only
    df, engineered = engineer(df)
    base_cols = [c for c in all_features(df.columns)
                 if c not in ("_sevrank", "_phrank", "_rep", "dataset")
                 and pd.api.types.is_numeric_dtype(df[c]) and c not in engineered]
    inv_cols = invariant_cols(df, engineered)
    print(f"faulty windows={len(df)}  baseline feats={len(base_cols)}  invariant feats={len(inv_cols)} "
          f"(+{len(engineered)} engineered asymmetry)")

    results = {}
    for name in ["XGBoost", "RandomForest"]:
        results[name] = {"baseline": evaluate(df, base_cols, name),
                         "invariant+asym": evaluate(df, inv_cols, name)}
        for fs in results[name]:
            for proto, m in results[name][fs].items():
                print(f"  {name:13s} {fs:15s} {proto:10s} "
                      f"window_F1={m['window_f1']:.3f}  record_F1={m['record_f1']:.3f}")
    (DS / "phase_hw_improve.json").write_text(json.dumps(results, indent=2))
    print("\nsaved dataset/phase_hw_improve.json")
    # headline: baseline window LORO vs invariant record LORO (XGBoost)
    b = results["XGBoost"]["baseline"]["loro"]["window_f1"]
    g = results["XGBoost"]["invariant+asym"]["loro"]["record_f1"]
    print(f"\nHEADLINE XGBoost LOLO phase F1: baseline window {b:.3f} -> invariant+vote {g:.3f} "
          f"(+{g-b:.3f})")


if __name__ == "__main__":
    main()
