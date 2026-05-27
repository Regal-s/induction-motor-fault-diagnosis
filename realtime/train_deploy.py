r"""
Train the deployable cascade on the REAL machine data (ibarram) and save a portable bundle for
the Jetson receiver. Leakage-safe split: train on repetitions 01-04, hold out repetition 05 as the
"live stream" test set. Models = XGBoost (fast, exportable, no license token) for detection /
severity (4-level) / faulted-phase. The sim-trained models are NOT used (they need PADA on real data).

Saves to realtime/deploy/:
  detect.json, severity.json, phase.json   (XGBoost boosters)
  meta.json                                (feature columns, label maps, fs/f0/W/stride, onset threshold)
Run:  python realtime/train_deploy.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from feature_sets import all_features, severity_features
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
import gpu

EXP = ROOT / "experimental" / "ibarram"
OUT = ROOT / "realtime" / "deploy"; OUT.mkdir(parents=True, exist_ok=True)
SEV_RANK = {10.0: 0, 20.0: 1, 30.0: 2, 40.0: 3}; SEV_LEVELS = [10, 20, 30, 40]
PHASE_RANK = {"A": 0, "B": 1, "C": 2}
FS, F0, W, STRIDE = 1000.0, 60.0, 500, 250
TEST_REP = "05"


def xgb(nc):
    kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, n_jobs=-1, tree_method="hist", device=gpu.xgb_device(), random_state=0)
    if nc == 2:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
    return XGBClassifier(objective="multi:softprob", num_class=nc, eval_metric="mlogloss", **kw)


def main():
    df = pd.read_parquet(EXP / "features.parquet").reset_index(drop=True)
    df["_rep"] = df["case_id"].str.extract(r"Repetition(\d+)")[0]
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    tr_mask = df["_rep"] != TEST_REP

    # Exclude (a) metadata cols not produced by build_feature_frame, and (b) the COMPENSATED
    # negative-sequence features I2c_* -- they need per-recording context (a healthy baseline) that a
    # live single-window stream lacks, so they cannot be reproduced online. The plain |I2|/|I1| features
    # carry the same inter-turn signal and are fully self-contained. (Online I2c would need on-target
    # baseline calibration -- see JETSON_DEPLOYMENT.md.)
    META_COLS = ("_rep", "_sevrank", "_phrank", "dataset", "fs", "f0")
    drop = lambda c: c in META_COLS or c.startswith("I2c")
    allf = [c for c in all_features(df.columns) if not drop(c) and pd.api.types.is_numeric_dtype(df[c])]
    sevf = [c for c in severity_features(df.columns, load_aware=False)
            if not drop(c) and pd.api.types.is_numeric_dtype(df[c])]

    meta = {"fs": FS, "f0": F0, "W": W, "stride": STRIDE, "test_rep": TEST_REP,
            "features_all": allf, "features_severity": sevf,
            "sev_levels": SEV_LEVELS, "phase_labels": ["A", "B", "C"], "metrics": {}}

    # ---- detection ----
    Xtr = df[tr_mask][allf].to_numpy(np.float32); ytr = df[tr_mask]["label"].to_numpy(int)
    m = xgb(2).fit(Xtr, ytr); m.save_model(str(OUT / "detect.json"))
    te = df[~tr_mask]; p = m.predict(te[allf].to_numpy(np.float32))
    meta["metrics"]["detection"] = {"f1": float(f1_score(te["label"], p, average="macro")),
                                    "acc": float(accuracy_score(te["label"], p))}

    # ---- severity (faulty only) ----
    f_tr = df[tr_mask & (df.label == 1)]; f_te = df[~tr_mask & (df.label == 1)]
    m = xgb(4).fit(f_tr[sevf].to_numpy(np.float32), f_tr["_sevrank"].to_numpy(int)); m.save_model(str(OUT / "severity.json"))
    p = m.predict(f_te[sevf].to_numpy(np.float32)); yt = f_te["_sevrank"].to_numpy(int)
    meta["metrics"]["severity"] = {"within1": float((np.abs(yt - p) <= 1).mean()),
                                   "exact": float((yt == p).mean()),
                                   "f1": float(f1_score(yt, p, average="macro"))}

    # ---- phase (faulty only) ----
    m = xgb(3).fit(f_tr[allf].to_numpy(np.float32), f_tr["_phrank"].to_numpy(int)); m.save_model(str(OUT / "phase.json"))
    p = m.predict(f_te[allf].to_numpy(np.float32)); yt = f_te["_phrank"].to_numpy(int)
    meta["metrics"]["phase"] = {"f1": float(f1_score(yt, p, average="macro")), "acc": float(accuracy_score(yt, p))}

    # ---- Stage-0 onset threshold from healthy training windows ----
    hcol = "I2_I1_ratio"
    if hcol in df.columns:
        h = df[tr_mask & (df.label == 0)][hcol].to_numpy(float)
        meta["onset"] = {"feature": hcol, "threshold": float(h.mean() + 6 * h.std()),
                         "healthy_mean": float(h.mean()), "healthy_std": float(h.std())}

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved realtime/deploy/{detect,severity,phase}.json + meta.json")
    print("held-out rep-05 metrics:", json.dumps(meta["metrics"], indent=2))
    if "onset" in meta:
        print("onset threshold |I2|/|I1| >", round(meta["onset"]["threshold"], 4))


if __name__ == "__main__":
    main()
