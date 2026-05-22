r"""
Phase 3 — Noise robustness. Train each stage on CLEAN features (GroupKFold), then test
on the SAME held-out windows corrupted with additive Gaussian noise at SNR 40/30/20 dB
(features recomputed on the noisy signals). Measures deployment robustness.

Run:  python noise_robustness.py  ->  dataset/noise_robustness.json + .png
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import build_feature_frame
from splits import stratified_group_folds
from feature_sets import all_features, severity_features

DS = Path(r"D:\Naveen") / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV)}
PMAP = {"A": 0, "B": 1, "C": 2}
SNRS = [40, 30, 20]


def add_noise(X, snr_db, rng):
    P = (X ** 2).mean(axis=2, keepdims=True)
    sigma = np.sqrt(P / (10 ** (snr_db / 10.0)))
    return (X + rng.standard_normal(X.shape).astype(np.float32) * sigma).astype(np.float32)


def main():
    X = np.asarray(np.load(DS / "windows.npy"))
    lab = pd.read_csv(DS / "window_labels.csv", low_memory=False)
    clean = pd.read_parquet(DS / "features.parquet")
    fa, fs = all_features(clean.columns), severity_features(clean.columns)
    label = lab["label"].to_numpy(int)
    rank = lab["severity_pct"].map(RANK).fillna(-1).to_numpy(int)
    phase = lab["phase"].map(PMAP).fillna(-1).to_numpy(int)

    # feature frames per condition (clean + noisy recomputed)
    rng = np.random.default_rng(0)
    feats = {"clean": clean}
    for snr in SNRS:
        print(f"building noisy features @ SNR={snr} dB ...")
        feats[snr] = build_feature_frame(add_noise(X, snr, rng), lab)
    Xall = {k: v[fa].to_numpy(np.float32) for k, v in feats.items()}
    Xsev = {k: v[fs].to_numpy(np.float32) for k, v in feats.items()}

    conds = ["clean"] + SNRS
    oof_lab = {c: np.full(len(lab), -1, int) for c in conds}
    oof_rank = {c: np.full(len(lab), -1, int) for c in conds}
    oof_phase = {c: np.full(len(lab), -1, int) for c in conds}

    Xa_clean, Xs_clean = Xall["clean"], Xsev["clean"]
    for tr, te in stratified_group_folds(clean):
        f = label[tr] == 1
        npos, nneg = int(f.sum()), int((~f).sum())
        s1 = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, tree_method="hist",
                           scale_pos_weight=nneg / max(npos, 1), random_state=42).fit(Xa_clean[tr], label[tr])
        s2 = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                          random_state=42).fit(Xs_clean[tr][f], rank[tr][f])
        s3 = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, num_class=3, objective="multi:softprob",
                           eval_metric="mlogloss", n_jobs=-1, tree_method="hist",
                           random_state=42).fit(Xa_clean[tr][f], phase[tr][f])
        for c in conds:
            oof_lab[c][te] = s1.predict(Xall[c][te])
            oof_rank[c][te] = np.clip(np.round(s2.predict(Xsev[c][te])), 0, 6).astype(int)
            oof_phase[c][te] = s3.predict(Xall[c][te])

    res = {}
    truef = label == 1
    print("\nSNR(dB) | Stage1 detect F1 | Stage2 within-1 | Stage3 phase acc")
    for c in conds:
        det_f1 = f1_score(label, oof_lab[c], average="macro")
        sev_w1 = float((np.abs(oof_rank[c][truef] - rank[truef]) <= 1).mean())
        ph_acc = float((oof_phase[c][truef] == phase[truef]).mean())
        res[str(c)] = dict(detect_f1=float(det_f1), severity_within1=sev_w1, phase_acc=ph_acc)
        print(f"  {str(c):>5} |      {det_f1:.4f}     |     {sev_w1:.4f}    |    {ph_acc:.4f}")

    with open(DS / "noise_robustness.json", "w") as f:
        json.dump(res, f, indent=2)
    xs = ["clean"] + [str(s) for s in SNRS]
    xi = [60] + SNRS  # treat clean as 60 dB for plotting
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lbl in [("detect_f1", "Stage1 detect F1"), ("severity_within1", "Stage2 sev within-1"),
                     ("phase_acc", "Stage3 phase acc")]:
        ax.plot(xi, [res[str(c)][key] for c in conds], "o-", label=lbl)
    ax.set_xlabel("SNR (dB)  [clean plotted at 60]"); ax.set_ylabel("metric")
    ax.set_title("Noise robustness (train clean, test noisy)")
    ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(DS / "noise_robustness.png", dpi=130)
    print(f"\nSaved {DS/'noise_robustness.json'}, {DS/'noise_robustness.png'}")


if __name__ == "__main__":
    main()
