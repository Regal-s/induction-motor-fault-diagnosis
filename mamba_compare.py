r"""
Mamba-2 comparison (CPU, reduced scale) — mirrors xlstm_improve_proof.py for the compact Mamba-2
selective-SSM (mamba_model.py), on the SAME leakage-safe split, so the deep sequence models
(xLSTM vs Mamba-2) are directly comparable, both against the feature/foundation baselines.

Conditions: baseline (raw 3-phase, small net) and upgraded (physics channels + augmentation +
larger net). Tasks: detection, phase. -> dataset/mamba_compare.json
Run:  python mamba_compare.py
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
from splits import stratified_group_folds
import xlstm_model as xm        # decimate + physics_channels
import mamba_model as mm
import augment as aug

PHASE_RANK = {"A": 0, "B": 1, "C": 2}; CAP = 5000


def cw(y, nc):
    nb = np.bincount(y, minlength=nc).astype(float); return nb.sum() / (np.maximum(nb, 1) * nc)


def subcap(idx, y, cap, seed=0):
    if len(idx) <= cap: return idx
    rng = np.random.RandomState(seed); parts = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]; k = max(1, int(round(cap * len(ci) / len(idx))))
        parts.append(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.concatenate(parts)


def evaluate(task):
    df = pd.read_parquet(DS / "features.parquet").reset_index(drop=True)
    Xw = np.asarray(np.load(DS / "windows.npy", mmap_mode="r"), np.float32)
    if task == "detection":
        y = df["label"].to_numpy(int); nc = 2; col = "label"; mask = np.ones(len(df), bool)
    else:
        mask = df["label"].to_numpy() == 1
        y = df[mask]["phase"].map(PHASE_RANK).to_numpy(int); nc = 3; col = "phase"
    sub = df[mask].reset_index(drop=True); Xw = Xw[mask]
    tr, te = next(stratified_group_folds(sub, y_col=col, n_splits=5))
    trb = subcap(tr, y, CAP); out = {}

    Xd = xm.decimate(Xw, 8)
    t0 = time.time()
    pb = mm.train_predict(Xd[trb], y[trb], Xd[te], nc, epochs=12, d_model=64, n_blocks=2,
                          class_weight=cw(y[trb], nc), device="cpu")
    out["baseline"] = {"f1": float(f1_score(y[te], pb, average="macro")),
                       "acc": float(accuracy_score(y[te], pb)), "sec": round(time.time() - t0)}

    Xtr_raw = Xw[trb]
    Xaug, src = aug.augment_batch(Xtr_raw, samples_per_cycle=500, n_aug=1, seed=0)
    Xtr_all = np.concatenate([Xtr_raw, Xaug], 0); ytr_all = np.concatenate([y[trb], y[trb][src]], 0)
    Xtr_ch = xm.decimate(xm.physics_channels(Xtr_all), 8); Xte_ch = xm.decimate(xm.physics_channels(Xw[te]), 8)
    t0 = time.time()
    pu = mm.train_predict(Xtr_ch, ytr_all, Xte_ch, nc, epochs=16, d_model=96, n_blocks=3,
                          class_weight=cw(ytr_all, nc), device="cpu")
    out["upgraded"] = {"f1": float(f1_score(y[te], pu, average="macro")),
                       "acc": float(accuracy_score(y[te], pu)), "sec": round(time.time() - t0)}
    out["delta_f1"] = round(out["upgraded"]["f1"] - out["baseline"]["f1"], 4)
    print(f"[mamba2 {task}] baseline F1={out['baseline']['f1']:.3f}  upgraded F1={out['upgraded']['f1']:.3f}  "
          f"delta={out['delta_f1']:+.3f}")
    return out


def main():
    res = {}
    for task in ["detection", "phase"]:
        print(f"=== mamba2 {task} ===")
        res[task] = evaluate(task)
        (DS / "mamba_compare.json").write_text(json.dumps(res, indent=2))
    print("\nsaved dataset/mamba_compare.json")


if __name__ == "__main__":
    main()
