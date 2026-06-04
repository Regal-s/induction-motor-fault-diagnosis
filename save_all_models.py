r"""
Persist trained models for the benchmark to disk, so the project's results can
be reproduced (or just inferred against) on a different system without re-running
the multi-hour benchmark.

Strategy:
  * FAST models (LogReg, SVM-RBF, KNN, MLP, RandomForest, XGBoost, NG-RC, ESN,
    iTransformer) are trained once on the full simulation corpus per task
    {detection, severity, phase} with the same feature sets and hyperparameters
    used in benchmark_models.py, then pickled (joblib for sklearn / NumPy state,
    JSON for XGBoost, torch state_dict for PyTorch).
  * SLOW models (xLSTM, ROCKET full-data, TimesNet, PatchTST at full data,
    TTM/Chronos-Bolt full encoder pass, Mamba) are NOT retrained here but their
    hyperparameters are captured into trained_models/__configs__/ so a future
    system can reconstruct them via reservoir_models.py / modern_models.py /
    xlstm_model.py.
  * TabPFN-2.5 does not have parameters to save -- the model performs in-context
    inference at predict-time; the recipe (sample cap + ensemble size) is
    captured in the config.

Outputs (per task):
  trained_models/<Model>/<task>.{joblib | json | pt | npz}
  trained_models/<Model>/<task>.meta.json   <- features used, training time,
                                               hyperparameters, fit metrics

Run: python save_all_models.py            (all FAST models)
     python save_all_models.py --quick    (smaller cap; ~half time)
     python save_all_models.py --models XGBoost,RandomForest
"""
from __future__ import annotations
import sys, json, time, argparse, warnings, joblib
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DS = ROOT / "dataset"
OUT = ROOT / "trained_models"
OUT.mkdir(exist_ok=True)

from splits import TT_LOADS   # noqa: E402
from feature_sets import all_features, severity_features   # noqa: E402

SEV_LEVELS = [0.3, 0.5, 1, 2, 3, 4, 5]
SEV_RANK = {s: i for i, s in enumerate(SEV_LEVELS)}
PHASE_RANK = {"A": 0, "B": 1, "C": 2}
HELPER_COLS = {"_sevrank", "_phrank"}

TASK_CFG = {
    "detection": dict(faulty_only=False, feats="all", n_classes=2, ycol="label"),
    "severity":  dict(faulty_only=True,  feats="severity", n_classes=7, ycol="_sevrank"),
    "phase":     dict(faulty_only=True,  feats="all", n_classes=3, ycol="_phrank"),
}


def _feature_columns(df, task_cfg):
    cols = (all_features(df.columns) if task_cfg["feats"] == "all"
            else severity_features(df.columns, load_aware=True))
    return [c for c in cols if c not in HELPER_COLS]


def _save_meta(out_dir: Path, task: str, meta: dict):
    (out_dir / f"{task}.meta.json").write_text(json.dumps(meta, indent=2))


def _class_weight(y, n_classes):
    nb = np.bincount(y, minlength=n_classes).astype(float)
    return dict(enumerate(nb.sum() / (np.maximum(nb, 1) * n_classes)))


# ============================================================================
# Per-model fit-and-save functions
# ============================================================================
def fit_save_sklearn(name, df, X_cols, task, task_cfg, out_dir, quick):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if name == "LogReg":
        from sklearn.linear_model import LogisticRegression
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    elif name == "SVM-RBF":
        from sklearn.svm import SVC
        mdl = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                                                  probability=True))
    elif name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        mdl = make_pipeline(StandardScaler(), KNeighborsClassifier(7))
    elif name == "MLP":
        from sklearn.neural_network import MLPClassifier
        mdl = make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=300, random_state=0))
    elif name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)
    else:
        raise ValueError(name)

    X = df[X_cols].to_numpy(np.float32)
    y = df[task_cfg["ycol"]].to_numpy(int)

    # Optional cap for slow models
    cap = {"SVM-RBF": 6000, "KNN": 6000, "MLP": 8000}.get(name)
    if cap and quick:
        cap = min(cap, 4000)
    if cap and len(X) > cap:
        rng = np.random.RandomState(0)
        idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            k = max(1, int(round(cap * len(ci) / len(X))))
            idx.append(rng.choice(ci, min(k, len(ci)), replace=False))
        idx = np.concatenate(idx)
        Xf, yf = X[idx], y[idx]
    else:
        Xf, yf = X, y

    t0 = time.time()
    mdl.fit(Xf, yf)
    train_t = time.time() - t0
    pred = mdl.predict(Xf[:min(1000, len(Xf))])
    fit_acc = float((pred == yf[:min(1000, len(yf))]).mean())

    joblib.dump(mdl, out_dir / f"{task}.joblib")
    _save_meta(out_dir, task, {
        "model": name, "task": task, "n_train": int(len(Xf)),
        "n_features": int(Xf.shape[1]), "features": X_cols,
        "train_seconds": round(train_t, 2), "fit_acc_first_1000": fit_acc,
        "format": "joblib (sklearn Pipeline)", "load_with": "joblib.load",
    })


def fit_save_xgboost(df, X_cols, task, task_cfg, out_dir):
    from xgboost import XGBClassifier
    X = df[X_cols].to_numpy(np.float32)
    y = df[task_cfg["ycol"]].to_numpy(int)
    nb = np.bincount(y)
    spw = (nb[0] / max(nb[1], 1)) if len(nb) == 2 else 1.0
    mdl = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                        eval_metric="mlogloss", random_state=0,
                        scale_pos_weight=spw if len(nb) == 2 else 1.0)
    t0 = time.time(); mdl.fit(X, y); train_t = time.time() - t0
    mdl.save_model(str(out_dir / f"{task}.json"))
    _save_meta(out_dir, task, {
        "model": "XGBoost", "task": task, "n_train": int(len(X)),
        "n_features": int(X.shape[1]), "features": X_cols,
        "train_seconds": round(train_t, 2),
        "format": "XGBoost native JSON",
        "load_with": "from xgboost import XGBClassifier; m=XGBClassifier(); m.load_model('<file>.json')",
    })


def fit_save_ngrc(df, Xdec, task, task_cfg, out_dir):
    import reservoir_models as rc
    from sklearn.linear_model import RidgeClassifierCV
    Xw = Xdec[df.index.to_numpy()] if "_orig_idx" in df.columns else Xdec
    y = df[task_cfg["ycol"]].to_numpy(int)
    t0 = time.time()
    Ftr = np.stack([rc._ngrc_features(x, 4, 20) for x in Xw])
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), class_weight="balanced")
    clf.fit(Ftr, y)
    train_t = time.time() - t0
    joblib.dump(clf, out_dir / f"{task}.ridge.joblib")
    _save_meta(out_dir, task, {
        "model": "NG-RC", "task": task, "n_train": int(len(Xw)),
        "n_features": int(Ftr.shape[1]),
        "ngrc_params": {"n_taps": 4, "n_samples": 20},
        "train_seconds": round(train_t, 2),
        "format": "ridge classifier (joblib)",
        "load_with": "joblib.load + reservoir_models._ngrc_features for new samples",
    })


def fit_save_esn(df, Xdec, task, task_cfg, out_dir):
    import reservoir_models as rc
    from sklearn.linear_model import RidgeClassifierCV
    Xw = Xdec[df.index.to_numpy()] if "_orig_idx" in df.columns else Xdec
    y = df[task_cfg["ycol"]].to_numpy(int)
    feat = rc.ESNFeaturizer(n_reservoir=200, seed=0)
    t0 = time.time()
    Ftr = feat.transform(Xw)
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), class_weight="balanced")
    clf.fit(Ftr, y)
    train_t = time.time() - t0
    np.savez_compressed(out_dir / f"{task}.esn.npz",
                        W_in=feat.W_in, W=feat.W, leak=feat.leak)
    joblib.dump(clf, out_dir / f"{task}.ridge.joblib")
    _save_meta(out_dir, task, {
        "model": "ESN", "task": task, "n_train": int(len(Xw)),
        "n_features": int(Ftr.shape[1]),
        "esn_params": {"n_reservoir": 200, "spectral_radius": 0.9, "leak": 0.3},
        "train_seconds": round(train_t, 2),
        "format": "reservoir weights (npz) + ridge classifier (joblib)",
        "load_with": "see reservoir_models.ESNFeaturizer + joblib.load",
    })


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None)
    args = ap.parse_args()

    print("loading features.parquet ...")
    df = pd.read_parquet(DS / "features.parquet")
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    df["_orig_idx"] = np.arange(len(df))
    faulty_mask = df["label"].to_numpy() == 1

    # Decimated windows (lazy load only if needed)
    Xdec = None
    if any(m in (args.models or "ESN,NG-RC") for m in ("ESN", "NG-RC")):
        print("loading + decimating windows.npy ...")
        import xlstm_model as xm
        Xw = np.asarray(np.load(DS / "windows.npy"), np.float32)
        Xdec = xm.decimate(Xw, 8)
        del Xw

    sklearn_models = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest"]
    other_models = ["XGBoost", "NG-RC", "ESN"]
    roster = sklearn_models + other_models
    if args.models:
        roster = [m for m in roster if m in args.models.split(",")]

    for name in roster:
        out_dir = OUT / name; out_dir.mkdir(exist_ok=True)
        print(f"\n=== {name} ===")
        for task, cfg in TASK_CFG.items():
            if cfg["faulty_only"]:
                sub = df[faulty_mask].reset_index(drop=True)
                Xd = Xdec[faulty_mask] if Xdec is not None else None
            else:
                sub = df
                Xd = Xdec
            X_cols = _feature_columns(sub, cfg)
            if (out_dir / f"{task}.meta.json").exists():
                print(f"  [{task}] already saved, skip")
                continue
            t0 = time.time()
            try:
                if name in sklearn_models:
                    fit_save_sklearn(name, sub, X_cols, task, cfg, out_dir, args.quick)
                elif name == "XGBoost":
                    fit_save_xgboost(sub, X_cols, task, cfg, out_dir)
                elif name == "NG-RC":
                    fit_save_ngrc(sub, Xd, task, cfg, out_dir)
                elif name == "ESN":
                    fit_save_esn(sub, Xd, task, cfg, out_dir)
                print(f"  [{task}] saved ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [{task}] FAILED: {e}")

    # Capture configs for the deep / foundation / slow models so a future system
    # can reproduce them.
    cfg_dir = OUT / "__configs__"; cfg_dir.mkdir(exist_ok=True)
    SLOW = {
        "xLSTM": dict(source="xlstm_model.py", decimation=8, epochs=16,
                      class_balanced=True, cap_per_fold=9000),
        "ROCKET": dict(source="reservoir_models.py:train_predict_rocket",
                       n_kernels=1500, kernel_lengths=[7, 9, 11], cap_per_fold=1500),
        "PatchTST": dict(source="modern_models.py:train_predict_patchtst",
                         patch=24, stride=12, d_model=48, n_heads=4, n_layers=2,
                         d_ff=96, dropout=0.1, epochs=6, batch=96, cap_per_fold=1500),
        "iTransformer": dict(source="modern_models.py:train_predict_itransformer",
                             d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                             epochs=8, batch=64, cap_per_fold=2500),
        "TimesNet": dict(source="modern_models.py:train_predict_timesnet",
                         n_blocks=2, hidden=32, top_k=2, epochs=5, batch=32, cap_per_fold=1500),
        "Mamba": dict(source="modern_models.py:train_predict_mamba",
                      d_model=32, d_state=4, n_blocks=1, stride=5, epochs=6, batch=48,
                      cpu_prohibitive=True),
        "TTM": dict(source="modern_models.py:train_predict_foundation('TTM')",
                    hf_model="ibm-granite/granite-timeseries-ttm-r2",
                    head="RidgeClassifierCV alphas=(0.1, 1.0, 10.0)", cap_per_fold=4000),
        "Chronos-Bolt": dict(source="modern_models.py:train_predict_foundation('Chronos-Bolt')",
                             hf_model="amazon/chronos-bolt-small",
                             head="RidgeClassifierCV alphas=(0.1, 1.0, 10.0)", cap_per_fold=4000),
        "TabPFN-2.5": dict(source="benchmark_models.py:make_tabpfn",
                           in_context_inference=True, context_cap=2000, n_estimators=2,
                           note="no trained parameters to save -- runs in-context at predict time"),
    }
    (cfg_dir / "slow_models_reproducibility.json").write_text(json.dumps(SLOW, indent=2))
    print(f"\nwrote {cfg_dir/'slow_models_reproducibility.json'}")
    print(f"\nDone. Trained-model artefacts in {OUT}/")


if __name__ == "__main__":
    main()
