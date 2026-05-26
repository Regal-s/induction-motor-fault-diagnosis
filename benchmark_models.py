r"""
Unified model benchmark — adds TabPFN-2.5 (tabular foundation model) and xLSTM
(deep sequence model) to the project's model roster and compares them, on IDENTICAL
leakage-safe splits, against the previously used models across all three tasks:

  detection (Healthy vs Faulty)  | severity (% shorted turns, 7 ordinal levels) | phase (A/B/C)

Protocols (reused from splits.py so numbers are directly comparable to the stage scripts):
  - StratifiedGroupKFold (in-distribution, 5-fold pooled OOF)
  - Leave-One-Load-Out   (cross-load, PRIMARY; pooled over held-out loads)

Feature-track models run on dataset/features.parquet (72 physics features):
  LogReg, SVM-RBF, KNN, MLP, RandomForest, XGBoost, TabPFN-2.5
Deep (raw-signal) model runs on dataset/windows.npy (decimated 3-phase current):
  xLSTM      (compact CPU xLSTM, see xlstm_model.py)

TabPFN-2.5 needs a one-time license token (env TABPFN_TOKEN, free from
https://ux.priorlabs.ai). If absent, TabPFN is skipped with a clear note and the
rest of the benchmark still runs.

Outputs (every model's results saved):
  dataset/benchmark_results.json   nested model -> stage -> protocol -> metrics
  dataset/benchmark_table.csv      flat tidy table (one row per model/stage/protocol)
  dataset/benchmark_run.log        console log

Run:  python benchmark_models.py            (all available models)
      python benchmark_models.py --quick    (smaller subsamples / fewer epochs)
      python benchmark_models.py --models XGBoost,xLSTM,TabPFN-2.5
"""
from __future__ import annotations
import os, sys, json, time, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
DS = ROOT / "dataset"
from splits import stratified_group_folds, leave_one_load_out, TT_LOADS   # noqa: E402
from feature_sets import all_features, severity_features                  # noqa: E402

from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score)

SEV_LEVELS = [0.3, 0.5, 1, 2, 3, 4, 5]                 # severity -> rank 0..6
SEV_RANK = {s: i for i, s in enumerate(SEV_LEVELS)}
PHASE_RANK = {"A": 0, "B": 1, "C": 2}
HELPER_COLS = {"_sevrank", "_phrank"}                   # derived label cols — NEVER use as features

# tasks: (name, faulty_only, feature_set_fn, n_classes)
TASKS = {
    "detection": dict(faulty_only=False, feats="all", n_classes=2),
    "severity":  dict(faulty_only=True,  feats="severity", n_classes=7),
    "phase":     dict(faulty_only=True,  feats="all", n_classes=3),
}


# --------------------------------------------------------------------------- metrics
def score(task, y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    m = {"acc": float(accuracy_score(y_true, y_pred)),
         "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    if task == "severity":
        m["mae"] = float(mean_absolute_error(y_true, y_pred))
        m["within1"] = float((np.abs(y_true - y_pred) <= 1).mean())
        m["exact"] = m["acc"]
        try:
            m["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except Exception:
            m["qwk"] = float("nan")
    return m


# --------------------------------------------------------------------------- feature models
def make_feature_model(name, y_tr, quick):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    if name == "SVM-RBF":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced"))
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(7))
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=300, random_state=0))
    if name == "RandomForest":
        return RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)
    if name == "XGBoost":
        nb = (np.bincount(y_tr))
        spw = (nb[0] / max(nb[1], 1)) if len(nb) == 2 else 1.0
        return XGBClassifier(n_estimators=200 if quick else 400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                             eval_metric="mlogloss", random_state=0,
                             scale_pos_weight=spw if len(nb) == 2 else 1.0)
    raise ValueError(name)


def make_tabpfn(quick):
    # CPU is transductive-forward bound: cost ~ context^2 * n_estimators * query.
    # context capped at 2000 (see SUBSAMPLE) and a small ensemble keep it tractable.
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device="cpu", ignore_pretraining_limits=True,
                            n_estimators=1 if quick else 2, random_state=0)


# subsample cap for the slow / context-based models (per training fold)
SUBSAMPLE = {"SVM-RBF": 6000, "KNN": 6000, "MLP": 8000, "TabPFN-2.5": 2000}


def run_feature_model(name, df, Xall, task_cfg, task, quick):
    """Return dict protocol -> metrics for one feature-track model on one task."""
    feats_cols = (all_features(df.columns) if task_cfg["feats"] == "all"
                  else severity_features(df.columns, load_aware=True))
    feats_cols = [c for c in feats_cols if c not in HELPER_COLS]   # guard against label leakage
    Xt = df[feats_cols].to_numpy(np.float32)
    y = _task_labels(df, task)
    out = {}

    # ---- StratifiedGroupKFold (pooled OOF) ----
    oof = np.full(len(df), -1, dtype=int)
    for tr, te in stratified_group_folds(df, y_col=_ycol(task)):
        tr_use = _subsample(name, tr, y, quick)
        mdl = _fit(name, Xt[tr_use], y[tr_use], quick)
        oof[te] = _predict(mdl, Xt[te])
    mask = oof >= 0
    out["groupkfold"] = score(task, y[mask], oof[mask])

    # ---- Leave-One-Load-Out (pooled over held-out loads) ----
    yt_all, yp_all = [], []
    for L, tr, te in leave_one_load_out(df):
        if not len(te):
            continue
        tr_use = _subsample(name, tr, y, quick)
        mdl = _fit(name, Xt[tr_use], y[tr_use], quick)
        yp = _predict(mdl, Xt[te])
        yt_all.append(y[te]); yp_all.append(yp)
    if yt_all:
        out["lolo"] = score(task, np.concatenate(yt_all), np.concatenate(yp_all))
    return out


def _fit(name, X, y, quick):
    if name == "TabPFN-2.5":
        mdl = make_tabpfn(quick)
    else:
        mdl = make_feature_model(name, y, quick)
    mdl.fit(X, y)
    return mdl


def _predict(mdl, X):
    # chunk predictions (TabPFN can be slow / memory-heavy on big test sets)
    if len(X) <= 4096:
        return np.asarray(mdl.predict(X))
    return np.concatenate([np.asarray(mdl.predict(X[i:i + 4096])) for i in range(0, len(X), 4096)])


def _subsample(name, idx, y, quick):
    cap = SUBSAMPLE.get(name)
    if quick:
        cap = min(cap or 4000, 4000)
    if cap is None or len(idx) <= cap:
        return idx
    rng = np.random.RandomState(0)
    # stratified-ish: sample within each class to keep all labels present
    parts = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = max(1, int(round(cap * len(ci) / len(idx))))
        parts.append(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.concatenate(parts)


# --------------------------------------------------------------------------- xLSTM (deep)
def run_xlstm(df, Xdec, task, quick):
    import xlstm_model as xm
    y = _task_labels(df, task)
    n_classes = TASKS[task]["n_classes"]
    epochs = 8 if quick else 16
    cap = 6000 if quick else 9000

    def cw(yt):
        nb = np.bincount(yt, minlength=n_classes).astype(float)
        w = nb.sum() / (np.maximum(nb, 1) * n_classes)
        return w

    out = {}
    # ---- GroupKFold pooled OOF ----
    oof = np.full(len(df), -1, dtype=int)
    for tr, te in stratified_group_folds(df, y_col=_ycol(task)):
        tr_use = _subsample("xLSTM_cap", tr, y, quick) if len(tr) > cap else tr
        tr_use = _cap(tr_use, y, cap)
        pred = xm.train_predict(Xdec[tr_use], y[tr_use], Xdec[te], n_classes,
                                epochs=epochs, class_weight=cw(y[tr_use]))
        oof[te] = pred
    mask = oof >= 0
    out["groupkfold"] = score(task, y[mask], oof[mask])

    # ---- LOLO pooled ----
    yt_all, yp_all = [], []
    for L, tr, te in leave_one_load_out(df):
        if not len(te):
            continue
        tr_use = _cap(tr, y, cap)
        pred = xm.train_predict(Xdec[tr_use], y[tr_use], Xdec[te], n_classes,
                                epochs=epochs, class_weight=cw(y[tr_use]))
        yt_all.append(y[te]); yp_all.append(pred)
    if yt_all:
        out["lolo"] = score(task, np.concatenate(yt_all), np.concatenate(yp_all))
    return out


def _cap(idx, y, cap):
    if len(idx) <= cap:
        return idx
    rng = np.random.RandomState(0)
    parts = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = max(1, int(round(cap * len(ci) / len(idx))))
        parts.append(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.concatenate(parts)


# --------------------------------------------------------------------------- label helpers
def _ycol(task):
    return {"detection": "label", "severity": "_sevrank", "phase": "_phrank"}[task]


def _task_labels(df, task):
    if task == "detection":
        return df["label"].to_numpy(int)
    if task == "severity":
        return df["_sevrank"].to_numpy(int)
    return df["_phrank"].to_numpy(int)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None, help="comma list to restrict roster")
    ap.add_argument("--tasks", default="detection,severity,phase")
    args = ap.parse_args()

    logp = open(DS / "benchmark_run.log", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s); logp.write(s + "\n"); logp.flush()

    df = pd.read_parquet(DS / "features.parquet")
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK)
    df["_phrank"] = df["phase"].map(PHASE_RANK)

    feature_models = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest", "XGBoost", "TabPFN-2.5"]
    deep_models = ["xLSTM"]
    roster = feature_models + deep_models
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        roster = [m for m in roster if m in want]
    tasks = [t.strip() for t in args.tasks.split(",")]

    # TabPFN availability
    tabpfn_ok = "TabPFN-2.5" in roster
    if tabpfn_ok:
        try:
            import tabpfn  # noqa
            if not os.environ.get("TABPFN_TOKEN"):
                # try a cached token quickly; otherwise skip with a clear message
                log("NOTE: TABPFN_TOKEN not set. TabPFN-2.5 will attempt a cached license token; "
                    "if none exists it will be SKIPPED. Get a free token at https://ux.priorlabs.ai")
        except Exception as e:
            log(f"NOTE: tabpfn import failed ({e}); skipping TabPFN-2.5.")
            tabpfn_ok = False

    # frozen-up-front faulty subframe (shared by severity & phase) keeps Xdec aligned
    faulty_mask = df["label"].to_numpy() == 1

    # decimated windows for deep model (load lazily only if a deep model is in roster)
    Xdec_full = None
    if any(m in roster for m in deep_models):
        import xlstm_model as xm
        log("loading + decimating windows.npy for deep model ...")
        Xw = np.asarray(np.load(DS / "windows.npy"), np.float32)
        Xdec_full = xm.decimate(Xw, 8)
        del Xw
        log(f"  decimated windows: {Xdec_full.shape}")

    results = {}
    t_start = time.time()
    for name in roster:
        if name == "TabPFN-2.5" and not tabpfn_ok:
            continue
        results[name] = {}
        for task in tasks:
            cfg = TASKS[task]
            sub = df[faulty_mask].reset_index(drop=True) if cfg["faulty_only"] else df
            log(f"\n=== {name} | {task} (n={len(sub)}) ===")
            t0 = time.time()
            try:
                if name in deep_models:
                    Xd = Xdec_full[faulty_mask] if cfg["faulty_only"] else Xdec_full
                    res = run_xlstm(sub, Xd, task, args.quick)
                else:
                    res = run_feature_model(name, sub, None, cfg, task, args.quick)
                results[name][task] = res
                for proto, m in res.items():
                    extra = (f" within1={m.get('within1'):.3f} mae={m.get('mae'):.3f} qwk={m.get('qwk'):.3f}"
                             if task == "severity" else "")
                    log(f"  {proto:11s} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}{extra}")
            except Exception as e:
                import traceback
                log(f"  FAILED: {e}")
                log(traceback.format_exc())
                if name == "TabPFN-2.5" and "license" in str(e).lower():
                    log("  -> TabPFN-2.5 needs a license token; set TABPFN_TOKEN and re-run "
                        "`python benchmark_models.py --models TabPFN-2.5`.")
                results[name][task] = {"error": str(e)}
            log(f"  ({time.time()-t0:.1f}s)")

    # ---- MERGE into any existing results (so running a subset doesn't clobber prior models) ----
    merged = {}
    bjson = DS / "benchmark_results.json"
    if bjson.exists():
        try:
            merged = json.load(open(bjson)).get("results", {})
        except Exception:
            merged = {}
    merged.update(results)   # new/re-run models overwrite their own old entry
    results = merged

    # ---- save nested json ----
    payload = {"results": results, "meta": {
        "quick": args.quick, "tasks": tasks, "roster": roster,
        "elapsed_sec": round(time.time() - t_start, 1),
        "protocols": ["groupkfold", "lolo"],
        "severity_levels": SEV_LEVELS,
        "note": "Identical leakage-safe splits as splits.py. Severity scored as 7-level "
                "ordinal classification (mae/within1/qwk). Feature models on features.parquet; "
                "xLSTM on decimated windows.npy.",
    }}
    (DS / "benchmark_results.json").write_text(json.dumps(payload, indent=2))

    # ---- flat tidy table ----
    rows = []
    for mdl, td in results.items():
        for task, pd_ in td.items():
            for proto, m in pd_.items():
                if "error" in m:
                    rows.append(dict(model=mdl, task=task, protocol=proto, error=m["error"]))
                else:
                    rows.append(dict(model=mdl, task=task, protocol=proto, **m))
    pd.DataFrame(rows).to_csv(DS / "benchmark_table.csv", index=False)
    log(f"\nSaved dataset/benchmark_results.json and dataset/benchmark_table.csv "
        f"({time.time()-t_start:.1f}s total)")
    logp.close()


if __name__ == "__main__":
    main()
