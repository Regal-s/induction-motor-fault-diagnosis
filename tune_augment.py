r"""
Best-results pipeline: leakage-safe hyperparameter tuning (Optuna TPE, cross-load inner
objective) + physics-consistent training-fold augmentation, on the deployed XGBoost feature
pipeline, evaluated under 10-fold StratifiedGroupKFold (in-distribution) and Leave-One-Load-Out
(LOLO, primary). Keeps every method that has helped: load-aware severity, the load-invariant
severity feature set, and per-recording aggregation.

Protocol (papers/AUGMENTATION_TUNING_RESEARCH.md):
  - Tune for CROSS-LOAD generalization: inner objective = leave-one-load-out over the loads
    (DomainBed model-selection principle); XGBoost early-stops on an inner held-out load.
  - HPs are selected by a single inner-LOLO study and then FROZEN for both the 10-fold GK and the
    LOLO reports (flat selection ~ nested per Wainer & Cawley 2021; mild LOLO optimism noted).
  - Every final fit (all conditions) holds out one training load as the early-stopping validation.
Augmentation (train fold only, raw windows re-featurised): joint amplitude scaling + per-channel
SNR jitter + mild magnitude-warp + phase-coherent shift; the phase task also adds faulted-phase
rotation (exact symmetry). Conditions per task: {default, tuned, tuned+aug}.

Outputs: dataset/tune_augment_results.json, dataset/tune_augment_table.md.
Run:  python tune_augment.py [--tasks ...] [--trials 25] [--naug 1] [--folds 10] [--quick]
"""
from __future__ import annotations
import sys, json, argparse, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import f1_score, mean_absolute_error, cohen_kappa_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from splits import stratified_group_folds, leave_one_load_out
from feature_sets import all_features, severity_features
import augment as aug
from features import build_feature_frame

SEV_LEVELS = [0.3, 0.5, 1, 2, 3, 4, 5]
SEV_RANK = {s: i for i, s in enumerate(SEV_LEVELS)}
PHASE_RANK = {"A": 0, "B": 1, "C": 2}
FS, F0 = 25000.0, 50.0
TASKS = {"detection": ("all", "label", False, "clf", 2),
         "severity":  ("severity", "_sevrank", True, "reg", 7),
         "phase":     ("all", "_phrank", True, "clf", 3)}
DEFAULT = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, reg_lambda=1.0)


def primary(task, yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    if task == "severity":
        return float((np.abs(yt - np.rint(yp)) <= 1).mean())
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def full_metrics(task, yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    if task == "severity":
        yr = np.rint(yp).astype(int).clip(0, 6)
        return {"within1": float((np.abs(yt - yr) <= 1).mean()), "exact": float((yt == yr).mean()),
                "mae": float(mean_absolute_error(yt, yr)),
                "qwk": float(cohen_kappa_score(yt, yr, weights="quadratic", labels=list(range(7))))}
    return {"macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "acc": float(accuracy_score(yt, yp))}


def make_model(task, params, es=False):
    import gpu
    p = dict(n_jobs=-1, tree_method="hist", device=gpu.xgb_device(), random_state=0, **params)
    if es:
        p["early_stopping_rounds"] = 50
    if TASKS[task][3] == "reg":
        return XGBRegressor(objective="reg:squarederror", **p)
    nc = TASKS[task][4]
    if nc == 2:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **p)
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", num_class=nc, **p)


def fit_es(task, params, Xtr, ytr, Xval, yval):
    m = make_model(task, {**params}, es=True)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return m


def sample_params(trial):
    return dict(n_estimators=2000,
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 5.0))


def tune(task, X, y, loads, n_trials, seed=0):
    inner = [L for L in np.unique(loads) if (loads == L).sum() > 0]
    def objective(trial):
        params = sample_params(trial); sc = []
        for L in inner:
            tr = loads != L; va = loads == L
            if len(np.unique(y[tr])) < 2:
                continue
            m = fit_es(task, params, X[tr], y[tr], X[va], y[va])
            sc.append(primary(task, y[va], m.predict(X[va])))
        return float(np.mean(sc)) if sc else 0.0
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    bp = study.best_params; bp["n_estimators"] = 2000
    return bp, float(study.best_value)


def augment_feats(idx, Xw_sub, sub, feats, task, ycol, naug, seed):
    """Augment raw training windows and re-featurise. Signal features come from
    build_feature_frame; non-signal features (e.g. measured load_pct) are carried from each
    augmented window's SOURCE row (augmentation does not change the recording's labelled load)."""
    Xr = np.asarray(Xw_sub[idx]); yt = sub.iloc[idx][ycol].to_numpy(int)
    Xa, src = aug.augment_batch(Xr, samples_per_cycle=500, n_aug=naug, seed=seed)
    ya = yt[src]; src_all = src
    if task == "phase":
        for k in (1, 2):
            Xrot, yrot = aug.phase_rotate(Xr, yt, k)
            Xa = np.concatenate([Xa, Xrot], 0); ya = np.concatenate([ya, yrot], 0)
            src_all = np.concatenate([src_all, np.arange(len(idx))], 0)
    lab = pd.DataFrame({"case_id": np.arange(len(Xa)), "label": np.ones(len(Xa), int)})
    fr = build_feature_frame(Xa, lab, fs=FS, f0=F0, verbose=False)
    sub_idx = sub.iloc[idx].reset_index(drop=True)
    out = np.empty((len(Xa), len(feats)), np.float32)
    for j, f in enumerate(feats):
        if f in fr.columns:
            out[:, j] = fr[f].to_numpy(np.float32)
        else:                                   # non-signal feature (e.g. load_pct): carry from source
            out[:, j] = sub_idx[f].to_numpy(np.float32)[src_all]
    return out, ya


def final_fit(task, params, tr, X, y, loads, sub, Xw_sub, feats, ycol, use_aug, naug, seed):
    """Fit on training rows `tr` (sub-positional idx). Hold out one training load for ES;
    augment only the inner-train when use_aug."""
    tl = np.unique(loads[tr])
    Lv = tl[-1] if len(tl) > 1 else tl[0]
    va = loads[tr] == Lv; tri = ~va
    Xt, yt = X[tr][tri], y[tr][tri]
    Xv, yv = X[tr][va], y[tr][va]
    if len(Xv) == 0 or len(np.unique(yt)) < 2:    # fallback: no grouped val
        return make_model(task, params).fit(X[tr], y[tr])
    if use_aug:
        Xag, yag = augment_feats(tr[tri], Xw_sub, sub, feats, task, ycol, naug, seed)
        Xt = np.concatenate([Xt, Xag], 0); yt = np.concatenate([yt, yag], 0)
    return fit_es(task, params, Xt, yt, Xv, yv)


def evaluate(task, df, Xw, n_trials, naug, n_folds, quick):
    feat_set, ycol, fonly, _, _ = TASKS[task]
    sub_mask = (df["label"].to_numpy() == 1) if fonly else np.ones(len(df), bool)
    sub = df[sub_mask].reset_index(drop=True)
    Xw_sub = np.asarray(Xw[sub_mask])
    feats = (all_features(sub.columns) if feat_set == "all"
             else severity_features(sub.columns, load_aware=True))
    feats = [c for c in feats if c not in ("_sevrank", "_phrank") and pd.api.types.is_numeric_dtype(sub[c])]
    X = sub[feats].to_numpy(np.float32); y = sub[ycol].to_numpy(int)
    loads = sub["load_label"].astype(str).to_numpy(); cid = sub["case_id"].to_numpy()

    tuned_hp, best = (DEFAULT, 0.0) if quick else tune(task, X, y, loads, n_trials)
    print(f"  [{task}] tuned inner-LOLO {primary.__name__}={best:.3f}  HP={ {k:(round(v,4) if isinstance(v,float) else v) for k,v in tuned_hp.items() if k!='n_estimators'} }")

    def aggregate(yt, yp, cids):
        d = pd.DataFrame({"c": cids, "t": yt, "p": yp}); rt, rp = [], []
        for c, g in d.groupby("c"):
            rt.append(g["t"].iloc[0])
            rp.append(int(np.rint(g["p"].median())) if task == "severity" else int(g["p"].mode().iloc[0]))
        return np.array(rt), np.array(rp)

    conds = [("default", DEFAULT, False), ("tuned", tuned_hp, False), ("tuned_aug", tuned_hp, True)]
    if quick:
        conds = conds[:2]
    res = {}
    for cname, hp, use_aug in conds:
        res[cname] = {}
        # ---- LOLO ----
        yt_all, yp_all, c_all = [], [], []
        for i, (L, tr, te) in enumerate(leave_one_load_out(sub)):
            if not len(te):
                continue
            m = final_fit(task, hp, tr, X, y, loads, sub, Xw_sub, feats, ycol, use_aug, naug, seed=100 + i)
            yp_all.append(m.predict(X[te])); yt_all.append(y[te]); c_all.append(cid[te])
        yt, yp, cc = map(np.concatenate, (yt_all, yp_all, c_all))
        res[cname]["lolo_window"] = full_metrics(task, yt, yp)
        # ---- 10-fold StratifiedGroupKFold ----
        oof = np.full(len(sub), np.nan)
        for i, (tr, te) in enumerate(stratified_group_folds(sub, y_col=ycol, n_splits=n_folds)):
            m = final_fit(task, hp, tr, X, y, loads, sub, Xw_sub, feats, ycol, use_aug, naug, seed=200 + i)
            oof[te] = m.predict(X[te])
        mask = ~np.isnan(oof)
        res[cname]["gk_window"] = full_metrics(task, y[mask], oof[mask])
        # per-recording aggregation: ONLY valid for faulty-only tasks (a recording mixes H/F for detection)
        rec_msg = ""
        if fonly:
            rt, rp = aggregate(yt, yp, cc); rt2, rp2 = aggregate(y[mask], oof[mask], cid[mask])
            res[cname]["lolo_record"] = full_metrics(task, rt, rp)
            res[cname]["gk_record"] = full_metrics(task, rt2, rp2)
            rec_msg = f" rec {primary(task,rt,rp):.3f}"
            rec_msg2 = f" rec {primary(task,rt2,rp2):.3f}"
        else:
            rec_msg = rec_msg2 = " rec n/a"
        print(f"  [{task}] {cname:10s} | LOLO win {primary(task,yt,yp):.3f}{rec_msg}"
              f" | GK win {primary(task,y[mask],oof[mask]):.3f}{rec_msg2}")
    res["_tuned_hp"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in tuned_hp.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="detection,severity,phase")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--naug", type=int, default=1)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(DS / "features.parquet").reset_index(drop=True)
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    Xw = np.load(DS / "windows.npy", mmap_mode="r")
    assert len(df) == Xw.shape[0]
    out = {}
    t0 = time.time()
    for task in args.tasks.split(","):
        print(f"=== {task} (trials={args.trials} naug={args.naug} folds={args.folds}) ===")
        out[task] = evaluate(task, df, Xw, args.trials, args.naug, args.folds, args.quick)
        out["_meta"] = {"trials": args.trials, "naug": args.naug, "folds": args.folds,
                        "note": "XGBoost; HPs tuned via inner leave-one-load-out (Optuna TPE), frozen for "
                                "GK & LOLO; train-fold augmentation re-featurised; per-recording aggregation."}
        (DS / "tune_augment_results.json").write_text(json.dumps(out, indent=2))
        print(f"  saved (elapsed {time.time()-t0:.0f}s)")
    print(f"\nDONE {time.time()-t0:.0f}s -> dataset/tune_augment_results.json")


if __name__ == "__main__":
    main()
