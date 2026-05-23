r"""
PADA — Physics-Anchored Domain Adaptation (simulation -> real ibarram).

The naive sim-trained model collapses on real data (detection F1 0.48, phase F1 0.32) because
feature distributions shift across domains. We evaluate a ladder of adaptation strategies for the
feature/XGBoost track and report the recovery:

  (0) naive                 : train sim, test real (no adaptation)
  (1) z-score align         : standardize features within each domain (unsupervised on target marginals)
  (2) quantile align        : per-domain rank/quantile transform (marginal alignment, robust for trees)
  (3) CORAL                 : align source 2nd-order statistics to target covariance (unsupervised)
  (4) physics-anchored      : quantile-align ONLY the domain-stable physics features (ratios/angles/
                              |I1|-normalised) -> the proposed anchor (novelty)
  (5) few-shot (+k reps)    : add k labelled target repetitions to training, test on held-out reps

Detection (binary) and faulted-phase (3-class) transfer directly; severity scales differ across
datasets and are excluded. Run:  python experimental/pada.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from feature_sets import all_features, INVARIANT, I1NORM  # noqa: E402

PMAP = {"A": 0, "B": 1, "C": 2}
RNG = 42


def clf(num_class=None, y=None):
    kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=RNG)
    if num_class:
        return XGBClassifier(objective="multi:softprob", num_class=num_class,
                             eval_metric="mlogloss", **kw)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         scale_pos_weight=nneg / max(npos, 1), **kw)


def coral(Xs, Xt, eps=1e-3):
    """Recolor source features to target covariance (CORAL), unsupervised."""
    def msqrt(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1/np.sqrt(w)) @ V.T
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps*np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps*np.eye(Xt.shape[1])
    _, Cs_isqrt = msqrt(Cs); Ct_sqrt, _ = msqrt(Ct)
    Xs2 = (Xs - ms) @ Cs_isqrt @ Ct_sqrt + mt
    return Xs2.astype(np.float32)


def run_task(sim, exp, feats, target_col, num_class, task):
    Xs = sim[feats].to_numpy(np.float32)
    Xt = exp[feats].to_numpy(np.float32)
    if task == "detect":
        ys = sim["label"].to_numpy(int); yt = exp["label"].to_numpy(int)
        mk = lambda: clf(y=ys)
    else:
        ys = sim[target_col].map(PMAP).to_numpy(int); yt = exp[target_col].map(PMAP).to_numpy(int)
        mk = lambda: clf(num_class=num_class)
    res = {}

    def score(pred):
        return dict(acc=round(float(accuracy_score(yt, pred)), 4),
                    macro_f1=round(float(f1_score(yt, pred, average="macro")), 4))

    # (0) naive
    res["0_naive"] = score(mk().fit(Xs, ys).predict(Xt))
    # (1) z-score per-domain
    ss, st = StandardScaler().fit(Xs), StandardScaler().fit(Xt)
    res["1_zscore"] = score(mk().fit(ss.transform(Xs), ys).predict(st.transform(Xt)))
    # (2) quantile per-domain
    qs = QuantileTransformer(output_distribution="normal", random_state=RNG,
                             n_quantiles=min(1000, len(Xs))).fit(Xs)
    qt = QuantileTransformer(output_distribution="normal", random_state=RNG,
                             n_quantiles=min(1000, len(Xt))).fit(Xt)
    res["2_quantile"] = score(mk().fit(qs.transform(Xs), ys).predict(qt.transform(Xt)))
    # (3) CORAL
    res["3_coral"] = score(mk().fit(coral(Xs, Xt), ys).predict(Xt))
    return res, (ys, yt)


def run_physics_anchored(sim, exp, feats_all, target_col, num_class, task):
    """Quantile-align ONLY the domain-stable physics features (ratios/angles/|I1|-norm)."""
    anchor = [f for f in (INVARIANT + I1NORM +
              ["I2_angle_sin", "I2_angle_cos", "I2_rel_angle_sin", "I2_rel_angle_cos",
               "argmax_rms_phase"]) if f in sim.columns and f in exp.columns]
    Xs = sim[anchor].to_numpy(np.float32); Xt = exp[anchor].to_numpy(np.float32)
    if task == "detect":
        ys, yt = sim["label"].to_numpy(int), exp["label"].to_numpy(int); mk = lambda: clf(y=ys)
    else:
        ys = sim[target_col].map(PMAP).to_numpy(int); yt = exp[target_col].map(PMAP).to_numpy(int)
        mk = lambda: clf(num_class=num_class)
    qs = QuantileTransformer(output_distribution="normal", random_state=RNG,
                             n_quantiles=min(1000, len(Xs))).fit(Xs)
    qt = QuantileTransformer(output_distribution="normal", random_state=RNG,
                             n_quantiles=min(1000, len(Xt))).fit(Xt)
    pred = mk().fit(qs.transform(Xs), ys).predict(qt.transform(Xt))
    return dict(n_anchor_feats=len(anchor),
                acc=round(float(accuracy_score(yt, pred)), 4),
                macro_f1=round(float(f1_score(yt, pred, average="macro")), 4))


def run_fewshot(sim, exp, feats, target_col, num_class, task, train_reps=("Repetition01", "Repetition02")):
    rep = exp["case_id"].str.extract(r"_(Repetition\d+)$")[0]
    tr_mask = rep.isin(train_reps).to_numpy()
    Xs = sim[feats].to_numpy(np.float32)
    Xt = exp[feats].to_numpy(np.float32)
    if task == "detect":
        ys = sim["label"].to_numpy(int); yt = exp["label"].to_numpy(int); mk = lambda: clf(y=np.r_[ys, yt[tr_mask]])
    else:
        ys = sim[target_col].map(PMAP).to_numpy(int); yt = exp[target_col].map(PMAP).to_numpy(int)
        mk = lambda: clf(num_class=num_class)
    ss, st = StandardScaler().fit(Xs), StandardScaler().fit(Xt)
    Xs_a, Xt_a = ss.transform(Xs), st.transform(Xt)
    Xtr = np.vstack([Xs_a, Xt_a[tr_mask]]); ytr = np.r_[ys, yt[tr_mask]]
    pred = mk().fit(Xtr, ytr).predict(Xt_a[~tr_mask])
    yte = yt[~tr_mask]
    return dict(n_target_train=int(tr_mask.sum()), n_target_test=int((~tr_mask).sum()),
                acc=round(float(accuracy_score(yte, pred)), 4),
                macro_f1=round(float(f1_score(yte, pred, average="macro")), 4))


def main():
    sim = pd.read_parquet(ROOT / "dataset" / "features.parquet")
    exp = pd.read_parquet(ROOT / "experimental" / "ibarram" / "features.parquet")
    feats = [c for c in all_features(sim.columns) if c in exp.columns]
    print(f"shared feats={len(feats)}  sim={len(sim)}  exp={len(exp)}\n")
    out = {}
    for task, num_class, label in [("detect", None, "DETECTION (binary)"),
                                   ("phase", 3, "FAULTED PHASE (3-class)")]:
        subS = sim if task == "detect" else sim[sim.label == 1]
        subE = exp if task == "detect" else exp[exp.label == 1]
        res, _ = run_task(subS, subE, feats, "phase", num_class, task)
        res["4_physics_anchored"] = run_physics_anchored(subS, subE, feats, "phase", num_class, task)
        res["5_fewshot_2reps"] = run_fewshot(subS, subE, feats, "phase", num_class, task)
        out[task] = res
        print(f"=== {label}: sim -> real (macro-F1) ===")
        for k, v in res.items():
            print(f"  {k:20} F1={v['macro_f1']:.4f}  acc={v['acc']:.4f}"
                  + (f"  [{v.get('n_anchor_feats','')} feats]" if 'n_anchor_feats' in v else "")
                  + (f"  [test n={v.get('n_target_test','')}]" if 'n_target_test' in v else ""))
        print()
    json.dump(out, open(ROOT / "experimental" / "ibarram" / "pada_results.json", "w"), indent=2)
    print("saved experimental/ibarram/pada_results.json")


if __name__ == "__main__":
    main()
