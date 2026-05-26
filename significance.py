r"""
Statistical significance across models (Demšar protocol).

The benchmark (benchmark_models.py) reports pooled metrics; here we collect the
PER-FOLD (StratifiedGroupKFold) and PER-LOAD (Leave-One-Load-Out) primary-metric
scores on identical folds, then test whether the models differ:

  * Friedman omnibus test across all models (blocks = folds / held-out loads).
  * Pairwise Wilcoxon signed-rank tests vs two references (XGBoost = deployed,
    TabPFN-2.5 = best), Holm-corrected for multiple comparisons.
  * Average Friedman rank per model (lower = better).

Primary metric: detection & phase = macro-F1; severity = within-one-level accuracy.
Models: the 7 feature/foundation models (xLSTM omitted from the inferential test —
its per-fold re-run is hours of CPU and it trails all others by margins far beyond
fold spread; reported descriptively in the paper).

Outputs: dataset/significance_results.json, dataset/significance_table.md
Run:  python significance.py          (full)
      python significance.py --quick  (smoke test: fewer models)
"""
from __future__ import annotations
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
from splits import stratified_group_folds, leave_one_load_out          # noqa: E402
from feature_sets import all_features, severity_features               # noqa: E402
from benchmark_models import (make_feature_model, make_tabpfn, _subsample,  # noqa: E402
                              SEV_RANK, PHASE_RANK)

MODELS = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest", "XGBoost", "TabPFN-2.5"]
TASKS = {"detection": ("all", "label", False),
         "severity":  ("severity", "_sevrank", True),
         "phase":     ("all", "_phrank", True)}


def primary(task, yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    if task == "severity":
        return float((np.abs(yt - yp) <= 1).mean())          # within-1
    return float(f1_score(yt, yp, average="macro", zero_division=0))  # macro-F1


def fit_predict(name, Xtr, ytr, Xte, quick):
    mdl = make_tabpfn(quick) if name == "TabPFN-2.5" else make_feature_model(name, ytr, quick)
    mdl.fit(Xtr, ytr)
    if len(Xte) <= 4096:
        return np.asarray(mdl.predict(Xte))
    return np.concatenate([np.asarray(mdl.predict(Xte[i:i+4096])) for i in range(0, len(Xte), 4096)])


def per_block_scores(name, df, Xcols, task, ycol, quick):
    """Return dict protocol -> list of per-fold / per-load primary scores."""
    X = df[Xcols].to_numpy(np.float32)
    y = df[ycol].to_numpy(int)
    out = {"groupkfold": [], "lolo": []}
    for tr, te in stratified_group_folds(df, y_col=ycol):
        tru = _subsample(name, tr, y, quick)
        out["groupkfold"].append(primary(task, y[te], fit_predict(name, X[tru], y[tru], X[te], quick)))
    for L, tr, te in leave_one_load_out(df):
        if not len(te):
            continue
        tru = _subsample(name, tr, y, quick)
        out["lolo"].append(primary(task, y[te], fit_predict(name, X[tru], y[tru], X[te], quick)))
    return out


def holm(pairs):
    """pairs: list of (label, p). Return list of (label, p, p_holm, reject@0.05)."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs); adj = [0.0] * m; running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pairs[i][1])
        adj[i] = min(1.0, running)
    return [(pairs[i][0], pairs[i][1], adj[i], adj[i] < 0.05) for i in range(m)]


def safe_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    models = [m for m in MODELS if (not args.models or m in args.models.split(","))]

    df = pd.read_parquet(DS / "features.parquet")
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    faulty = df["label"].to_numpy() == 1

    # collect per-block scores: scores[task][protocol][model] = [per-fold/-load]
    scores = {t: {"groupkfold": {}, "lolo": {}} for t in TASKS}
    for name in models:
        for task, (feat, ycol, fonly) in TASKS.items():
            sub = df[faulty].reset_index(drop=True) if fonly else df
            cols = (all_features(sub.columns) if feat == "all"
                    else severity_features(sub.columns, load_aware=True))
            cols = [c for c in cols if c not in ("_sevrank", "_phrank")]
            res = per_block_scores(name, sub, cols, task, ycol, args.quick)
            for proto in ("groupkfold", "lolo"):
                scores[task][proto][name] = res[proto]
            print(f"{name:13s} {task:9s} GK={np.mean(res['groupkfold']):.3f} "
                  f"LOLO={np.mean(res['lolo']):.3f}")

    # ---- tests ----
    results = {}
    for task in TASKS:
        results[task] = {}
        for proto in ("groupkfold", "lolo"):
            mat = {m: scores[task][proto][m] for m in models}
            n_blocks = min(len(v) for v in mat.values())
            arrs = [np.array(mat[m][:n_blocks]) for m in models]
            # Friedman omnibus
            try:
                fr = friedmanchisquare(*arrs)
                fried = {"stat": float(fr.statistic), "p": float(fr.pvalue)}
            except Exception as e:
                fried = {"stat": None, "p": None, "err": str(e)}
            # average Friedman rank (higher score -> rank 1); ranks per block then mean
            R = np.vstack([rankdata(-np.array([mat[m][b] for m in models])) for b in range(n_blocks)])
            avg_rank = {m: float(R[:, j].mean()) for j, m in enumerate(models)}
            # pairwise Wilcoxon vs references, Holm-corrected
            pw = {}
            for ref in ("XGBoost", "TabPFN-2.5"):
                if ref not in models:
                    continue
                pairs = [(m, safe_wilcoxon(mat[m][:n_blocks], mat[ref][:n_blocks]))
                         for m in models if m != ref]
                pw[ref] = [{"model": lbl, "p": p, "p_holm": ph, "sig": bool(sig)}
                           for (lbl, p, ph, sig) in holm(pairs)]
            results[task][proto] = {"n_blocks": n_blocks, "friedman": fried,
                                    "avg_rank": avg_rank, "means": {m: float(np.mean(mat[m])) for m in models},
                                    "pairwise_vs": pw}

    (DS / "significance_results.json").write_text(json.dumps(
        {"models": models, "results": results,
         "note": "Friedman omnibus + Holm-corrected pairwise Wilcoxon on per-fold (GK) / "
                 "per-load (LOLO) primary scores. detection/phase=macro-F1, severity=within-1. "
                 "xLSTM excluded from the inferential test (cost); trails all models descriptively."},
        indent=2))

    # ---- markdown ----
    L = ["# Significance across models (Demšar protocol)", "",
         "Friedman omnibus + Holm-corrected pairwise Wilcoxon signed-rank on per-fold "
         "(GroupKFold, 5 blocks) and per-load (LOLO, 6 blocks) primary scores "
         "(detection/phase = macro-F1, severity = within-1). Lower average rank = better.", ""]
    for task in TASKS:
        L.append(f"## {task.capitalize()}")
        for proto in ("groupkfold", "lolo"):
            r = results[task][proto]; fr = r["friedman"]
            pstr = (f"Friedman χ²={fr['stat']:.2f}, p={fr['p']:.4f} "
                    f"({'significant' if fr['p'] is not None and fr['p'] < 0.05 else 'n.s.'})"
                    if fr.get("p") is not None else "Friedman: n/a")
            L += [f"", f"**{proto}** (blocks={r['n_blocks']}): {pstr}", "",
                  "| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |",
                  "|---|---|---|---|---|"]
            holm_xgb = {d["model"]: d for d in r["pairwise_vs"].get("XGBoost", [])}
            holm_tab = {d["model"]: d for d in r["pairwise_vs"].get("TabPFN-2.5", [])}
            for m in models:
                hx = holm_xgb.get(m); ht = holm_tab.get(m)
                fx = ("—" if m == "XGBoost" else (f"{hx['p_holm']:.3f}{'*' if hx['sig'] else ''}" if hx else "—"))
                ft = ("—" if m == "TabPFN-2.5" else (f"{ht['p_holm']:.3f}{'*' if ht['sig'] else ''}" if ht else "—"))
                L.append(f"| {m} | {r['means'][m]:.3f} | {r['avg_rank'][m]:.2f} | {fx} | {ft} |")
        L.append("")
    (DS / "significance_table.md").write_text("\n".join(L), encoding="utf-8")
    print("\nsaved dataset/significance_results.json + dataset/significance_table.md")


if __name__ == "__main__":
    main()
