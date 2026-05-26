r"""
WITHIN-HARDWARE evaluation — train AND test our models on the real (ibarram) motor data
itself, leakage-safe, for every task the dataset supports. This is the "do the models work
on a real motor" test, complementary to the sim->real transfer study (pada.py).

Dataset: ibarram/ITSC, real 0.75 hp IM, 3-phase current @1 kHz/60 Hz, no load.
  1235 windows; 65 recordings = 13 classes x 5 repetitions; 1140 faulty / 95 healthy.
  Severity has its OWN 4-level scale {10,20,30,40}% (so within-hardware severity IS
  evaluable, even though sim->real severity transfer is not, due to the scale mismatch).

Tasks:  detection (H/F) | severity (4 levels) | phase (A/B/C, faulty only)
Models: LogReg, SVM-RBF, KNN, MLP, RandomForest, XGBoost, TabPFN-2.5 (features); xLSTM (raw).
Leakage-safe protocols (no load variation exists, so LOLO is replaced by repetitions):
  - StratifiedGroupKFold grouped by recording (case_id), 5 folds      [in-domain]
  - Leave-One-Repetition-Out (LORO): hold out one of the 5 reps        [unseen recording]

Outputs: dataset/hardware_results.json, dataset/hardware_table.md,
         figures_svg/fig_hw_confusion_{detection,severity,phase}.svg/.pdf (XGBoost OOF).
TabPFN-2.5 needs TABPFN_TOKEN. Run:  python hardware_eval.py [--quick] [--models ...]
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
EXP = ROOT / "experimental" / "ibarram"
OUTF = ROOT / "figures_svg"
from splits import stratified_group_folds                       # noqa: E402
from feature_sets import all_features, severity_features        # noqa: E402
from benchmark_models import make_feature_model, make_tabpfn, _subsample, score  # noqa: E402

SEV_RANK_HW = {10.0: 0, 20.0: 1, 30.0: 2, 40.0: 3}   # ibarram severity scale
SEV_LABELS = ["10", "20", "30", "40"]
PHASE_RANK = {"A": 0, "B": 1, "C": 2}
MODELS = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest", "XGBoost", "TabPFN-2.5", "xLSTM"]
TASKS = {  # name -> (feature_set, label_col, faulty_only, n_classes)
    "detection": ("all", "label", False, 2),
    "severity":  ("severity", "_sevrank", True, 4),
    "phase":     ("all", "_phrank", True, 3),
}


def leave_one_rep_out(df):
    rep = df["_rep"].to_numpy()
    for r in sorted(np.unique(rep)):
        te = np.where(rep == r)[0]; tr = np.where(rep != r)[0]
        if len(te) and len(np.unique(df["label"].to_numpy()[tr])) > 1 or len(te):
            yield r, tr, te


def fit_predict_feat(name, Xtr, ytr, Xte, quick):
    mdl = make_tabpfn(quick) if name == "TabPFN-2.5" else make_feature_model(name, ytr, quick)
    mdl.fit(Xtr, ytr)
    if len(Xte) <= 4096:
        return np.asarray(mdl.predict(Xte))
    return np.concatenate([np.asarray(mdl.predict(Xte[i:i+4096])) for i in range(0, len(Xte), 4096)])


def run_feature(name, df, cols, task, ycol, quick):
    X = df[cols].to_numpy(np.float32); y = df[ycol].to_numpy(int)
    out, oof = {}, np.full(len(df), -1, dtype=int)
    for tr, te in stratified_group_folds(df, y_col=ycol, group_col="case_id"):
        tru = _subsample(name, tr, y, quick)
        oof[te] = fit_predict_feat(name, X[tru], y[tru], X[te], quick)
    out["groupkfold"] = score(task, y[oof >= 0], oof[oof >= 0])
    yt, yp = [], []
    for r, tr, te in leave_one_rep_out(df):
        tru = _subsample(name, tr, y, quick)
        yp.append(fit_predict_feat(name, X[tru], y[tru], X[te], quick)); yt.append(y[te])
    out["loro"] = score(task, np.concatenate(yt), np.concatenate(yp))
    return out, oof


def run_xlstm(df, Xdec, task, ycol, n_classes, quick):
    import xlstm_model as xm
    y = df[ycol].to_numpy(int)
    ep = 10 if quick else 25

    def cw(yt):
        nb = np.bincount(yt, minlength=n_classes).astype(float)
        return nb.sum() / (np.maximum(nb, 1) * n_classes)
    out, oof = {}, np.full(len(df), -1, dtype=int)
    for tr, te in stratified_group_folds(df, y_col=ycol, group_col="case_id"):
        oof[te] = xm.train_predict(Xdec[tr], y[tr], Xdec[te], n_classes, epochs=ep, class_weight=cw(y[tr]))
    out["groupkfold"] = score(task, y[oof >= 0], oof[oof >= 0])
    yt, yp = [], []
    for r, tr, te in leave_one_rep_out(df):
        yp.append(xm.train_predict(Xdec[tr], y[tr], Xdec[te], n_classes, epochs=ep, class_weight=cw(y[tr])))
        yt.append(y[te])
    out["loro"] = score(task, np.concatenate(yt), np.concatenate(yp))
    return out, oof


def cm_fig(cm, labels, title, name, cmap):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(0.5*len(labels)+1.8, 0.5*len(labels)+1.6))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    th = cm.max()/2 if cm.max() else 1
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(int(v)), ha="center", va="center", color="white" if v > th else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(OUTF / (name + ".svg"), format="svg", bbox_inches="tight")
    (ROOT/"manuscript"/"figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT/"manuscript"/"figs"/(name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    roster = [m for m in MODELS if (not args.models or m in args.models.split(","))]

    df = pd.read_parquet(EXP / "features.parquet").reset_index(drop=True)
    df["_rep"] = df["case_id"].str.extract(r"Repetition(\d+)")[0].astype(int)
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK_HW)
    df["_phrank"] = df["phase"].map(PHASE_RANK)
    faulty = df["label"].to_numpy() == 1
    print(f"hardware set: {len(df)} windows, {df['case_id'].nunique()} recordings, "
          f"{int(faulty.sum())} faulty / {int((~faulty).sum())} healthy, reps={sorted(df['_rep'].unique())}")

    Xdec = None
    if "xLSTM" in roster:
        import xlstm_model as xm
        Xw = np.load(EXP / "windows.npy").astype(np.float32)
        Xdec = xm.decimate(Xw, 2)   # 500 -> 250 samples
        print("decimated real windows:", Xdec.shape)

    results, oof_store = {}, {}
    for name in roster:
        if name == "TabPFN-2.5" and not (os.environ.get("TABPFN_TOKEN")):
            print("skip TabPFN-2.5 (no TABPFN_TOKEN)"); continue
        results[name] = {}
        for task, (feat, ycol, fonly, nc) in TASKS.items():
            sub = df[faulty].reset_index(drop=True) if fonly else df
            try:
                if name == "xLSTM":
                    Xd = Xdec[faulty] if fonly else Xdec
                    res, oof = run_xlstm(sub, Xd, task, ycol, nc, args.quick)
                else:
                    cols = (all_features(sub.columns) if feat == "all"
                            else severity_features(sub.columns, load_aware=False))  # no load in hardware
                    # keep only numeric feature columns (drop helper/meta like 'dataset', '_rep')
                    cols = [c for c in cols if c not in ("_sevrank", "_phrank", "_rep", "dataset")
                            and pd.api.types.is_numeric_dtype(sub[c])]
                    res, oof = run_feature(name, sub, cols, task, ycol, args.quick)
                results[name][task] = res
                oof_store[(name, task)] = (sub, oof, ycol)
                for proto, m in res.items():
                    extra = (f" w1={m['within1']:.3f} mae={m['mae']:.3f} qwk={m['qwk']:.3f}"
                             if task == "severity" else "")
                    print(f"  {name:12s} {task:9s} {proto:10s} acc={m['acc']:.3f} F1={m['macro_f1']:.3f}{extra}")
            except Exception as e:
                import traceback; print(f"  {name} {task} FAILED: {e}"); traceback.print_exc()
                results[name][task] = {"error": str(e)}

    payload = {"results": results, "meta": {
        "dataset": "ibarram/ITSC real 0.75hp IM, 1kHz/60Hz, no-load",
        "n_windows": len(df), "n_recordings": int(df["case_id"].nunique()),
        "protocols": ["groupkfold(case_id)", "loro(leave-one-repetition-out)"],
        "severity_levels_pct": [10, 20, 30, 40],
        "note": "Within-hardware train+test (NOT sim->real transfer). Leakage-safe by recording. "
                "No load variation in this dataset, so cross-load (LOLO) is not assessable here."}}
    (DS / "hardware_results.json").write_text(json.dumps(payload, indent=2))

    rows = []
    for mdl, td in results.items():
        for task, pd_ in td.items():
            if "error" in pd_:                       # whole task failed
                rows.append(dict(model=mdl, task=task, protocol="-", error=pd_["error"]))
                continue
            for proto, m in pd_.items():             # m is a metric dict per protocol
                rows.append(dict(model=mdl, task=task, protocol=proto, **m))
    pd.DataFrame(rows).to_csv(DS / "hardware_table.csv", index=False)

    # confusion figures from XGBoost OOF (groupkfold), if available
    if ("XGBoost", "detection") in oof_store:
        for task, labels, cmap, fname in [
            ("detection", ["Healthy", "Faulty"], "Blues", "fig_hw_confusion_detection"),
            ("severity", SEV_LABELS, "Purples", "fig_hw_confusion_severity"),
            ("phase", ["A", "B", "C"], "Greens", "fig_hw_confusion_phase")]:
            if ("XGBoost", task) not in oof_store:
                continue
            sub, oof, ycol = oof_store[("XGBoost", task)]
            mask = oof >= 0
            cm = confusion_matrix(sub[ycol].to_numpy(int)[mask], oof[mask], labels=list(range(len(labels))))
            cm_fig(cm, labels, f"Hardware {task} (XGBoost, GK)", fname, cmap)
        print("saved hardware confusion figures")
    print("\nsaved dataset/hardware_results.json + dataset/hardware_table.csv")


if __name__ == "__main__":
    main()
