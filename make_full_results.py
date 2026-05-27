r"""
Comprehensive results package -> editable SVG (Visio/Inkscape: svg.fonttype='none') in results_svg/,
plus a metrics table (results_svg/RESULTS.md + full_metrics.csv).

Generates:
  * Precision / Recall / F1 / Accuracy for ALL models, per task (detection/severity/phase) -> table + bar SVGs.
  * Confusion matrices (deployed XGBoost OOF): detection 2x2, severity 7x7, phase 3x3.
  * ROC curves for ALL CLASSES: detection (binary + per-load), severity (7 one-vs-rest), phase (3 one-vs-rest).
  * Convergence curves with rounds/epochs: XGBoost train/val log-loss per stage; 1-D CNN train-loss/val-F1 per epoch.
  * Robustness: macro metrics vs SNR (noise), per-held-out-load detection (LOLO), Stage-1 calibration reliability.
All on identical leakage-safe StratifiedGroupKFold OOF. Feature/foundation models computed fresh; xLSTM/ResNet
pulled from the stored benchmark (deep re-run is hours) and clearly tagged.

Run:  python make_full_results.py   (set TABPFN_TOKEN to include TabPFN-2.5)
"""
from __future__ import annotations
import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 120})

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
OUT = ROOT / "results_svg"; OUT.mkdir(exist_ok=True)
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from splits import stratified_group_folds
from feature_sets import all_features, severity_features
from benchmark_models import make_feature_model, make_tabpfn, _subsample

SEV_RANK = {0.3: 0, 0.5: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
SEVL = ["0.3", "0.5", "1", "2", "3", "4", "5"]; PH = ["A", "B", "C"]
PHASE_RANK = {"A": 0, "B": 1, "C": 2}
FEATURE_MODELS = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest", "XGBoost"]
TASKS = {"detection": ("all", "label", 2, ["Healthy", "Faulty"]),
         "severity":  ("severity", "_sevrank", 7, SEVL),
         "phase":     ("all", "_phrank", 3, PH)}


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / (name + ".svg"), format="svg", bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  saved", name + ".svg")


def load_data():
    df = pd.read_parquet(DS / "features.parquet").reset_index(drop=True)
    df["_sevrank"] = df["severity_pct"].map(SEV_RANK); df["_phrank"] = df["phase"].map(PHASE_RANK)
    return df


def oof_predict(df, name, task, want_proba=False):
    feat_set, ycol, nc, _ = TASKS[task]
    sub = df[df["label"] == 1].reset_index(drop=True) if task != "detection" else df
    cols = (all_features(sub.columns) if feat_set == "all"
            else severity_features(sub.columns, load_aware=True))
    cols = [c for c in cols if c not in ("_sevrank", "_phrank") and pd.api.types.is_numeric_dtype(sub[c])]
    X = sub[cols].to_numpy(np.float32); y = sub[ycol].to_numpy(int)
    pred = np.full(len(sub), -1, int); proba = np.full((len(sub), nc), np.nan)
    for tr, te in stratified_group_folds(sub, y_col=ycol, n_splits=5):
        tru = _subsample(name, tr, y, False)
        mdl = make_tabpfn(False) if name == "TabPFN-2.5" else make_feature_model(name, y[tru], False)
        mdl.fit(X[tru], y[tru]); pred[te] = mdl.predict(X[te])
        if want_proba:
            try: proba[te] = mdl.predict_proba(X[te])
            except Exception: pass
    return y, pred, proba, sub


def macro_prf(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(precision=float(pr), recall=float(rc), f1=float(f1), accuracy=float(accuracy_score(y, p)))


def cm_fig(cm, labels, title, name, cmap):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 2.0, 0.5 * len(labels) + 1.8))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    th = cm.max() / 2 if cm.max() else 1
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(int(v)), ha="center", va="center", color="white" if v > th else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04); save(fig, name)


def roc_multiclass(y, proba, labels, title, name):
    Yb = label_binarize(y, classes=list(range(len(labels))))
    if Yb.shape[1] == 1:  # binary edge
        Yb = np.hstack([1 - Yb, Yb])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for k, lab in enumerate(labels):
        if np.isnan(proba[:, k]).all() or Yb[:, k].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(Yb[:, k], proba[:, k]); A = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.4, label=f"{lab} (AUC={A:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3); save(fig, name)


def main():
    df = load_data()
    have_tabpfn = bool(os.environ.get("TABPFN_TOKEN"))
    models = FEATURE_MODELS + (["TabPFN-2.5"] if have_tabpfn else [])

    # ---------- 1. all-model P/R/F1/Acc ----------
    metrics = {t: {} for t in TASKS}
    xgb_oof = {}
    for task in TASKS:
        print(f"[metrics] {task}")
        for name in models:
            wp = (name == "XGBoost")
            y, p, pr, sub = oof_predict(df, name, task, want_proba=wp)
            metrics[task][name] = macro_prf(y, p)
            if wp:
                xgb_oof[task] = (y, p, pr, sub)
    # pull deep models from stored benchmark (macro-F1 + acc only)
    try:
        bench = json.load(open(DS / "benchmark_results.json"))["results"]
        for dm in ["xLSTM"]:
            for task in TASKS:
                gk = bench.get(dm, {}).get(task, {}).get("groupkfold")
                if gk:
                    metrics[task][dm] = {"precision": np.nan, "recall": np.nan,
                                         "f1": gk["macro_f1"], "accuracy": gk["acc"]}
    except Exception:
        pass

    # metrics table (md + csv)
    rows = []
    for task in TASKS:
        for name, m in metrics[task].items():
            rows.append(dict(task=task, model=name, **{k: round(v, 4) for k, v in m.items()}))
    pd.DataFrame(rows).to_csv(OUT / "full_metrics.csv", index=False)
    L = ["# All-model metrics (StratifiedGroupKFold OOF): Precision / Recall / F1 / Accuracy", ""]
    for task in TASKS:
        L += [f"## {task.capitalize()}", "", "| Model | Precision | Recall | F1 | Accuracy |", "|---|---|---|---|---|"]
        for name, m in metrics[task].items():
            f = lambda v: f"{v:.3f}" if v == v else "—"   # NaN -> em-dash
            L.append(f"| {name} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['accuracy'])} |")
        L.append("")
    L += ["*xLSTM from stored benchmark (macro-F1 / accuracy; per-class P/R not re-computed — deep re-run is hours). "
          "Severity scored as 7-level classification here for P/R/F1; the deployed model reports within-1/MAE.*"]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("saved results_svg/RESULTS.md + full_metrics.csv")

    # P/R/F1/Acc grouped bars per task
    for task in TASKS:
        names = list(metrics[task]); mk = ["precision", "recall", "f1", "accuracy"]
        x = np.arange(len(names)); w = 0.2
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 3.6))
        for i, k in enumerate(mk):
            vals = [metrics[task][n][k] for n in names]
            ax.bar(x + (i - 1.5) * w, vals, w, label=k.capitalize())
        ax.set_xticks(x, names, rotation=30, ha="right"); ax.set_ylim(0, 1.05)
        ax.set_ylabel("score (macro)"); ax.set_title(f"{task.capitalize()}: Precision/Recall/F1/Accuracy by model")
        ax.legend(ncol=4, fontsize=7.5); ax.grid(axis="y", alpha=0.3); save(fig, f"fig_prf_{task}")

    # ---------- 2. confusion matrices (deployed XGBoost OOF) ----------
    for task, cmap in [("detection", "Blues"), ("severity", "Purples"), ("phase", "Greens")]:
        y, p, _, _ = xgb_oof[task]; _, _, _, labels = TASKS[task]
        cm = confusion_matrix(y, p, labels=list(range(len(labels))))
        cm_fig(cm, labels, f"{task.capitalize()} confusion (XGBoost, GroupKFold OOF)", f"fig_confusion_{task}", cmap)

    # ---------- 3. ROC for all classes ----------
    # detection: binary overall + per-load (from stored OOF probs)
    try:
        oof = pd.read_csv(DS / "stage1_oof_predictions.csv")
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        fpr, tpr, _ = roc_curve(oof["label"], oof["prob_faulty"]); A = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.4, color="k", label=f"Overall (AUC={A:.3f})")
        for Ld in ["NL", "20", "40", "60", "80", "100"]:
            s = oof[oof["load_label"].astype(str) == Ld]
            if s["label"].nunique() == 2:
                f_, t_, _ = roc_curve(s["label"], s["prob_faulty"])
                ax.plot(f_, t_, lw=1.1, alpha=0.85, label=f"load {Ld} (AUC={auc(f_,t_):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate"); ax.set_title("Detection ROC (overall + per-load)")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3); save(fig, "fig_roc_detection")
    except Exception as e:
        print("detection ROC skipped:", e)
    # severity (7 OVR) + phase (3 OVR) from XGBoost OOF proba
    for task, title in [("severity", "Severity ROC (one-vs-rest, 7 classes)"),
                        ("phase", "Faulted-phase ROC (one-vs-rest)")]:
        y, p, pr, _ = xgb_oof[task]; _, _, _, labels = TASKS[task]
        if not np.isnan(pr).all():
            roc_multiclass(y, pr, labels, title, f"fig_roc_{task}")

    # ---------- 4. convergence curves (XGBoost train/val log-loss per stage) ----------
    from xgboost import XGBClassifier
    for task in ["detection", "severity", "phase"]:
        feat_set, ycol, nc, _ = TASKS[task]
        sub = df[df["label"] == 1].reset_index(drop=True) if task != "detection" else df
        cols = (all_features(sub.columns) if feat_set == "all"
                else severity_features(sub.columns, load_aware=True))
        cols = [c for c in cols if c not in ("_sevrank", "_phrank") and pd.api.types.is_numeric_dtype(sub[c])]
        X = sub[cols].to_numpy(np.float32); y = sub[ycol].to_numpy(int)
        tr, te = next(stratified_group_folds(sub, y_col=ycol, n_splits=5))
        em = "logloss" if nc == 2 else "mlogloss"
        clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1, tree_method="hist", eval_metric=em,
                            objective=("binary:logistic" if nc == 2 else "multi:softprob"),
                            num_class=(None if nc == 2 else nc), random_state=0)
        clf.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[te], y[te])], verbose=False)
        ev = clf.evals_result()
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(ev["validation_0"][em], label="train")
        ax.plot(ev["validation_1"][em], label="validation")
        ax.set_xlabel("boosting round (epoch)"); ax.set_ylabel(em)
        ax.set_title(f"{task.capitalize()} XGBoost convergence"); ax.legend(); ax.grid(alpha=0.3)
        save(fig, f"fig_convergence_{task}")

    # ---------- 5. deep train/val learning curve (1-D CNN) ----------
    try:
        import torch, torch.nn as nn
        from sklearn.metrics import f1_score
        Xw = np.asarray(np.load(DS / "windows.npy"), np.float32)
        n = Xw.shape[2] // 8; Xd = Xw[:, :, :n*8].reshape(Xw.shape[0], 3, n, 8).mean(-1).astype(np.float32)
        y = df["label"].to_numpy(int)
        tr, te = next(stratified_group_folds(df, y_col="label", n_splits=5))
        mu, sd = Xd[tr].mean((0, 2), keepdims=True), Xd[tr].std((0, 2), keepdims=True) + 1e-6
        Xd = (Xd - mu) / sd
        net = nn.Sequential(nn.Conv1d(3, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                            nn.Conv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
                            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        posw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
        Xtr_t = torch.tensor(Xd[tr]); ytr_t = torch.tensor(y[tr], dtype=torch.float32); Xte_t = torch.tensor(Xd[te])
        trl, vf1 = [], []
        for ep in range(25):
            net.train(); perm = torch.randperm(len(Xtr_t)); el = 0.0
            for i in range(0, len(Xtr_t), 256):
                idx = perm[i:i+256]; opt.zero_grad()
                o = net(Xtr_t[idx]).squeeze(-1); l = lossf(o, ytr_t[idx]); l.backward(); opt.step()
                el += l.item() * len(idx)
            trl.append(el / len(Xtr_t))
            net.eval()
            with torch.no_grad():
                pv = (net(Xte_t).squeeze(-1).numpy() > 0).astype(int)
            vf1.append(f1_score(y[te], pv, average="macro"))
        fig, ax = plt.subplots(figsize=(5.2, 3.4)); ax2 = ax.twinx()
        ax.plot(trl, "b-", label="train loss"); ax2.plot(vf1, "g-", label="val macro-F1")
        ax.set_xlabel("epoch"); ax.set_ylabel("train loss", color="b"); ax2.set_ylabel("val macro-F1", color="g")
        ax.set_title("1-D CNN training/validation curve (detection)"); ax.grid(alpha=0.3)
        save(fig, "fig_learning_curve_deep")
    except Exception as e:
        print("deep learning curve skipped:", e)

    # ---------- 6. robustness ----------
    try:
        nr = json.load(open(DS / "noise_robustness.json")); conds = ["clean", "40", "30", "20"]; xi = [60, 40, 30, 20]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for key, lbl in [("detect_f1", "Detection F1"), ("severity_within1", "Severity within-1"), ("phase_acc", "Phase acc")]:
            ax.plot(xi, [nr[c][key] for c in conds], "o-", label=lbl)
        ax.invert_xaxis(); ax.set_xlabel("SNR (dB)  [clean=60]"); ax.set_ylabel("score")
        ax.set_title("Noise robustness"); ax.grid(alpha=0.3); ax.legend(); save(fig, "fig_robustness_noise")
    except Exception as e:
        print("noise fig skipped:", e)
    try:
        m1 = json.load(open(DS / "stage1_metrics.json")); pl = m1["lolo"]["per_load"]
        loads = list(pl); f1s = [pl[L]["macro_f1"] for L in loads]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        b = ax.bar(loads, f1s, color="#4C78A8")
        for r in b: ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.01, f"{r.get_height():.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_xlabel("held-out load"); ax.set_ylabel("detection macro-F1")
        ax.set_title("Per-held-out-load detection (LOLO)"); ax.grid(axis="y", alpha=0.3); save(fig, "fig_robustness_perload")
    except Exception as e:
        print("perload fig skipped:", e)
    try:
        oof = pd.read_csv(DS / "stage1_oof_predictions.csv")
        prob = oof["prob_faulty"].to_numpy(); yt = oof["label"].to_numpy()
        bins = np.linspace(0, 1, 11); idx = np.digitize(prob, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 0:
                xs.append(prob[m].mean()); ys.append(yt[m].mean())
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        ax.plot(xs, ys, "o-", label="Stage-1 detector")
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("empirical fraction faulty")
        ax.set_title("Calibration reliability (detection)"); ax.legend(); ax.grid(alpha=0.3); save(fig, "fig_calibration")
    except Exception as e:
        print("calibration fig skipped:", e)

    print(f"\nALL results -> {OUT}  (editable SVG + PNG; RESULTS.md; full_metrics.csv)")


if __name__ == "__main__":
    main()
