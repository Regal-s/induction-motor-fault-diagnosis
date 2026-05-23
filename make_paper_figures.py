r"""
Generate publication figures as EDITABLE SVG (text kept as text -> Visio/Inkscape editable),
saved in D:\Naveen\figures_svg\. Uses real data/metrics throughout.

Figures:
  fig_confusion_*.svg     confusion matrices (detection / severity / phase)
  fig_roc.svg             detection ROC (overall + per-load) from OOF probabilities
  fig_metrics.svg         precision/recall/F1/accuracy per stage
  fig_model_comparison.svg  baseline ML/DL models on detection (group split)
  fig_learning_xgb.svg    XGBoost convergence (train/val logloss vs boosting round)
  fig_learning_cnn.svg    1D-CNN learning curves (train loss / val macro-F1 vs epoch)
  fig_inference_time.svg  inference time + model size (feature+XGBoost vs CNN)
  fig_noise_robustness.svg  metric vs SNR (robustness)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"svg.fonttype": "none", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "figure.dpi": 120, "lines.linewidth": 1.1})
COL, DCOL = 3.45, 7.16  # IEEE single- and double-column widths (inches)

from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_fscore_support, accuracy_score, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from feature_sets import all_features  # noqa: E402
DS = ROOT / "dataset"
OUT = ROOT / "figures_svg"; OUT.mkdir(exist_ok=True)
SEVL = [0.3, 0.5, 1, 2, 3, 4, 5]; PH = ["A", "B", "C"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    pdfdir = ROOT / "manuscript" / "figs"; pdfdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdfdir / name.replace(".svg", ".pdf"), format="pdf", bbox_inches="tight")  # vector for LaTeX
    plt.close(fig); print("saved", name)


def cm_fig(cm, labels, title, name, cmap):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(0.42*len(labels)+1.7, 0.42*len(labels)+1.5))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    th = cm.max()/2
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(int(v)), ha="center", va="center",
                color="white" if v > th else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04); save(fig, name)


def main():
    df = pd.read_parquet(DS / "features.parquet")
    feats = all_features(df.columns)
    X = df[feats].to_numpy(np.float32); y = df["label"].to_numpy(int)
    groups = df["group_id"].to_numpy()

    # ---------- confusion matrices (from stored metrics) ----------
    m1 = json.load(open(DS/"stage1_metrics.json")); cm_fig(m1["groupkfold"]["confusion"],
        ["Healthy", "Faulty"], "Detection (GroupKFold)", "fig_confusion_detection.svg", "Blues")
    m2 = json.load(open(DS/"stage2_metrics.json")); cm_fig(m2["groupkfold"]["confusion"],
        [str(s) for s in SEVL], "Severity % (GroupKFold)", "fig_confusion_severity.svg", "Purples")
    m3 = json.load(open(DS/"stage3_metrics.json")); cm_fig(m3["groupkfold"]["confusion"],
        PH, "Faulted phase (GroupKFold)", "fig_confusion_phase.svg", "Greens")

    # ---------- ROC (detection) overall + per-load, from OOF probs ----------
    oof = pd.read_csv(DS/"stage1_oof_predictions.csv")
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    fpr, tpr, _ = roc_curve(oof["label"], oof["prob_faulty"]); A = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, color="k", label=f"Overall (AUC={A:.3f})")
    for L in ["NL", "20", "40", "60", "80", "100"]:
        s = oof[oof["load_label"].astype(str) == L]
        if s["label"].nunique() == 2:
            f_, t_, _ = roc_curve(s["label"], s["prob_faulty"])
            ax.plot(f_, t_, lw=1.2, alpha=0.8, label=f"load {L} (AUC={auc(f_,t_):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate"); ax.set_title("Detection ROC (OOF, overall + per-load)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3); save(fig, "fig_roc.svg")

    # ---------- performance metrics per stage ----------
    pd1 = precision_recall_fscore_support(oof["label"], oof["pred"], average="macro", zero_division=0)
    s2 = pd.read_csv(DS/"stage2_oof_predictions.csv")
    sev_p = precision_recall_fscore_support(s2["true_rank"], s2["pred_rank"], average="macro", zero_division=0)
    sev_acc = accuracy_score(s2["true_rank"], s2["pred_rank"])
    sev_w1 = float((np.abs(s2["pred_rank"]-s2["true_rank"]) <= 1).mean())
    s3 = pd.read_csv(DS/"stage3_oof_predictions.csv")
    ph_p = precision_recall_fscore_support(s3["true_phase"], s3["pred_phase"], average="macro", zero_division=0)
    ph_acc = accuracy_score(s3["true_phase"], s3["pred_phase"])
    metrics = {"Precision": [pd1[0], sev_p[0], ph_p[0]], "Recall": [pd1[1], sev_p[1], ph_p[1]],
               "F1": [pd1[2], sev_p[2], ph_p[2]],
               "Accuracy": [accuracy_score(oof["label"], oof["pred"]), sev_acc, ph_acc]}
    stages = ["Detection", "Severity", "Phase"]; x = np.arange(3); w = 0.2
    fig, ax = plt.subplots(figsize=(DCOL, 3.1))
    for i, (k, v) in enumerate(metrics.items()):
        b = ax.bar(x + (i-1.5)*w, v, w, label=k)
        for r in b: ax.text(r.get_x()+w/2, r.get_height()+0.005, f"{r.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, stages); ax.set_ylim(0, 1.05); ax.set_ylabel("score (macro)")
    ax.set_title(f"Per-stage metrics (GroupKFold OOF); severity within-1={sev_w1:.2f}")
    ax.legend(ncol=4, fontsize=8, loc="lower center"); ax.grid(axis="y", alpha=0.3); save(fig, "fig_metrics.svg")

    # ---------- baseline model comparison (detection): 5-fold mean macro-F1 ----------
    from sklearn.base import clone
    def factory(nm):
        return {
            "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
            "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
            "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=300, random_state=0)),
            "RandomForest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0),
            "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=0),
        }[nm]
    names0 = ["LogReg", "SVM-RBF", "KNN", "MLP", "RandomForest", "XGBoost"]
    fold_f1 = {nm: [] for nm in names0}; fold_acc = {nm: [] for nm in names0}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        sub = np.random.RandomState(0).choice(tr, min(6000, len(tr)), replace=False)
        for nm in names0:
            mdl = clone(factory(nm))
            if nm == "XGBoost":
                mdl.set_params(scale_pos_weight=(y[tr]==0).sum()/max((y[tr]==1).sum(), 1))
            Xt, yt = (X[sub], y[sub]) if nm in ("SVM-RBF", "KNN", "MLP") else (X[tr], y[tr])
            mdl.fit(Xt, yt); p = mdl.predict(X[te])
            fold_f1[nm].append(f1_score(y[te], p, average="macro")); fold_acc[nm].append(accuracy_score(y[te], p))
    comp = {}
    for nm in names0:
        comp[nm] = (float(np.mean(fold_f1[nm])), float(np.mean(fold_acc[nm])))
        print(f"  {nm}: 5-fold F1={comp[nm][0]:.3f}+/-{np.std(fold_f1[nm]):.3f}")
    # add DL from stored results
    try: comp["ResNet-1D"] = (json.load(open(DS/"dl_metrics.json"))["stage1_grouped"]["macro_f1"], json.load(open(DS/"dl_metrics.json"))["stage1_grouped"]["acc"])
    except Exception: pass
    try:
        pj = json.load(open(DS/"pcmnet_results.json")); comp["PCM-Net"] = (pj["group_film=True"]["detect_f1"], pj["group_film=True"].get("acc", np.nan))
    except Exception: pass
    names = list(comp); f1s = [comp[n][0] for n in names]
    order = np.argsort(f1s); names = [names[i] for i in order]; f1s = [f1s[i] for i in order]
    fig, ax = plt.subplots(figsize=(COL, 3.4))
    cols = ["#4C78A8" if n not in ("ResNet-1D", "PCM-Net") else "#F58518" for n in names]
    b = ax.barh(names, f1s, color=cols)
    for r in b: ax.text(r.get_width()+0.005, r.get_y()+r.get_height()/2, f"{r.get_width():.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Detection macro-F1 (group split)")
    ax.set_title("Baseline model comparison (blue=feature ML, orange=deep)"); ax.grid(axis="x", alpha=0.3)
    save(fig, "fig_model_comparison.svg")
    json.dump({n: dict(f1=float(comp[n][0]), acc=float(comp[n][1])) for n in comp},
              open(OUT/"model_comparison.json", "w"), indent=2)

    # ---------- XGBoost convergence ----------
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, tree_method="hist", eval_metric="logloss",
                        scale_pos_weight=(y[tr]==0).sum()/max((y[tr]==1).sum(),1), random_state=0)
    xgb.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[te], y[te])], verbose=False)
    ev = xgb.evals_result()
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.plot(ev["validation_0"]["logloss"], label="train")
    ax.plot(ev["validation_1"]["logloss"], label="validation")
    ax.set_xlabel("boosting round"); ax.set_ylabel("log-loss"); ax.set_title("XGBoost detection convergence")
    ax.legend(); ax.grid(alpha=0.3); save(fig, "fig_learning_xgb.svg")

    # ---------- CNN learning curves ----------
    import torch, torch.nn as nn
    Xw = np.asarray(np.load(DS/"windows.npy"), np.float32)
    n = Xw.shape[2]//8; Xd = Xw[:, :, :n*8].reshape(Xw.shape[0], 3, n, 8).mean(-1).astype(np.float32)
    mu, sd = Xd[tr].mean((0,2), keepdims=True), Xd[tr].std((0,2), keepdims=True)+1e-6
    Xd = (Xd-mu)/sd
    net = nn.Sequential(nn.Conv1d(3,16,7,2,3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(16,32,5,2,2), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32,1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    posw = torch.tensor((y[tr]==0).sum()/max((y[tr]==1).sum(),1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    Xtr_t = torch.tensor(Xd[tr]); ytr_t = torch.tensor(y[tr], dtype=torch.float32)
    Xte_t = torch.tensor(Xd[te])
    bs = 256; tr_loss, va_f1 = [], []
    for ep in range(20):
        net.train(); perm = torch.randperm(len(Xtr_t)); el = 0
        for i in range(0, len(Xtr_t), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            o = net(Xtr_t[idx]).squeeze(-1); l = lossf(o, ytr_t[idx]); l.backward(); opt.step()
            el += l.item()*len(idx)
        tr_loss.append(el/len(Xtr_t))
        net.eval()
        with torch.no_grad():
            pv = (net(Xte_t).squeeze(-1).numpy() > 0).astype(int)
        va_f1.append(f1_score(y[te], pv, average="macro"))
    fig, ax = plt.subplots(figsize=(COL, 2.7)); ax2 = ax.twinx()
    ax.plot(tr_loss, "b-", label="train loss"); ax2.plot(va_f1, "g-", label="val macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss", color="b"); ax2.set_ylabel("val macro-F1", color="g")
    ax.set_title("1D-CNN learning curves (detection)"); ax.grid(alpha=0.3); save(fig, "fig_learning_cnn.svg")

    # ---------- inference time / efficiency ----------
    from features import build_feature_frame
    nb = 2000; Xb = Xw[:nb]; lb = df.iloc[:nb][["case_id", "label"]]
    t0 = time.perf_counter(); _ = build_feature_frame(Xb, lb); t_feat = (time.perf_counter()-t0)/nb*1e3
    Xf2 = X[:nb]; t0 = time.perf_counter(); _ = xgb.predict(Xf2); t_xgb = (time.perf_counter()-t0)/nb*1e3
    with torch.no_grad():
        t0 = time.perf_counter(); _ = net(torch.tensor(Xd[:nb])); t_cnn = (time.perf_counter()-t0)/nb*1e3
    cnn_params = sum(p.numel() for p in net.parameters())
    eff = {"Feature extract": t_feat, "XGBoost infer": t_xgb,
           "Feat+XGB total": t_feat+t_xgb, "1D-CNN infer": t_cnn}
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    b = ax.bar(list(eff), list(eff.values()), color=["#888", "#4C78A8", "#2a9d8f", "#F58518"])
    for r in b: ax.text(r.get_x()+r.get_width()/2, r.get_height()*1.02, f"{r.get_height():.3f}", ha="center", fontsize=8)
    ax.set_ylabel("ms per window (CPU)")
    ax.set_title(f"Inference efficiency  (XGB ~{xgb.n_estimators} trees; CNN {cnn_params/1e3:.1f}k params)")
    ax.grid(axis="y", alpha=0.3); save(fig, "fig_inference_time.svg")
    json.dump({**eff, "cnn_params": cnn_params}, open(OUT/"efficiency.json", "w"), indent=2)

    # ---------- noise robustness ----------
    nr = json.load(open(DS/"noise_robustness.json"))
    conds = ["clean", "40", "30", "20"]; xi = [60, 40, 30, 20]
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    for key, lbl in [("detect_f1", "Detection F1"), ("severity_within1", "Severity within-1"), ("phase_acc", "Phase acc")]:
        ax.plot(xi, [nr[c][key] for c in conds], "o-", label=lbl)
    ax.invert_xaxis(); ax.set_xlabel("SNR (dB)  [clean=60]"); ax.set_ylabel("score")
    ax.set_title("Noise robustness"); ax.grid(alpha=0.3); ax.legend(); save(fig, "fig_noise_robustness.svg")

    print("\nALL FIGURES saved to", OUT)


if __name__ == "__main__":
    main()
