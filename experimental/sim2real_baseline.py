r"""Naive simulation->experiment transfer baseline (NO domain adaptation): train on the full
PSCAD simulation, test on the real ibarram data. Quantifies the sim-to-real gap that PADA
(physics-anchored domain adaptation) must close. Detection and faulted-phase transfer directly;
severity scales differ between datasets so severity transfer is not evaluated here.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from feature_sets import all_features  # noqa: E402

sim = pd.read_parquet(ROOT / "dataset" / "features.parquet")
exp = pd.read_parquet(ROOT / "experimental" / "ibarram" / "features.parquet")
feats = [c for c in all_features(sim.columns) if c in exp.columns]
print(f"shared features: {len(feats)}  | sim windows {len(sim)}  exp windows {len(exp)}")

PMAP = {"A": 0, "B": 1, "C": 2}


def xgb(y):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, tree_method="hist",
                         scale_pos_weight=nneg / max(npos, 1), random_state=42)

# ---- Detection: train sim -> test experimental ----
Xs, ys = sim[feats].to_numpy(np.float32), sim["label"].to_numpy(int)
Xe, ye = exp[feats].to_numpy(np.float32), exp["label"].to_numpy(int)
m = xgb(ys).fit(Xs, ys)
pe = m.predict(Xe)
print("\n=== DETECTION (sim-trained, tested on real, NO adaptation) ===")
print(f"  accuracy={accuracy_score(ye, pe):.4f}  macro-F1={f1_score(ye, pe, average='macro'):.4f}")
print(f"  confusion [rows true H/F]: {confusion_matrix(ye, pe, labels=[0,1]).tolist()}")
# in-domain reference (5-fold-ish single split on sim already known ~0.95-0.98)
print("  (in-domain sim detection macro-F1 was ~0.95-0.98; gap = sim-to-real cost)")

# ---- Faulted phase: train sim faulty -> test exp faulty ----
sf = sim[sim.label == 1]; ef = exp[exp.label == 1]
Xsf = sf[feats].to_numpy(np.float32); ysf = sf["phase"].map(PMAP).to_numpy(int)
Xef = ef[feats].to_numpy(np.float32); yef = ef["phase"].map(PMAP).to_numpy(int)
mp = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, num_class=3, objective="multi:softprob",
                   eval_metric="mlogloss", n_jobs=-1, tree_method="hist", random_state=42).fit(Xsf, ysf)
pp = mp.predict(Xef)
print("\n=== FAULTED PHASE (sim-trained, tested on real, NO adaptation) ===")
print(f"  accuracy={accuracy_score(yef, pp):.4f}  macro-F1={f1_score(yef, pp, average='macro'):.4f}")
print(f"  confusion [rows true A/B/C]: {confusion_matrix(yef, pp, labels=[0,1,2]).tolist()}")
print("  (in-domain sim phase macro-F1 was 0.97-1.0; drop quantifies the sim-to-real gap)")
