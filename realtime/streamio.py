r"""
Shared helpers for the real-time stream: reconstruct a recording's continuous 3-phase signal
(to replay) and the CascadeInfer that loads the deployed bundle and classifies a window.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from features import build_feature_frame
from xgboost import XGBClassifier

EXP = ROOT / "experimental" / "ibarram"
DEPLOY = ROOT / "realtime" / "deploy"


def list_recordings():
    lab = pd.read_csv(EXP / "window_labels.csv")
    g = lab.groupby("case_id").agg(label=("label", "first"), severity_pct=("severity_pct", "first"),
                                   phase=("phase", "first"), rep=("case_id", "first"))
    g["rep"] = g.index.str.extract(r"Repetition(\d+)")[0].values
    return g.reset_index()


def reconstruct_recording(case_id):
    """Rebuild the continuous (3, L) normalised signal of a recording from its windows + src_start.
    Returns (signal, truth_dict)."""
    lab = pd.read_csv(EXP / "window_labels.csv").reset_index(drop=True)
    W = np.load(EXP / "windows.npy", mmap_mode="r")
    rows = lab.index[lab["case_id"] == case_id].to_numpy()
    if len(rows) == 0:
        raise ValueError(f"no recording {case_id}")
    starts = lab.loc[rows, "src_start"].to_numpy(int)
    Wlen = W.shape[2]; L = int(starts.max() + Wlen)
    sig = np.zeros((3, L), np.float32)
    for r, s in zip(rows, starts):
        sig[:, s:s + Wlen] = W[r]
    r0 = lab.loc[rows[0]]
    truth = {"case_id": case_id, "label": int(r0["label"]), "severity_pct": float(r0["severity_pct"]),
             "phase": (None if r0["label"] == 0 else str(r0["phase"]))}
    return sig, truth


class CascadeInfer:
    """Loads realtime/deploy/* and classifies a (3,W) window: onset -> detect -> severity -> phase."""

    def __init__(self, deploy=DEPLOY):
        self.meta = json.loads((deploy / "meta.json").read_text())
        self.fs = self.meta["fs"]; self.f0 = self.meta["f0"]
        self.allf = self.meta["features_all"]; self.sevf = self.meta["features_severity"]
        self.sev_levels = self.meta["sev_levels"]; self.phase_labels = self.meta["phase_labels"]
        self.onset = self.meta.get("onset")
        self.detect = XGBClassifier(); self.detect.load_model(str(deploy / "detect.json"))
        self.severity = XGBClassifier(); self.severity.load_model(str(deploy / "severity.json"))
        self.phase = XGBClassifier(); self.phase.load_model(str(deploy / "phase.json"))

    def _features(self, window):
        lab = pd.DataFrame({"case_id": [0], "label": [1]})
        fr = build_feature_frame(window[None].astype(np.float32), lab, fs=self.fs, f0=self.f0, verbose=False)
        return fr

    def classify(self, window):
        fr = self._features(window)
        out = {}
        if self.onset:
            v = float(fr[self.onset["feature"]].iloc[0])
            out["onset_index"] = v; out["onset"] = bool(v > self.onset["threshold"])
        det = int(self.detect.predict(fr[self.allf].to_numpy(np.float32))[0])
        out["state"] = "faulty" if det == 1 else "healthy"
        if det == 1:
            sr = int(self.severity.predict(fr[self.sevf].to_numpy(np.float32))[0])
            out["severity_pct"] = self.sev_levels[max(0, min(sr, len(self.sev_levels) - 1))]
            ph = int(self.phase.predict(fr[self.allf].to_numpy(np.float32))[0])
            out["phase"] = self.phase_labels[ph]
        return out
