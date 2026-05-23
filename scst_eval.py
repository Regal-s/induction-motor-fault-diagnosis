r"""Validate the SCST representation: train a compact 2D-CNN on SCST tensors vs the raw-phase
Stockwell control, under a group-aware split, for detection / faulted-phase / severity.
-> dataset/scst/scst_eval.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

torch.manual_seed(0); np.random.seed(0)
DS = Path(r"D:\Naveen") / "dataset" / "scst"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]; RANK = {s: i for i, s in enumerate(SEV)}
PMAP = {"A": 0, "B": 1, "C": 2}


class SmallCNN(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, out))

    def forward(self, x): return self.net(x)


def prep(X):
    X = np.log1p(X)                       # compress magnitude dynamic range
    mu = X.mean((0, 2, 3), keepdims=True); sd = X.std((0, 2, 3), keepdims=True) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def train_eval(X, y, tr, te, task, epochs=25, bs=64):
    out = 1 if task != "phase" else 3
    if task == "severity":
        out = 1
    m = SmallCNN(out)
    ds = TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr]))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    if task == "detect":
        npos = float((y[tr] == 1).sum()); nneg = float(len(tr) - npos)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / max(npos, 1)))
    elif task == "phase":
        lossf = nn.CrossEntropyLoss()
    else:
        lossf = nn.SmoothL1Loss()
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); o = m(xb)
            if task == "detect":
                loss = lossf(o.squeeze(-1), yb.float())
            elif task == "phase":
                loss = lossf(o, yb.long())
            else:
                loss = lossf(o.squeeze(-1), yb.float())
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o = m(torch.tensor(X[te])).numpy()
    if task == "detect":
        pred = (o.squeeze(-1) > 0).astype(int)
        return dict(macro_f1=round(float(f1_score(y[te], pred, average="macro")), 4),
                    acc=round(float(accuracy_score(y[te], pred)), 4))
    elif task == "phase":
        pred = o.argmax(1)
        return dict(macro_f1=round(float(f1_score(y[te], pred, average="macro")), 4),
                    acc=round(float(accuracy_score(y[te], pred)), 4))
    else:
        pr = np.clip(np.round(o.squeeze(-1)), 0, 6).astype(int)
        return dict(within1=round(float((np.abs(pr - y[te]) <= 1).mean()), 4),
                    exact=round(float((pr == y[te]).mean()), 4))


def split(lab, mask, ycol):
    idx = np.where(mask)[0]
    g = lab.loc[idx, "group_id"].to_numpy()
    y = lab.loc[idx, ycol].to_numpy()
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(idx, y, g))
    return idx[tr], idx[te]


def main():
    scst = np.load(DS / "scst_X.npy"); raw = np.load(DS / "rawst_X.npy")
    lab = pd.read_csv(DS / "scst_labels.csv", low_memory=False)
    Xs, Xr = prep(scst), prep(raw)
    res = {}

    # detection (all)
    yd = lab["label"].to_numpy(int)
    tr, te = split(lab, np.ones(len(lab), bool), "label")
    res["detection"] = {"SCST": train_eval(Xs, yd, tr, te, "detect"),
                        "raw-phase": train_eval(Xr, yd, tr, te, "detect")}
    # faulty subset for phase & severity
    fmask = (lab["label"] == 1).to_numpy()
    yp = lab["phase"].map(PMAP).fillna(-1).to_numpy(int)
    trp, tep = split(lab, fmask, "phase")
    res["phase"] = {"SCST": train_eval(Xs, yp, trp, tep, "phase"),
                    "raw-phase": train_eval(Xr, yp, trp, tep, "phase")}
    yr = lab["severity_pct"].map(RANK).fillna(-1).to_numpy(int)
    res["severity"] = {"SCST": train_eval(Xs, yr, trp, tep, "severity"),
                       "raw-phase": train_eval(Xr, yr, trp, tep, "severity")}

    print(json.dumps(res, indent=2))
    json.dump(res, open(DS / "scst_eval.json", "w"), indent=2)
    print("\nsaved", DS / "scst_eval.json")


if __name__ == "__main__":
    main()
