r"""
Phase 3b — ResNet-1D deep baseline (Track B) on raw 3-phase current windows,
compared against the XGBoost (Track A) feature-based models.

Windows are decimated 8x (anti-aliased mean-pool: 2000->250 samples) for CPU speed.
Evaluated on a grouped train/test split for all 3 stages (+ one Leave-One-Load-Out
fold, load=20, for detection). Saves dataset/dl_metrics.json.

Run:  python dl_resnet1d.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from splits import stratified_group_folds, leave_one_load_out

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(max(1, torch.get_num_threads()))
DEV = "cpu"
DS = Path(r"D:\Naveen") / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RANK = {s: i for i, s in enumerate(SEV)}
PMAP = {"A": 0, "B": 1, "C": 2}
DECIM = 8


def load():
    X = np.asarray(np.load(DS / "windows.npy"), dtype=np.float32)   # (N,3,2000)
    n = X.shape[2] // DECIM
    X = X[:, :, :n * DECIM].reshape(X.shape[0], 3, n, DECIM).mean(-1)  # anti-aliased decimate -> (N,3,250)
    lab = pd.read_csv(DS / "window_labels.csv", low_memory=False)
    return X.astype(np.float32), lab


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm1d(cout)
        self.sc = nn.Sequential() if (stride == 1 and cin == cout) else \
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        o = self.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return self.relu(o + self.sc(x))


class ResNet1D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, 16, 7, 2, 3, bias=False), nn.BatchNorm1d(16),
                                  nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.s1 = Block(16, 16); self.s2 = Block(16, 32, 2); self.s3 = Block(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.stem(x); x = self.s1(x); x = self.s2(x); x = self.s3(x)
        return self.fc(self.pool(x).squeeze(-1))


def standardize(Xtr, Xte):
    mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train(model, Xtr, ytr, task, epochs=14, bs=256, lr=1e-3):
    model.to(DEV)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    # 15% internal val for early stopping
    nval = max(1, int(0.15 * len(ds)))
    tr_ds, va_ds = torch.utils.data.random_split(ds, [len(ds) - nval, nval],
                                                 generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    vl = DataLoader(va_ds, batch_size=512)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if task == "binary":
        npos = float(ytr.sum()); nneg = float(len(ytr) - npos)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / max(npos, 1)))
    elif task == "multi":
        lossf = nn.CrossEntropyLoss()
    else:
        lossf = nn.SmoothL1Loss()
    best, best_state, patience = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); out = model(xb)
            if task == "binary":
                loss = lossf(out.squeeze(-1), yb.float())
            elif task == "multi":
                loss = lossf(out, yb.long())
            else:
                loss = lossf(out.squeeze(-1), yb.float())
            loss.backward(); opt.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb)
                if task == "binary":
                    vloss += lossf(out.squeeze(-1), yb.float()).item() * len(yb)
                elif task == "multi":
                    vloss += lossf(out, yb.long()).item() * len(yb)
                else:
                    vloss += lossf(out.squeeze(-1), yb.float()).item() * len(yb)
        vloss /= len(va_ds)
        if vloss < best - 1e-4:
            best, best_state, patience = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 3:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, X, bs=512):
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(torch.tensor(X[i:i + bs])).cpu().numpy())
    return np.concatenate(outs)


def main():
    X, lab = load()
    print(f"decimated windows: {X.shape}")
    label = lab["label"].to_numpy(int)
    rank = lab["severity_pct"].map(RANK).fillna(-1).to_numpy(int)
    phase = lab["phase"].map(PMAP).fillna(-1).to_numpy(int)
    df = pd.read_parquet(DS / "features.parquet")  # for the same group structure
    res = {}

    tr, te = next(stratified_group_folds(df))   # one grouped split

    # ---- Stage 1: detection (grouped split) ----
    Xtr, Xte = standardize(X[tr], X[te])
    m = train(ResNet1D(1), Xtr, label[tr], "binary")
    pl = (predict(m, Xte).squeeze(-1) > 0).astype(int)
    res["stage1_grouped"] = dict(macro_f1=float(f1_score(label[te], pl, average="macro")),
                                 acc=float(accuracy_score(label[te], pl)))
    print(f"Stage1 (grouped): F1={res['stage1_grouped']['macro_f1']:.4f}")

    # ---- Stage 1: one LOLO fold (hold out load 20) ----
    for L, ltr, lte in leave_one_load_out(df):
        if L != "20":
            continue
        Xltr, Xlte = standardize(X[ltr], X[lte])
        m = train(ResNet1D(1), Xltr, label[ltr], "binary")
        pl = (predict(m, Xlte).squeeze(-1) > 0).astype(int)
        res["stage1_lolo20"] = dict(macro_f1=float(f1_score(label[lte], pl, average="macro")),
                                    acc=float(accuracy_score(label[lte], pl)))
        print(f"Stage1 (LOLO load=20): F1={res['stage1_lolo20']['macro_f1']:.4f}")

    # faulty-only mask for stages 2/3
    ftr = tr[label[tr] == 1]; fte = te[label[te] == 1]

    # ---- Stage 3: phase (grouped) ----
    Xtr, Xte = standardize(X[ftr], X[fte])
    m = train(ResNet1D(3), Xtr, phase[ftr], "multi")
    pp = predict(m, Xte).argmax(1)
    res["stage3_grouped"] = dict(acc=float(accuracy_score(phase[fte], pp)),
                                 macro_f1=float(f1_score(phase[fte], pp, average="macro")))
    print(f"Stage3 (grouped): acc={res['stage3_grouped']['acc']:.4f}")

    # ---- Stage 2: severity regression (grouped) ----
    Xtr, Xte = standardize(X[ftr], X[fte])
    m = train(ResNet1D(1), Xtr, rank[ftr].astype(np.float32), "reg")
    pr = np.clip(np.round(predict(m, Xte).squeeze(-1)), 0, 6).astype(int)
    res["stage2_grouped"] = dict(within1=float((np.abs(pr - rank[fte]) <= 1).mean()),
                                 exact=float((pr == rank[fte]).mean()))
    print(f"Stage2 (grouped): within-1={res['stage2_grouped']['within1']:.4f}")

    res["note"] = ("ResNet-1D, decimated 8x (250 samples). Compare to XGBoost GroupKFold: "
                   "Stage1 F1 0.977, Stage2 within-1 0.990, Stage3 acc 0.972; LOLO load20 detect F1 0.749.")
    with open(DS / "dl_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {DS/'dl_metrics.json'}\n{json.dumps(res, indent=2)}")


if __name__ == "__main__":
    main()
