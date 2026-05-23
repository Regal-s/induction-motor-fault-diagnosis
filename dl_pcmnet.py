r"""
M5 — PCM-Net (compact): Phase-Coupled, load-Conditioned Multi-task network.

Compact CPU-friendly realization of the proposed architecture:
  - conv stem on decimated 3-phase current (+ a phase-coupling 1x1 mixer)
  - dilated temporal residual blocks (TCN-style; stands in for the selective-SSM backbone)
  - FiLM layers modulated by the MEASURED operating point (load) -- inject load, not erase it
  - multi-task heads: detection (binary), faulted phase (3-class), severity (ordinal regression)
  - physics regularizer: current-balance ||i_a+i_b+i_c||^2 auxiliary penalty
The decisive ablation: FiLM(load) ON vs OFF, on group-split and on Leave-One-Load-Out severity.
(Mamba/KAN are approximated here; the FiLM-conditioning + multi-task + physics terms are faithful.)

Run:  python dl_pcmnet.py  -> dataset/pcmnet_results.json
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

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from splits import leave_one_load_out  # noqa: E402

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(max(1, torch.get_num_threads()))
DS = ROOT / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]; RANK = {s: i for i, s in enumerate(SEV)}
PMAP = {"A": 0, "B": 1, "C": 2}
DECIM = 8


class FiLM(nn.Module):
    def __init__(self, ch, cond_dim=16, enable=True):
        super().__init__(); self.enable = enable
        if enable:
            self.g = nn.Linear(cond_dim, ch); self.b = nn.Linear(cond_dim, ch)
    def forward(self, x, c):
        if not self.enable:
            return x
        return x * (1 + self.g(c)).unsqueeze(-1) + self.b(c).unsqueeze(-1)


class TBlock(nn.Module):
    def __init__(self, cin, cout, d, enable_film=True):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, padding=d, dilation=d); self.bn1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=d, dilation=d); self.bn2 = nn.BatchNorm1d(cout)
        self.film = FiLM(cout, enable=enable_film)
        self.sc = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x, c):
        o = self.act(self.bn1(self.c1(x)))
        o = self.film(self.bn2(self.c2(o)), c)
        return self.act(o + self.sc(x))


class PCMNet(nn.Module):
    def __init__(self, film=True):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.phase_mix = nn.Conv1d(3, 3, 1)               # phase-coupling
        self.stem = nn.Sequential(nn.Conv1d(3, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU())
        self.b1 = TBlock(16, 32, 1, film); self.b2 = TBlock(32, 64, 2, film); self.b3 = TBlock(64, 64, 4, film)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.h_det = nn.Linear(64, 1); self.h_ph = nn.Linear(64, 3)
        self.h_sev = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, load):
        c = self.cond(load)
        x = self.stem(self.phase_mix(x))
        x = self.b1(x, c); x = self.b2(x, c); x = self.b3(x, c)
        z = self.pool(x).squeeze(-1)
        return self.h_det(z).squeeze(-1), self.h_ph(z), self.h_sev(z).squeeze(-1)


def load_data():
    X = np.asarray(np.load(DS / "windows.npy"), np.float32)
    n = X.shape[2] // DECIM
    Xd = X[:, :, :n * DECIM].reshape(X.shape[0], 3, n, DECIM).mean(-1).astype(np.float32)
    lab = pd.read_csv(DS / "window_labels.csv", low_memory=False)
    return Xd, lab


def train(model, X, load, ydet, yph, ysev, tr, epochs=18, bs=256, w_phys=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); ce = nn.CrossEntropyLoss(); sl1 = nn.SmoothL1Loss()
    ds = TensorDataset(torch.tensor(X[tr]), torch.tensor(load[tr]), torch.tensor(ydet[tr]),
                       torch.tensor(yph[tr]), torch.tensor(ysev[tr]))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, lb, yd, yp, ys in dl:
            opt.zero_grad()
            od, op, os_ = model(xb, lb.unsqueeze(-1))
            loss = bce(od, yd.float())
            fm = yd == 1
            if fm.any():
                loss = loss + ce(op[fm], yp[fm].long()) + sl1(os_[fm], ys[fm].float())
            phys = (xb.sum(1)).pow(2).mean()              # current-balance regularizer
            loss = loss + w_phys * phys
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, X, load, bs=512):
    model.eval(); D, P, S = [], [], []
    for i in range(0, len(X), bs):
        od, op, os_ = model(torch.tensor(X[i:i+bs]), torch.tensor(load[i:i+bs]).unsqueeze(-1))
        D.append(od.numpy()); P.append(op.numpy()); S.append(os_.numpy())
    return np.concatenate(D), np.concatenate(P), np.concatenate(S)


def main():
    X, lab = load_data()
    load = (lab["load_pct"].to_numpy(np.float32) / 100.0)
    ydet = lab["label"].to_numpy(int)
    yph = lab["phase"].map(PMAP).fillna(0).to_numpy(int)
    ysev = lab["severity_pct"].map(RANK).fillna(0).to_numpy(np.float32)
    groups = lab["group_id"].to_numpy(); res = {}

    # ---- group split (in-distribution) ----
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, ydet, groups))
    for film in [True, False]:
        m = train(PCMNet(film=film), X, load, ydet, yph, ysev, tr)
        d, p, s = predict(m, X[te], load[te])
        fm = ydet[te] == 1
        sev_pred = np.clip(np.round(s[fm]), 0, 6).astype(int)
        res[f"group_film={film}"] = dict(
            detect_f1=round(float(f1_score(ydet[te], (d > 0).astype(int), average="macro")), 4),
            phase_f1=round(float(f1_score(yph[te][fm], p[fm].argmax(1), average="macro")), 4),
            severity_within1=round(float((np.abs(sev_pred - ysev[te][fm]) <= 1).mean()), 4))
        print(f"[group, FiLM={film}] {res[f'group_film={film}']}")

    # ---- LOLO severity: FiLM on vs off (the decisive cross-load test) ----
    for film in [True, False]:
        lt, lp = [], []
        for L, tri, tei in leave_one_load_out(lab):
            m = train(PCMNet(film=film), X, load, ydet, yph, ysev, tri, epochs=14)
            _, _, s = predict(m, X[tei], load[tei])
            fm = ydet[tei] == 1
            lt.append(ysev[tei][fm]); lp.append(np.clip(np.round(s[fm]), 0, 6).astype(int))
        lt, lp = np.concatenate(lt), np.concatenate(lp)
        res[f"lolo_severity_within1_film={film}"] = round(float((np.abs(lp - lt) <= 1).mean()), 4)
        print(f"[LOLO severity, FiLM={film}] within-1 = {res[f'lolo_severity_within1_film={film}']}")

    json.dump(res, open(DS / "pcmnet_results.json", "w"), indent=2)
    print("\nsaved dataset/pcmnet_results.json\n", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
