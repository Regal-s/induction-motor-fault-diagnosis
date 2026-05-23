r"""
PC-Diff — Physics-Constrained conditional generative augmentation of 3-phase current.

A conditional VAE generates decimated 3-phase current windows conditioned on
(severity, phase, load). Two variants:
  - vanilla CVAE  (no physics)
  - PC-CVAE       (+ HARD Kirchhoff current-balance projection: i_a+i_b+i_c=0,
                   + SOFT negative-sequence-vs-severity consistency loss)
Validation: (1) physics validity of generated signals (current-balance residual; correlation of
generated |I2|/|I1| with conditioned severity); (2) downstream value -- under scarce real training
data, does augmenting with generated samples improve severity within-1, and does the physics-
constrained generator beat the vanilla one?  -> dataset/pcdiff_results.json

Run:  python pcdiff.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from features import build_feature_frame  # noqa: E402
from feature_sets import severity_features  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
DS = ROOT / "dataset"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]; RANK = {s: i for i, s in enumerate(SEV)}
PMAP = {"A": 0, "B": 1, "C": 2}
L, DECIM = 250, 8
FS_D, F0 = 25000.0 / DECIM, 50.0      # decimated sampling rate, fundamental
BIN = int(round(F0 * L / FS_D))       # fundamental FFT bin (=4)
A = np.exp(2j * np.pi / 3)
CDIM = 5                              # cond = [rank/6, load/100, phaseA,B,C]


def neg_seq_ratio(x):
    """torch (B,3,L) -> |I2|/|I1| at fundamental (differentiable)."""
    n = torch.arange(L, dtype=x.dtype)
    cos = torch.cos(2 * np.pi * BIN * n / L); sin = torch.sin(2 * np.pi * BIN * n / L)
    Re = (x * cos).sum(-1) * 2 / L; Im = -(x * sin).sum(-1) * 2 / L   # (B,3)
    Pa = torch.complex(Re[:, 0], Im[:, 0]); Pb = torch.complex(Re[:, 1], Im[:, 1])
    Pc = torch.complex(Re[:, 2], Im[:, 2])
    a = torch.complex(torch.tensor(np.cos(2*np.pi/3)), torch.tensor(np.sin(2*np.pi/3)))
    I1 = (Pa + a * Pb + a*a * Pc) / 3; I2 = (Pa + a*a * Pb + a * Pc) / 3
    return I2.abs() / (I1.abs() + 1e-6)


class CVAE(nn.Module):
    def __init__(self, lat=24, physics=False):
        super().__init__(); self.physics = physics
        self.enc = nn.Sequential(nn.Linear(3*L + CDIM, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.mu = nn.Linear(128, lat); self.lv = nn.Linear(128, lat)
        self.dec = nn.Sequential(nn.Linear(lat + CDIM, 128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 3*L))

    def encode(self, x, c):
        h = self.enc(torch.cat([x.flatten(1), c], 1)); return self.mu(h), self.lv(h)

    def decode(self, z, c):
        out = self.dec(torch.cat([z, c], 1)).view(-1, 3, L)
        if self.physics:                       # hard current-balance projection
            out = out - out.mean(dim=1, keepdim=True)
        return out

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z, c), mu, lv


def cond_vec(rank, phase, load):
    c = np.zeros((len(rank), CDIM), np.float32)
    c[:, 0] = rank / 6.0; c[:, 1] = load / 100.0
    for i, p in enumerate(phase):
        if p in PMAP: c[i, 2 + PMAP[p]] = 1.0
    return c


def train_cvae(X, C, target_negseq, physics, epochs=40, bs=256):
    m = CVAE(physics=physics)
    if physics:                                # balance the reconstruction target too
        Xt = X - X.mean(1, keepdims=True)
    else:
        Xt = X
    ds = TensorDataset(torch.tensor(Xt), torch.tensor(C), torch.tensor(target_negseq))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        m.train()
        for xb, cb, tb in dl:
            opt.zero_grad()
            out, mu, lv = m(xb, cb)
            rec = ((out - xb) ** 2).mean()
            kl = -0.5 * torch.mean(1 + lv - mu**2 - lv.exp())
            loss = rec + 0.001 * kl
            if physics:                        # soft negative-seq vs severity consistency
                loss = loss + 0.5 * ((neg_seq_ratio(out) - tb) ** 2).mean()
            loss.backward(); opt.step()
    return m


@torch.no_grad()
def generate(m, rank, phase, load, n_per=40):
    R, P, Lo = [], [], []
    for r, p, l in zip(rank, phase, load):
        R += [r]*n_per; P += [p]*n_per; Lo += [l]*n_per
    C = torch.tensor(cond_vec(np.array(R), P, np.array(Lo)))
    z = torch.randn(len(C), m.mu.out_features)
    X = m.decode(z, C).numpy().astype(np.float32)
    return X, np.array(R), np.array(P), np.array(Lo)


def main():
    Xf = np.asarray(np.load(DS / "windows.npy"), np.float32)
    n = Xf.shape[2] // DECIM
    Xd = Xf[:, :, :n*DECIM].reshape(Xf.shape[0], 3, n, DECIM).mean(-1).astype(np.float32)
    lab = pd.read_csv(DS / "window_labels.csv", low_memory=False)
    fmask = (lab.label == 1).to_numpy()
    Xfa = Xd[fmask]; labf = lab[fmask].reset_index(drop=True)
    rank = labf.severity_pct.map(RANK).to_numpy(); phase = labf.phase.to_numpy(); load = labf.load_pct.to_numpy()
    C = cond_vec(rank, phase, load)
    # per-severity target negative-seq ratio (from real, decimated)
    real_nsr = neg_seq_ratio(torch.tensor(Xfa)).numpy()
    tgt_by_rank = {r: float(real_nsr[rank == r].mean()) for r in range(7)}
    target = np.array([tgt_by_rank[r] for r in rank], np.float32)
    print(f"faulty decimated windows={len(Xfa)}  real |I2|/|I1| by severity rank: "
          + ", ".join(f"{r}:{tgt_by_rank[r]:.3f}" for r in range(7)))

    res = {}
    gens = {}
    for physics in [True, False]:
        m = train_cvae(Xfa, C, target, physics)
        # generate across all severity x phase x load cells
        cells_r, cells_p, cells_l = [], [], []
        for r in range(7):
            for p in ["A", "B", "C"]:
                for l in [0, 20, 40, 60, 80, 100]:
                    cells_r.append(r); cells_p.append(p); cells_l.append(l)
        Xg, rg, pg, lg = generate(m, cells_r, cells_p, cells_l, n_per=20)
        gens[physics] = (Xg, rg, pg, lg)
        # physics validity
        bal = np.abs(Xg.sum(1)).mean() / (np.abs(Xg).mean() + 1e-9)
        nsr_g = neg_seq_ratio(torch.tensor(Xg)).numpy()
        corr = float(np.corrcoef(rg, nsr_g)[0, 1])
        res[f"physics={physics}"] = dict(balance_residual=round(float(bal), 4),
                                         negseq_severity_corr=round(corr, 4))
        print(f"[gen physics={physics}] current-balance residual={bal:.4f}  "
              f"corr(|I2|/|I1|, severity)={corr:.3f}")

    # ---- downstream: scarce-data severity, augment with generated ----
    feats = None
    def feat_of(Xwin, lbls):
        df = build_feature_frame(Xwin, lbls, fs=FS_D, f0=F0)
        return df
    # real features (decimated, faulty)
    real_df = feat_of(Xfa, labf.assign(case_id=labf.case_id, label=1))
    fcols = severity_features(real_df.columns)
    # scarce split: train on few operating-point groups, test on the rest
    groups = labf.group_id.to_numpy()
    tr_g, te_g = next(StratifiedGroupKFold(5, shuffle=True, random_state=1).split(Xfa, rank, groups))
    # use only 30% of train groups -> scarce
    rng = np.random.RandomState(0)
    keep = rng.rand(len(tr_g)) < 0.3
    tr_g = tr_g[keep]
    Xtr_real = real_df.iloc[tr_g][fcols].to_numpy(np.float32); ytr_real = rank[tr_g]
    Xte = real_df.iloc[te_g][fcols].to_numpy(np.float32); yte = rank[te_g]

    def within1(Xtr, ytr):
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=42).fit(Xtr, ytr)
        pr = np.clip(np.round(m.predict(Xte)), 0, 6).astype(int)
        return round(float((np.abs(pr - yte) <= 1).mean()), 4)

    res["downstream"] = {"no_aug": within1(Xtr_real, ytr_real)}
    for physics in [True, False]:
        Xg, rg, pg, lg = gens[physics]
        glab = pd.DataFrame(dict(case_id=[f"gen_{i}" for i in range(len(Xg))], label=1))
        gdf = feat_of(Xg, glab)
        Xg_f = gdf[fcols].to_numpy(np.float32)
        Xtr = np.vstack([Xtr_real, Xg_f]); ytr = np.r_[ytr_real, rg]
        res["downstream"][f"aug_physics={physics}"] = within1(Xtr, ytr)
    print("\ndownstream severity within-1 (scarce real train):")
    for k, v in res["downstream"].items():
        print(f"  {k:18} {v}")
    json.dump(res, open(DS / "pcdiff_results.json", "w"), indent=2)
    print("\nsaved dataset/pcdiff_results.json")


if __name__ == "__main__":
    main()
