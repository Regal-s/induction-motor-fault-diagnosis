r"""
Compact CPU xLSTM (Extended LSTM, Beck et al. 2024) for 3-phase current windows.

The official NX-AI `xlstm` package is GPU/Triton-oriented and does not install on
CPU-only Windows, so this is a faithful-but-compact PyTorch re-implementation of the
xLSTM's key ideas, runnable on CPU (same pragmatic route the project used for PCM-Net):

  * sLSTM cell with EXPONENTIAL input/forget gating + a stabilizer state `m`
    (the core xLSTM innovation that fixes vanilla-LSTM gate saturation) and a
    normalizer state `n` so the hidden state is c/n.
  * xLSTM residual block: LayerNorm -> sLSTM -> output projection -> residual.
  * A multi-scale 1-D conv stem turns raw 3-phase current into d_model tokens,
    then a stack of xLSTM blocks runs over time, mean-pool -> linear task head.

Used by benchmark_models.py as the deep (raw-signal) competitor to TabPFN / XGBoost.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class sLSTMCell(nn.Module):
    """Scalar xLSTM cell with exponential gating + stabilization (Beck et al. 2024, eq. 15-21)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        # input projections (x_t) and recurrent projections (h_{t-1}) for the 4 gates
        self.W = nn.Linear(d, 4 * d, bias=True)
        self.R = nn.Linear(d, 4 * d, bias=False)
        # forget-gate bias initialised positive -> remember by default (standard LSTM trick)
        with torch.no_grad():
            self.W.bias[2 * d:3 * d].fill_(1.0)

    def forward(self, x):  # x: (B, T, d)
        B, T, d = x.shape
        h = x.new_zeros(B, d)
        c = x.new_zeros(B, d)
        n = x.new_zeros(B, d)
        m = x.new_zeros(B, d)  # stabilizer state
        outs = []
        for t in range(T):
            g = self.W(x[:, t]) + self.R(h)
            z, i_pre, f_pre, o_pre = g.chunk(4, dim=-1)
            z = torch.tanh(z)
            o = torch.sigmoid(o_pre)
            # exponential gating with stabilization:
            m_new = torch.maximum(f_pre + m, i_pre)
            i = torch.exp(i_pre - m_new)
            f = torch.exp(f_pre + m - m_new)
            c = f * c + i * z
            n = f * n + i
            h = o * (c / (n + 1e-6))
            m = m_new
            outs.append(h)
        return torch.stack(outs, dim=1)  # (B, T, d)


class xLSTMBlock(nn.Module):
    """Pre-norm residual xLSTM block: LN -> sLSTM -> proj -> dropout -> +residual."""

    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.cell = sLSTMCell(d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.drop(self.proj(self.cell(self.norm(x))))


class xLSTMNet(nn.Module):
    """Conv stem -> n xLSTM blocks -> mean-pool -> linear head."""

    def __init__(self, in_ch=3, d_model=64, n_blocks=2, n_classes=2, p_drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.blocks = nn.ModuleList([xLSTMBlock(d_model, p_drop) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):           # x: (B, in_ch, L)
        z = self.stem(x)            # (B, d, L')
        z = z.transpose(1, 2)       # (B, L', d)
        for blk in self.blocks:
            z = blk(z)
        z = z.mean(dim=1)           # mean-pool over time
        return self.head(z)


# ----------------------------------------------------------------------------- helpers
def decimate(Xw: np.ndarray, factor: int = 8) -> np.ndarray:
    """Block-average decimation 2000 -> 2000/factor samples (anti-alias by averaging)."""
    n = Xw.shape[2] // factor
    return Xw[:, :, :n * factor].reshape(Xw.shape[0], Xw.shape[1], n, factor).mean(-1).astype(np.float32)


def train_predict(Xtr, ytr, Xte, n_classes, epochs=18, bs=256, lr=1e-3,
                  d_model=64, n_blocks=2, class_weight=None, seed=0, device="cpu",
                  verbose=False):
    """Train xLSTMNet on (Xtr,ytr) and return predicted labels for Xte.
    Xtr/Xte: (N,3,L) float32 already decimated; ytr: int labels in [0,n_classes)."""
    torch.manual_seed(seed); np.random.seed(seed)
    # per-channel standardization using train stats
    mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    net = xLSTMNet(in_ch=Xtr.shape[1], d_model=d_model, n_blocks=n_blocks,
                   n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    if class_weight is not None:
        class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=class_weight)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr_t)); tot = 0.0
        for i in range(0, len(Xtr_t), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            out = net(Xtr_t[idx]); loss = lossf(out, ytr_t[idx])
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += loss.item() * len(idx)
        if verbose:
            print(f"    epoch {ep:2d} loss={tot/len(Xtr_t):.4f}")
    net.eval(); preds = []
    Xte_t = torch.tensor(Xte, device=device)
    with torch.no_grad():
        for i in range(0, len(Xte_t), 512):
            preds.append(net(Xte_t[i:i + 512]).argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)
