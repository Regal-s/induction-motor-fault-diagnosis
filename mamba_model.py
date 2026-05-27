r"""
Compact CPU Mamba-2 (selective state-space) for 3-phase current windows.

The official `mamba-ssm` needs CUDA + Triton (GPU-only), so this is a faithful compact PyTorch
re-implementation of the Mamba/Mamba-2 selective-SSM core (Gu & Dao 2023; Dao & Gu 2024),
runnable on CPU (same pragmatic route used for xLSTM). Provides a `train_predict` with the SAME
signature as xlstm_model.train_predict so it drops into the deep-model comparison.

Block: input projection -> depthwise conv -> data-dependent (Delta, B, C) -> diagonal selective
scan h_t = exp(Delta*A) h_{t-1} + Delta*B*x_t, y_t = C h_t + D x_t -> SiLU(z) gate -> out proj,
with a multi-scale conv stem and a linear task head.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.di = expand * d
        self.dt_rank = max(1, d // 16)
        self.in_proj = nn.Linear(d, 2 * self.di)
        self.conv = nn.Conv1d(self.di, self.di, d_conv, groups=self.di, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.di, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.di, 1)
        self.A_log = nn.Parameter(torch.log(A))           # (di, d_state)
        self.D = nn.Parameter(torch.ones(self.di))
        self.out_proj = nn.Linear(self.di, d)
        self.d_state = d_state

    def forward(self, x):                                 # x: (B,L,d)
        B_, L, _ = x.shape
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)                       # (B,L,di)
        xs = self.conv(xs.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xs = F.silu(xs)
        dbl = self.x_proj(xs)                             # (B,L,dt_rank+2*d_state)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (B,L,di)
        A = -torch.exp(self.A_log)                        # (di,d_state)
        h = x.new_zeros(B_, self.di, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)             # (B,di,d_state)
            dBx = (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * xs[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))        # (B,di)
        y = torch.stack(ys, dim=1) + xs * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaResBlock(nn.Module):
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.mamba = MambaBlock(d); self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.drop(self.mamba(self.norm(x)))


class MambaNet(nn.Module):
    def __init__(self, in_ch=3, d_model=64, n_blocks=2, n_classes=2, p_drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, d_model, 7, 2, 3), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, 2, 2), nn.BatchNorm1d(d_model), nn.GELU())
        self.blocks = nn.ModuleList([MambaResBlock(d_model, p_drop) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        z = self.stem(x).transpose(1, 2)
        for b in self.blocks:
            z = b(z)
        return self.head(z.mean(1))


def train_predict(Xtr, ytr, Xte, n_classes, epochs=18, bs=256, lr=1e-3, d_model=64, n_blocks=2,
                  class_weight=None, seed=0, device=None, verbose=False):
    """Same interface as xlstm_model.train_predict. device=None auto-detects CUDA."""
    if device is None:
        try:
            import gpu; device = gpu.resolve()
        except Exception:
            device = "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    net = MambaNet(in_ch=Xtr.shape[1], d_model=d_model, n_blocks=n_blocks, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    if class_weight is not None:
        class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=class_weight)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), ytr_t[idx]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval(); preds = []; Xte_t = torch.tensor(Xte, device=device)
    with torch.no_grad():
        for i in range(0, len(Xte_t), 512):
            preds.append(net(Xte_t[i:i + 512]).argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)
