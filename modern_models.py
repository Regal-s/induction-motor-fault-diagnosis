r"""
Modern time-series classifiers added to the project benchmark:

  TRAINABLE ARCHITECTURES (from-scratch PyTorch, no extra deps):
    * PatchTST       — Nie et al., ICLR 2023.
                       Channel-independent patches + Transformer encoder.
    * iTransformer   — Liu et al., ICLR 2024.
                       Inverted attention: each channel = a token.
    * TimesNet       — Wu et al., ICLR 2023.
                       FFT-discovered periodicity + 2-D conv "Inception" block.
    * Mamba          — Gu & Dao, COLM 2023 (CPU-friendly minimal variant).
                       Selective-state-space sequence model with sequential scan.

  FOUNDATION-MODEL ENCODERS (via HuggingFace transformers; classification head trained):
    * TTM            — IBM Granite Tiny Time Mixer (~1 M params, CPU).
    * Chronos-Bolt   — Amazon Chronos-Bolt-small (T5-based, CPU-feasible).

All entries follow:
    train_predict(name, Xtr, ytr, Xte, n_classes, class_weight=None, **kw) -> pred

Xtr/Xte are (N, 3, T) float32 (decimated 3-phase windows).  Hyperparameters
are kept compact so a full benchmark fits in tens of minutes per model.
"""
from __future__ import annotations
import os, math, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _to_tensor(X):
    return torch.from_numpy(np.asarray(X, np.float32))


def _class_weight_tensor(class_weight, n_classes):
    if class_weight is None:
        return None
    w = np.asarray(class_weight, dtype=np.float32)
    if len(w) < n_classes:
        w = np.concatenate([w, np.ones(n_classes - len(w), np.float32)])
    return torch.from_numpy(w[:n_classes]).to(_DEVICE)


def _train_loop(model, Xtr, ytr, Xte, n_classes, class_weight=None,
                epochs=12, batch=64, lr=1e-3, weight_decay=1e-4):
    """Shared training loop: per-sample SGD with class-weighted CE + early-ish stop."""
    Xtr_t = _to_tensor(Xtr); ytr_t = torch.from_numpy(np.asarray(ytr, np.int64))
    Xte_t = _to_tensor(Xte)
    ds = TensorDataset(Xtr_t, ytr_t)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False,
                        num_workers=0)
    model = model.to(_DEVICE)
    wt = _class_weight_tensor(class_weight, n_classes)
    crit = nn.CrossEntropyLoss(weight=wt)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(_DEVICE); yb = yb.to(_DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
        sch.step()
    # predict
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), 256):
            xb = Xte_t[i:i + 256].to(_DEVICE)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds).astype(int)


# ============================================================================
# PatchTST  (Nie et al., ICLR 2023)
# ============================================================================
class PatchTST(nn.Module):
    """Channel-independent patch Transformer. Each of the 3 channels is patched
    along time, every patch is linearly embedded, a small Transformer encoder
    runs over the patch sequence, the patch-mean per channel is concatenated,
    and a linear head produces the class logits."""
    def __init__(self, n_channels=3, n_classes=2, patch=24, stride=12,
                 d_model=48, n_heads=4, n_layers=2, d_ff=96, dropout=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.embed = nn.Linear(patch, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model * n_channels, n_classes)
        self.n_channels = n_channels

    def _patchify(self, x):
        # x: (B, C, T) -> (B, C, n_patches, patch)
        B, C, T = x.shape
        # pad so (T - patch) is divisible by stride
        rem = (T - self.patch) % self.stride
        if rem:
            pad = self.stride - rem
            x = F.pad(x, (0, pad))
            T = x.shape[-1]
        n = (T - self.patch) // self.stride + 1
        patches = x.unfold(-1, self.patch, self.stride)        # (B, C, n, patch)
        return patches

    def forward(self, x):
        B = x.shape[0]
        patches = self._patchify(x)                            # (B, C, n, P)
        BC, n, P = B * self.n_channels, patches.shape[2], patches.shape[3]
        tokens = self.embed(patches.reshape(BC, n, P))         # (B*C, n, d)
        tokens = self.enc(tokens)                              # (B*C, n, d)
        pooled = tokens.mean(dim=1).reshape(B, self.n_channels, -1)  # (B, C, d)
        return self.head(pooled.reshape(B, -1))                # (B, n_classes)


def train_predict_patchtst(Xtr, ytr, Xte, n_classes, class_weight=None,
                           epochs=8, **_):
    _seed(0)
    model = PatchTST(n_channels=Xtr.shape[1], n_classes=n_classes)
    return _train_loop(model, Xtr, ytr, Xte, n_classes, class_weight, epochs=epochs, batch=96)


# ============================================================================
# iTransformer  (Liu et al., ICLR 2024)
# ============================================================================
class iTransformer(nn.Module):
    """Inverted Transformer: each channel's entire time series is projected to
    a single d-dim token; self-attention runs over the C tokens; pooled and
    classified."""
    def __init__(self, n_channels=3, T=250, n_classes=2,
                 d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(T, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model * n_channels, d_model),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))
        self.n_channels = n_channels
        self.T = T

    def forward(self, x):
        B, C, T = x.shape
        if T != self.T:
            x = F.adaptive_avg_pool1d(x, self.T)               # safety: resize
        tokens = self.embed(x)                                 # (B, C, d)
        tokens = self.enc(tokens)                              # (B, C, d)
        return self.head(tokens.reshape(B, -1))                # (B, n_classes)


def train_predict_itransformer(Xtr, ytr, Xte, n_classes, class_weight=None,
                               epochs=12, **_):
    _seed(0)
    model = iTransformer(n_channels=Xtr.shape[1], T=Xtr.shape[2], n_classes=n_classes)
    return _train_loop(model, Xtr, ytr, Xte, n_classes, class_weight, epochs=epochs)


# ============================================================================
# TimesNet  (Wu et al., ICLR 2023) — compact CPU-friendly variant
# ============================================================================
class TimesBlock(nn.Module):
    """FFT-discovered top-k dominant periods; for each period p, reshape
    (T, C) -> (p, T/p, C), apply a small 2-D conv, reshape back, sum and
    add a residual."""
    def __init__(self, n_channels=3, hidden=32, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.in_proj = nn.Linear(n_channels, hidden)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=1),
        )
        self.out_proj = nn.Linear(hidden, n_channels)

    @staticmethod
    def _top_k_periods(x_in, k):
        # x_in: (B, T, C); FFT along time and take top-k frequencies
        xf = torch.fft.rfft(x_in.mean(dim=-1), dim=1)          # (B, F)
        amp = xf.abs().mean(dim=0)                              # (F,)
        amp[0] = 0.0                                             # drop DC
        topk = torch.topk(amp, k=min(k, amp.numel() - 1)).indices.tolist()
        T = x_in.shape[1]
        periods = []
        for f in topk:
            if f <= 0:
                continue
            p = max(2, T // max(1, f))
            if p >= T:
                continue
            periods.append(p)
        if not periods:
            periods = [max(2, T // 4)]
        return periods

    def forward(self, x_in):
        # x_in: (B, T, C)
        B, T, C = x_in.shape
        periods = self._top_k_periods(x_in, self.top_k)
        h = self.in_proj(x_in)                                  # (B, T, hidden)
        outs = []
        for p in periods:
            T_pad = ((T + p - 1) // p) * p
            pad = T_pad - T
            xp = F.pad(h, (0, 0, 0, pad))                      # (B, T_pad, hidden)
            xp = xp.reshape(B, T_pad // p, p, h.shape[-1]).permute(0, 3, 1, 2)
            xp = self.conv(xp).permute(0, 2, 3, 1).reshape(B, T_pad, h.shape[-1])
            outs.append(xp[:, :T])
        h2 = torch.stack(outs, dim=0).mean(dim=0)               # average over periods
        h2 = self.out_proj(h2)                                  # back to C
        return x_in + h2


class TimesNet(nn.Module):
    def __init__(self, n_channels=3, T=250, n_classes=2, n_blocks=2, hidden=32, top_k=2):
        super().__init__()
        self.blocks = nn.ModuleList([TimesBlock(n_channels, hidden, top_k)
                                     for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(n_channels * T, 128),
                                  nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(128, n_classes))
        self.T = T

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        B, C, T = x.shape
        h = x.transpose(1, 2).contiguous()
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.reshape(B, -1))


def train_predict_timesnet(Xtr, ytr, Xte, n_classes, class_weight=None,
                           epochs=10, **_):
    _seed(0)
    model = TimesNet(n_channels=Xtr.shape[1], T=Xtr.shape[2], n_classes=n_classes)
    return _train_loop(model, Xtr, ytr, Xte, n_classes, class_weight,
                       epochs=epochs, batch=32)


# ============================================================================
# Mamba (CPU-friendly minimal variant — selective SSM via sequential scan)
# ============================================================================
class MambaBlock(nn.Module):
    """Minimal selective-state-space block: project input, run sequential
    scan with input-dependent (Delta, B, C) parameters, then a gated residual.
    Sequential scan keeps CPU memory low; not as fast as the CUDA selective_scan."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        # learnable log(A) with discretisation
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, D = x.shape
        xz = self.in_proj(x)                                    # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                            # each (B, T, d_inner)
        # local conv along time
        x_in = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_in = F.silu(x_in)
        # input-dependent (dt, B, C) per token
        dtBC = self.x_proj(x_in)                                # (B, T, d_state*2 + 1)
        dt, B_t, C_t = dtBC.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                       # (B, T, d_inner)
        A = -torch.exp(self.A_log)                              # (d_inner, d_state)
        # sequential scan: h_t = exp(dt * A) * h_{t-1} + dt * B_t * x_t
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = (dt[:, t].unsqueeze(-1) * B_t[:, t].unsqueeze(1))   # (B, d_inner, d_state)
            h = dA * h + dB * x_in[:, t].unsqueeze(-1)
            y = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x_in[:, t]
            ys.append(y)
        y = torch.stack(ys, dim=1)                              # (B, T, d_inner)
        y = y * F.silu(z)                                       # gated residual
        return self.out_proj(y)


class MambaClassifier(nn.Module):
    """CPU-budget Mamba: strided 1-D conv (stride=5) downsamples T -> T/5
    before the SSM block, so the sequential scan stays cheap. Sized down
    (d_model=32, d_state=4, 1 block) so each batch fits on a desktop CPU."""
    def __init__(self, n_channels=3, T=250, n_classes=2, d_model=32, d_state=4, n_blocks=1, stride=5):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=stride*2+1, stride=stride, padding=stride),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(d_model, n_classes))

    def forward(self, x):
        # x: (B, C, T) -> conv front -> (B, d_model, T/stride) -> SSM over T/stride
        h = self.front(x).transpose(1, 2)                       # (B, T_ds, d_model)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h.mean(dim=1))


def train_predict_mamba(Xtr, ytr, Xte, n_classes, class_weight=None, epochs=6, **_):
    _seed(0)
    model = MambaClassifier(n_channels=Xtr.shape[1], T=Xtr.shape[2], n_classes=n_classes)
    return _train_loop(model, Xtr, ytr, Xte, n_classes, class_weight,
                       epochs=epochs, batch=48, lr=8e-4)


# ============================================================================
# Foundation-model encoders -> linear classifier head
# ============================================================================
_FM_CACHE = {}


def _foundation_embed(name, X, batch=8):
    """Extract per-window embeddings from a pretrained foundation model.
    Each window is converted to univariate (channel-mean or channel-stack)
    series. Returned features: (N, d_emb)."""
    if name in _FM_CACHE:
        encoder, tok = _FM_CACHE[name]
    else:
        if name == "TTM":
            # Use granite-tsfm's TinyTimeMixerForPrediction encoder
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
            encoder = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r2").to(_DEVICE).eval()
            tok = None
        elif name == "Chronos-Bolt":
            try:
                from chronos import BaseChronosPipeline
            except ImportError:
                from chronos import ChronosBoltPipeline as BaseChronosPipeline
            encoder = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small", device_map=_DEVICE.type, torch_dtype=torch.float32)
            tok = None
        else:
            raise ValueError(name)
        _FM_CACHE[name] = (encoder, tok)

    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            batch_X = X[i:i + batch]                # (b, 3, T)
            # univariate proxy = channel-stack (3 univariate series per window concatenated)
            uni = batch_X.reshape(-1, batch_X.shape[-1])              # (3b, T)
            uni_t = torch.from_numpy(np.ascontiguousarray(uni.astype(np.float32))).to(_DEVICE)
            if name == "TTM":
                # TTM encoder expects (B, context_length, n_input_channels)
                # We resample window to TTM's expected context length and feed univariate
                ctx = getattr(encoder.config, "context_length", 512)
                if uni_t.shape[-1] != ctx:
                    u = torch.nn.functional.adaptive_avg_pool1d(uni_t.unsqueeze(1), ctx).squeeze(1)
                else:
                    u = uni_t
                u = u.unsqueeze(-1)                                  # (3b, ctx, 1)
                out = encoder.backbone(u)                            # type: ignore[attr-defined]
                emb = out.last_hidden_state.mean(dim=1).reshape(u.shape[0], -1)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
            else:  # Chronos-Bolt
                emb = encoder.embed(uni_t)[0]                         # (3b, d_emb)
                if emb.dim() == 3:                                    # (3b, T_token, d_emb)
                    emb = emb.mean(dim=1)
            emb = emb.reshape(len(batch_X), -1).cpu().numpy().astype(np.float32)   # (b, 3*d_emb)
            feats.append(emb)
    return np.concatenate(feats, axis=0)


def train_predict_foundation(name, Xtr, ytr, Xte, n_classes, class_weight=None, **_):
    from sklearn.linear_model import RidgeClassifierCV
    Ftr = _foundation_embed(name, Xtr)
    Fte = _foundation_embed(name, Xte)
    cw = "balanced" if class_weight is None else dict(enumerate(np.atleast_1d(class_weight)))
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), class_weight=cw)
    clf.fit(Ftr, ytr)
    return clf.predict(Fte).astype(int)


# ============================================================================
# Dispatcher
# ============================================================================
def train_predict(name, Xtr, ytr, Xte, n_classes, class_weight=None, **kw):
    if name == "PatchTST":     return train_predict_patchtst(Xtr, ytr, Xte, n_classes, class_weight, **kw)
    if name == "iTransformer": return train_predict_itransformer(Xtr, ytr, Xte, n_classes, class_weight, **kw)
    if name == "TimesNet":     return train_predict_timesnet(Xtr, ytr, Xte, n_classes, class_weight, **kw)
    if name == "Mamba":        return train_predict_mamba(Xtr, ytr, Xte, n_classes, class_weight, **kw)
    if name in ("TTM", "Chronos-Bolt"):
        return train_predict_foundation(name, Xtr, ytr, Xte, n_classes, class_weight, **kw)
    raise ValueError(f"unknown modern model: {name}")
