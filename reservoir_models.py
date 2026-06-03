r"""
Three modern time-series classifiers added to the project benchmark:

  * ESN     — Echo State Network (Jaeger 2001)
              fixed sparse random recurrent reservoir + ridge readout.
  * NG-RC   — Next-Generation Reservoir Computing (Gauthier et al., Nat. Commun. 2021)
              no random reservoir; polynomial (linear + quadratic) features of
              k-delayed input window + ridge readout.
  * ROCKET  — Random Convolutional Kernel Transform (Dempster et al., KDD 2020)
              k random 1-D conv kernels of varying length/dilation; per-channel
              {max, PPV} features + ridge readout.

All three follow the same interface as xlstm_model.train_predict so they plug
into benchmark_models.py:

    train_predict(Xtr, ytr, Xte, n_classes, class_weight=None, **kwargs) -> pred

Xtr / Xte are (N, 3, T) float32 (decimated 3-phase current windows).
ytr is (N,) integer class labels in {0..n_classes-1}.

The implementations are intentionally compact (NumPy only, no PyTorch) -- all
three are fast enough to fit inside the existing benchmark harness.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeClassifierCV


# ============================================================================
# Echo State Network  (Jaeger 2001)
# ============================================================================
class ESNFeaturizer:
    """Drive each (3, T) window through a fixed leaky tanh reservoir; return
    a feature vector that summarises the reservoir state trajectory."""
    def __init__(self, n_reservoir=300, spectral_radius=0.9, leak=0.3,
                 input_scale=1.0, sparsity=0.05, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n_reservoir; self.leak = leak; self.in_scale = input_scale
        # W_in: dense, scaled small
        self.W_in = rng.uniform(-1.0, 1.0, (n_reservoir, 3)).astype(np.float32) * input_scale
        # W: sparse with prescribed spectral radius
        W = rng.uniform(-1.0, 1.0, (n_reservoir, n_reservoir)).astype(np.float32)
        mask = rng.uniform(size=W.shape) < sparsity
        W *= mask
        try:
            sr = float(np.max(np.abs(np.linalg.eigvals(W))))
        except np.linalg.LinAlgError:
            sr = 1.0
        if sr > 0:
            W *= (spectral_radius / sr)
        self.W = W

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        """x shape (3, T) -> feature vector of length 3*n (mean, max, last)."""
        T = x.shape[1]
        state = np.zeros(self.n, dtype=np.float32)
        states = np.empty((T, self.n), dtype=np.float32)
        for t in range(T):
            new = np.tanh(self.W_in @ x[:, t] + self.W @ state)
            state = (1.0 - self.leak) * state + self.leak * new
            states[t] = state
        # Reservoir state summary: mean, max-abs, final state
        return np.concatenate([states.mean(axis=0), np.abs(states).max(axis=0), state])

    def transform(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        feats = np.empty((N, 3 * self.n), dtype=np.float32)
        for i in range(N):
            feats[i] = self.transform_one(X[i])
        return feats


def train_predict_esn(Xtr, ytr, Xte, n_classes, class_weight=None,
                      n_reservoir=300, seed=0, **_) -> np.ndarray:
    feat = ESNFeaturizer(n_reservoir=n_reservoir, seed=seed)
    Ftr = feat.transform(Xtr)
    Fte = feat.transform(Xte)
    cw = "balanced" if class_weight is None else dict(enumerate(np.atleast_1d(class_weight)))
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), class_weight=cw)
    clf.fit(Ftr, ytr)
    return clf.predict(Fte).astype(int)


# ============================================================================
# Next-Generation Reservoir Computing  (Gauthier et al., Nat. Commun. 2021)
# ============================================================================
def _ngrc_features(x: np.ndarray, n_taps=4, n_samples=20) -> np.ndarray:
    """x shape (3, T) -> NG-RC feature vector.
    For n_samples evenly-spaced timesteps t_k, build the linear vector
    O_lin = [x_c(t_k - d) for c in 3, d in 0..n_taps-1] and the unique
    quadratic outer-products O_quad = {O_i * O_j : i <= j}.  Average over
    the n_samples sampled timesteps."""
    T = x.shape[1]
    # sample timesteps that have enough history
    start = n_taps - 1
    if T - start < n_samples:
        ts = np.arange(start, T, dtype=int)
    else:
        ts = np.linspace(start, T - 1, n_samples).astype(int)
    feats = []
    for t in ts:
        lin = x[:, t - n_taps + 1: t + 1].reshape(-1)               # (3*n_taps,)
        # unique upper-triangular quadratic products
        L = len(lin)
        iu, ju = np.triu_indices(L)
        quad = lin[iu] * lin[ju]
        feats.append(np.concatenate([lin, quad]))
    feats = np.stack(feats, axis=0)
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])     # (2 * (3*n_taps + tri),)


def train_predict_ngrc(Xtr, ytr, Xte, n_classes, class_weight=None,
                       n_taps=4, n_samples=20, **_) -> np.ndarray:
    Ftr = np.stack([_ngrc_features(x, n_taps, n_samples) for x in Xtr])
    Fte = np.stack([_ngrc_features(x, n_taps, n_samples) for x in Xte])
    cw = "balanced" if class_weight is None else dict(enumerate(np.atleast_1d(class_weight)))
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), class_weight=cw)
    clf.fit(Ftr, ytr)
    return clf.predict(Fte).astype(int)


# ============================================================================
# ROCKET  (Dempster et al., KDD 2020)
# ============================================================================
class ROCKETFeaturizer:
    """Generate n_kernels random 1-D conv kernels; each kernel produces
    {max, PPV} features per channel after convolution.  Total feature
    dimension = 2 * n_kernels * n_channels."""
    def __init__(self, n_kernels=2000, seed=0):
        self.n_kernels = n_kernels
        rng = np.random.RandomState(seed)
        kernel_lengths = rng.choice([7, 9, 11], size=n_kernels)
        # Per-kernel: weights centred to zero-mean; random bias; random dilation; random padding
        self.kernels = []
        for L in kernel_lengths:
            w = rng.normal(0.0, 1.0, L).astype(np.float32)
            w -= w.mean()
            bias = float(rng.uniform(-1.0, 1.0))
            # dilation must allow the kernel to fit inside the typical window;
            # cap at 16 to avoid pathological cases.
            max_dil = max(1, int(np.floor(np.log2((250 - 1) / (L - 1)))))
            dil = int(2 ** rng.randint(0, max(1, max_dil + 1)))
            pad = (rng.randint(0, 2) == 1)
            self.kernels.append((w, bias, dil, pad))

    @staticmethod
    def _apply_kernel(x: np.ndarray, w: np.ndarray, bias: float,
                      dilation: int, pad: bool) -> np.ndarray:
        """x: (T,)  ->  conv output (T_out,) using a single dilated kernel."""
        L = len(w)
        if pad:
            p = ((L - 1) * dilation) // 2
            x = np.concatenate([np.zeros(p, np.float32), x, np.zeros(p, np.float32)])
        T = len(x)
        # dilated valid convolution
        n_out = T - (L - 1) * dilation
        if n_out <= 0:
            return np.zeros(1, np.float32)
        # build view of shape (n_out, L) via stride trick
        idx = np.arange(L) * dilation
        out = np.zeros(n_out, dtype=np.float32)
        for k, di in enumerate(idx):
            out += w[k] * x[di: di + n_out]
        return out + bias

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, C, T) -> (N, 2 * n_kernels * C) features {max, PPV} per kernel/channel."""
        N, C, T = X.shape
        F = np.empty((N, 2 * self.n_kernels * C), dtype=np.float32)
        for i in range(N):
            row = []
            for c in range(C):
                xc = X[i, c]
                for (w, bias, dil, pad) in self.kernels:
                    out = self._apply_kernel(xc, w, bias, dil, pad)
                    row.append(out.max())
                    row.append(float((out > 0).mean()))
            F[i] = np.asarray(row, dtype=np.float32)
        return F


def train_predict_rocket(Xtr, ytr, Xte, n_classes, class_weight=None,
                         n_kernels=2000, seed=0, **_) -> np.ndarray:
    feat = ROCKETFeaturizer(n_kernels=n_kernels, seed=seed)
    Ftr = feat.transform(Xtr)
    Fte = feat.transform(Xte)
    cw = "balanced" if class_weight is None else dict(enumerate(np.atleast_1d(class_weight)))
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), class_weight=cw)
    clf.fit(Ftr, ytr)
    return clf.predict(Fte).astype(int)


# Convenience dispatcher (keeps benchmark_models.py simple)
def train_predict(name: str, *args, **kwargs) -> np.ndarray:
    if name == "ESN":     return train_predict_esn(*args, **kwargs)
    if name == "NG-RC":   return train_predict_ngrc(*args, **kwargs)
    if name == "ROCKET":  return train_predict_rocket(*args, **kwargs)
    raise ValueError(f"unknown reservoir/ROCKET model: {name}")
