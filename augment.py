r"""
Physics-consistent data augmentation for 3-phase stator-current windows.

Applied to TRAINING DATA ONLY to improve generalization to unseen operating points
(loads) and recordings. The guiding rule: NEVER manufacture fake inter-phase imbalance
(that is the inter-turn fault signature). All amplitude/scale operations are therefore
applied JOINTLY to the three phases, which also preserves Kirchhoff current balance
(i_a+i_b+i_c = 0). Operates on float32 arrays of shape (N, 3, T).

Augmentations (each label-preserving):
  joint_load_scale : multiply all 3 phases by one random gain  -> simulates UNSEEN loads
                     (the key cross-load augmentation; |I2|/|I1| ratio is invariant to it,
                      while absolute magnitudes move, teaching load-invariance).
  magnitude_warp   : smooth random gain envelope shared by all 3 phases (slow drift).
  time_shift       : circular roll (phase-coherent across the 3 channels).
  snr_noise        : additive Gaussian noise at a random SNR (sensor noise).
Optionally mixup_ordinal() blends two same-or-adjacent-severity windows for the severity head.

Parameter ranges are conservative defaults (refined from the augmentation literature survey);
override via the AugConfig.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AugConfig:
    load_scale: tuple = (0.7, 1.4)     # joint amplitude gain range (simulate other loads)
    mag_warp_sigma: float = 0.08       # std of smooth gain envelope
    mag_warp_knots: int = 4
    max_shift_frac: float = 0.5        # max circular shift as fraction of one cycle is set by caller
    snr_db_range: tuple = (25.0, 45.0) # additive-noise SNR range (dB); None disables
    p_load_scale: float = 0.9
    p_mag_warp: float = 0.5
    p_time_shift: float = 0.7
    p_noise: float = 0.6


def _smooth_envelope(T, knots, sigma, rng):
    """A length-T smooth random multiplicative envelope ~ 1 + N(0,sigma) interpolated through knots."""
    xk = np.linspace(0, T - 1, knots)
    yk = 1.0 + rng.normal(0, sigma, size=knots)
    return np.interp(np.arange(T), xk, yk)


def augment_batch(X, samples_per_cycle=500, n_aug=1, cfg: AugConfig = None, seed=0):
    """Return (Xaug, idx) where Xaug stacks n_aug augmented copies of X and idx maps each
    augmented row back to its source row in X (so labels can be tiled). X: (N,3,T)."""
    cfg = cfg or AugConfig()
    rng = np.random.RandomState(seed)
    N, C, T = X.shape
    max_shift = max(1, int(cfg.max_shift_frac * samples_per_cycle))
    outs, idxs = [], []
    for a in range(n_aug):
        Y = X.copy()
        # joint load/amplitude scaling (one gain per window, shared across phases)
        m = rng.rand(N) < cfg.p_load_scale
        g = rng.uniform(*cfg.load_scale, size=N).astype(np.float32)
        g[~m] = 1.0
        Y *= g[:, None, None]
        # smooth magnitude warp (shared envelope per window)
        m = rng.rand(N) < cfg.p_mag_warp
        for i in np.where(m)[0]:
            Y[i] *= _smooth_envelope(T, cfg.mag_warp_knots, cfg.mag_warp_sigma, rng)[None, :].astype(np.float32)
        # phase-coherent circular time shift
        m = rng.rand(N) < cfg.p_time_shift
        sh = rng.randint(-max_shift, max_shift + 1, size=N)
        for i in np.where(m)[0]:
            if sh[i]:
                Y[i] = np.roll(Y[i], sh[i], axis=-1)
        # additive Gaussian noise at random SNR
        if cfg.snr_db_range is not None:
            m = rng.rand(N) < cfg.p_noise
            for i in np.where(m)[0]:
                snr = rng.uniform(*cfg.snr_db_range)
                p_sig = (Y[i] ** 2).mean()
                p_noise = p_sig / (10 ** (snr / 10))
                Y[i] += rng.normal(0, np.sqrt(p_noise), size=Y[i].shape).astype(np.float32)
        outs.append(Y.astype(np.float32))
        idxs.append(np.arange(N))
    return np.concatenate(outs, 0), np.concatenate(idxs, 0)


def phase_rotate(X, phase_rank, k):
    """Faulted-phase rotation (exact symmetry of a balanced machine): cyclically roll the three
    channels by k; a fault on phase p becomes a fault on phase (p+k)%3. Returns (X_rot, new_rank).
    Used to augment/balance the faulted-phase classifier without new measurements."""
    Xr = np.roll(X, k, axis=1).astype(np.float32)
    new_rank = (np.asarray(phase_rank) + k) % 3
    return Xr, new_rank


def mixup_ordinal(Xf, y_rank, alpha=0.2, max_level_gap=1, seed=0):
    """Feature-space mixup restricted to pairs within `max_level_gap` ordinal levels, so the
    blended (continuous) rank target stays meaningful. Xf: (N,F) features, y_rank: (N,) ints.
    Returns (Xmix, ymix_continuous)."""
    rng = np.random.RandomState(seed)
    N = len(Xf)
    order = np.argsort(y_rank + rng.rand(N) * 1e-3)
    j = order[(np.searchsorted(np.sort(y_rank), y_rank) + 1).clip(0, N - 1)]
    lam = rng.beta(alpha, alpha, size=N).astype(np.float32)
    ok = np.abs(y_rank - y_rank[j]) <= max_level_gap
    lam = np.where(ok, lam, 1.0)[:, None]
    Xmix = (lam * Xf + (1 - lam) * Xf[j]).astype(np.float32)
    ymix = (lam[:, 0] * y_rank + (1 - lam[:, 0]) * y_rank[j]).astype(np.float32)
    return Xmix, ymix
