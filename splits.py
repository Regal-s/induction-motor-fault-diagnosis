r"""
Phase 1 / M4 — Leakage-safe cross-validation splitters.

All splits operate at the WINDOW level but keep every window of a `group_id`
together (group_id = operating point = severity x phase x load, binding the regular
case + its 10 inception variants). This prevents window leakage.

  stratified_group_folds : StratifiedGroupKFold (in-distribution headline)
  leave_one_load_out     : hold out one load at a time (PRIMARY generalisation test)

Indices returned are positional row indices into the passed DataFrame.
"""
from __future__ import annotations
import numpy as np

TT_LOADS = ("NL", "20", "40", "60", "80", "100")


def stratified_group_folds(df, y_col="label", group_col="group_id", n_splits=5, seed=42):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(df, df[y_col].to_numpy(), groups=df[group_col].to_numpy()):
            yield tr, te
    except Exception:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(df, df[y_col].to_numpy(), groups=df[group_col].to_numpy()):
            yield tr, te


def leave_one_load_out(df, loads=TT_LOADS, load_col="load_label"):
    """Yield (load, train_idx, test_idx): test = all windows at that load."""
    ll = df[load_col].astype(str).to_numpy()
    for L in loads:
        te = np.where(ll == L)[0]
        tr = np.where(ll != L)[0]
        if len(te):
            yield L, tr, te
