r"""Shared feature-group definitions used by the stage trainers and the cascade.

- LABEL_COLS      : non-feature columns (labels/metadata) to exclude from X.
- all_features    : every feature column (used for Stage-1 detection & Stage-3 phase).
- severity_features: LOAD-INVARIANT set for Stage-2 severity = dimensionless ratios/
  shapes/angles + |I1|-normalised magnitudes (the n_* columns). Chosen because raw
  magnitudes don't transfer to unseen loads (see stage2_experiment.py).
"""
LABEL_COLS = {"window_id", "case_id", "group_id", "state", "label", "severity_pct",
              "phase", "load_label", "load_pct", "inception_s", "variant", "region",
              "src_start", "scale"}

INVARIANT = [
    "epva_sf", "I2_I1_ratio", "I0_I1_ratio", "I2c_I1_ratio",
    "pva_ecc", "pva_axratio", "pva_tilt_deg",
    "h3ratio_mean", "thd_mean",
    "pha_crest", "phb_crest", "phc_crest",
    "pha_skew", "phb_skew", "phc_skew",
    "pha_kurt", "phb_kurt", "phc_kurt",
    "pha_h3ratio", "phb_h3ratio", "phc_h3ratio",
    "pha_thd", "phb_thd", "phc_thd",
]
I1NORM = ["n_park_dc", "n_park_2f", "n_park_mod_mean", "n_park_mod_std",
          "n_pva_minor", "n_pva_major", "n_I0_abs", "n_I2_abs",
          "n_rms_imbalance_std", "n_rms_imbalance_maxmin",
          "n_rms_diff_ab", "n_rms_diff_bc", "n_rms_diff_ca"]


def all_features(columns):
    return [c for c in columns if c not in LABEL_COLS]


def severity_features(columns, load_aware=True):
    """Stage-2 severity features. load_aware=True appends the MEASURED load (load_pct)
    — legitimate (a sensor input, not a label) and lifts LOLO within-1 0.73 -> 0.95
    (see severity_deepdive.py). Set load_aware=False for the load-blind variant."""
    cols = set(columns)
    feats = [f for f in (INVARIANT + I1NORM) if f in cols]
    if load_aware and "load_pct" in cols:
        feats.append("load_pct")
    return feats
