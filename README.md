# Induction-Motor Stator Inter-Turn Fault Diagnosis

A hierarchical machine-learning pipeline that diagnoses **stator winding turn-to-turn
(inter-turn) short-circuit faults** in a three-phase induction motor from its **three-phase
stator current**, as a 3-stage cascade:

> **Detect** (healthy vs faulty) → **Severity** (% shorted turns) → **Faulted phase** (A / B / C)

Data is PSCAD/EMTDC time-domain simulation (50 Hz, 25 kHz sampling). The emphasis throughout
is **leakage-safe evaluation** and **physically interpretable features**, with
**Leave-One-Load-Out (LOLO)** as the primary (honest) generalisation metric.

## Headline results
Deployed cascade (XGBoost on physics features; severity is load-aware):

| | In-distribution (GroupKFold) | Cross-load (LOLO, primary) |
|---|---|---|
| **Detection** | macro-F1 0.977, ROC-AUC 0.994 | macro-F1 0.948 |
| **Severity** (within-1 level) | 0.997 | 0.95 |
| **Phase ID** | macro-F1 0.972 | **macro-F1 0.997** |
| **End-to-end** (state + severity±1 + phase) | within-1 **0.949** | within-1 **0.916** |

**Key finding:** detection and phase ID are load-robust (phase ID rides on the load-invariant
negative-sequence current *angle*); absolute severity is load-dependent, but supplying the
**measured load** restores cross-load severity (within-1 0.73 → 0.95). Full analysis in
[`report.md`](report.md).

## Pipeline
```
data_index.py     # 357-case manifest (labels, event/region bounds)
segment.py        # -> 18,033 labelled current windows (windows.npy + window_labels.csv)
features.py       # -> 72 physics features (features.parquet)  [EPVA, symmetrical components,
                  #    Park ellipse, harmonics/THD, |I1|-normalised magnitudes]
splits.py         # leakage-safe CV: StratifiedGroupKFold + Leave-One-Load-Out
train_stage1.py   # detection (XGBoost) + SHAP audit
train_stage2.py   # severity (ordinal regression, load-aware)
train_stage3.py   # faulted phase (3-class)
cascade.py        # end-to-end Stage1 -> {Stage2, Stage3}
```
Phase 3–4 (robustness, deep baseline, extras):
```
noise_robustness.py        # SNR 40/30/20 dB
calibration_uncertainty.py # Stage-1 ECE/Brier + quantile uncertainty
conformal_uncertainty.py   # conformalised severity intervals
dl_resnet1d.py             # ResNet-1D (Track B) vs XGBoost (Track A)
severity_deepdive.py       # feature-stability + load-aware severity fix
stage0_onset.py            # fault-onset detection (per-cycle |I2|/|I1|)
```

## Quick start
```bash
pip install numpy pandas scikit-learn xgboost scipy pyarrow shap  # + torch (cpu) for dl_resnet1d.py
python data_index.py && python segment.py && python features.py
python train_stage1.py && python train_stage2.py && python train_stage3.py
python cascade.py
```
Requires the raw PSCAD dataset locally (see below). The committed `dataset/features.parquet`
already lets you re-run the models without rebuilding from raw signals.

## Repository layout
- `*.py` — pipeline and experiment scripts
- `report.md` — full results report (per-stage tables, confusion matrices, Phase 3–4)
- `DATASET_KNOWLEDGE.md` — dataset format, channel map, event timeline
- `ML_PLAN_DETAILED.md` — research-grounded design (with literature references)
- `PROJECT_LOG.md` — chronological build log / continuity notes
- `dataset/` — feature table, labelled out-of-fold predictions, metrics (JSON), figures

## Data note
The raw PSCAD/EMTDC simulation files (~122 GB) and the 413 MB `windows.npy` are **not** in the
repo (`.gitignore`). The dataset covers: severity ∈ {0.3, 0.5, 1, 2, 3, 4, 5}% shorted turns,
faulted phase ∈ {A, B, C}, load ∈ {NL, 20, 40, 60, 80, 100}%, plus healthy baselines and
fault-inception-time variants.

## Method highlights
- **Leakage-safe splits** — grouped by operating point (regular case + its inception variants
  stay together); LOLO as the headline metric.
- **Matched-load healthy data** taken from the pre-fault region of each fault recording, removing
  load as a confound for detection.
- **Physics features** — EPVA severity factor, compensated negative-sequence current
  (`|I2|/|I1|`, ∠I2), Park's-vector ellipse, MCSA harmonics — verified via SHAP, not dataset shortcuts.

---
*Generated as a research project; simulation-only (no hardware validation). See `report.md`
§9–10 for limitations and next steps.*
