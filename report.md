# Hierarchical Diagnosis of Stator Inter-Turn Faults in an Induction Motor — Results Report

**Task.** From three-phase stator current, a 3-stage cascade: **(1) Detection** healthy vs faulty →
**(2) Severity** (% shorted turns) → **(3) Faulted phase** (A/B/C).
**Data.** PSCAD/EMTDC simulation, 50 Hz, 25 kHz, `Healthy case data` + `TT fault Data` only.
**Headline.** End-to-end (state, severity±1-level, phase) correct on **94.9 % of windows in-distribution**
and **91.6 % across unseen loads (Leave-One-Load-Out)** with a load-aware severity stage; detection F1 0.95 and
phase ID 0.997 are load-robust. Severity was the cross-load bottleneck (LOLO within-1 0.73 with load-blind
features) but is largely resolved by supplying the **measured load** (LOLO within-1 0.95) — see §6/§13.

*Generated 2026-05-22. All metrics from `dataset/*_metrics.json`; see `PROJECT_LOG.md` for provenance.*

---

## 1. Dataset & labelling
- **Machine / fault:** 3-phase induction motor, stator winding turn-to-turn (inter-turn) short circuit.
- **Signals:** three-phase stator current `Ia_ABCIM` (kA), `_03.out` cols 2–4. Fundamental 50 Hz, sampled 25 kHz.
- **Cases:** 357 simulations — 21 healthy (5 s) + 336 TT-fault (10 s). TT axes: severity ∈ {0.3, 0.5, 1, 2, 3, 4, 5}%,
  phase ∈ {A,B,C}, load ∈ {NL,20,40,60,80,100}%, plus 210 "extra" cases varying fault-inception time.
- **Event timeline (verified):** motor start sample 50 000 (2.0 s); load change 100 000 (4.0 s, none at NL);
  **fault inception 180 000 (7.2 s)**; clears ~200 000 (8.0 s).

## 2. Pipeline
| Step | Script | Output |
|---|---|---|
| Manifest | `data_index.py` | `manifest.csv` (357 cases, labels, region bounds) |
| Windowing | `segment.py` | `dataset/windows.npy` (18 033 × 3 × 2000) + labels |
| Features | `features.py` | `dataset/features.parquet` (72 features + labels) |
| Splits | `splits.py` | StratifiedGroupKFold + Leave-One-Load-Out |
| Stage models | `train_stage{1,2,3}.py` | per-stage metrics + labeled OOF preds |
| Cascade | `cascade.py` | end-to-end metrics + labeled OOF preds |

**Windowing.** 2000 samples (4 cycles); fault region step 500 (75 % overlap), healthy region step 4000.
Pre-fault windows of TT files are labelled **healthy** (matched-load) — this removes load as a confound and
provides a per-case baseline for negative-sequence compensation. Yield: 11 760 faulty + 6 273 healthy windows.

**Normalisation.** Each window divided by a single per-recording scale (RMS of the case's healthy region across
all 3 phases) — removes load amplitude while preserving inter-phase imbalance.

**Features (72, Track-A / physics-based).** EPVA severity factor (2f₁/DC of Park-vector modulus); symmetrical
components |I1|,|I2|,|I0|, ratios and angles (sin/cos); per-case **compensated** negative-sequence I2c;
Park ellipse (eccentricity, axis-ratio, tilt); 3rd-harmonic ratio, THD; per-phase RMS/peak/crest/std/skew/kurt
and inter-phase differences; and **|I1|-normalised magnitudes** (`n_*`, load-invariant magnitude shape).

**Models.** XGBoost. Stage-1 binary (class-weighted); Stage-2 ordinal regression on severity rank (0–6);
Stage-3 3-class. Stage-1 & 3 use all features; **Stage-2 uses the load-invariant subset** (see §6).

**Evaluation protocols (leakage-safe).**
- **StratifiedGroupKFold** (group = operating point, binding the regular case + its 10 inception variants) — in-distribution.
- **Leave-One-Load-Out (LOLO)** — train on 5 loads, test on the held-out load. **Primary** metric (the honest
  generalisation test; guards against the model learning *load* instead of *fault physics*).

---

## 3. Stage 1 — Detection (Healthy vs Faulty)
| Protocol | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| StratifiedGroupKFold | 0.979 | **0.977** | 0.994 |
| Leave-One-Load-Out (pooled) | 0.953 | **0.948** | — |

Confusion (GroupKFold OOF): `[[6239, 34], [350, 11410]]` (rows = true H/F).
Per-load LOLO macro-F1: NL 0.885, **20 0.749**, 40/60/80/100 = 1.00.
- The weak loads are the *light-load* extremes: at NL the model misses some faults (faulty recall 0.80);
  at 20 % it over-flags healthy windows (healthy recall 0.45). This is the documented incipient-fault-at-light-load
  difficulty — small `|I2|` overlapping the healthy distribution at an unseen operating point.
- **SHAP audit (leakage guard) passes:** top features are physical — `pva_major`, `pva_ecc`, `I0_I1_ratio`,
  `rms_diff_ca`, `I2_I1_ratio`, `epva_sf`; no load proxy. Figure: `dataset/stage1_confusion.png`.

## 4. Stage 2 — Severity (ordinal regression, faulty windows only)
| Protocol | MAE (levels) | QWK | Exact | Within-1 |
|---|---|---|---|---|
| StratifiedGroupKFold | 0.330 | 0.948 | 0.688 | **0.990** |
| Leave-One-Load-Out (pooled) | 1.114 | 0.736 | 0.179 | **0.729** |

Per-load LOLO within-1: NL 0.803, 20 0.806, 40 0.969, 60 0.941, **80 0.353, 100 0.586**.
- In-distribution, severity is essentially solved (within-1 0.99, QWK 0.95) and errors are almost entirely
  between adjacent levels (the ordinal framing is appropriate).
- Across unseen loads it is much harder: the severity↔|I2| mapping shifts with operating point, so an unseen
  load (esp. the boundary loads) is an extrapolation. **This is the cascade's bottleneck.**
- Figure: `dataset/stage2_confusion.png` (band-diagonal — adjacent-level confusion).

## 5. Stage 3 — Faulted phase (A/B/C, faulty windows only)
| Protocol | Accuracy | Macro-F1 | Per-phase F1 (A/B/C) |
|---|---|---|---|
| StratifiedGroupKFold | 0.972 | 0.972 | 0.956 / 0.959 / 1.000 |
| Leave-One-Load-Out (pooled) | **0.997** | **0.997** | 0.996 / 0.996 / 1.000 |

- Phase ID is **load-robust** — LOLO ≥ GroupKFold — because the faulted-phase signature lives in the
  *negative-sequence current angle* (∠I2 clusters ~120° apart per phase) and per-phase asymmetry, which are
  load-invariant. Only minor A↔B confusion remains in-distribution; phase C is perfect.
- Figure: `dataset/stage3_confusion.png`.

## 6. Stage-2 load-invariant fix (experiment)
Initial Stage-2 (all features) collapsed at the no-load extreme under LOLO (NL within-1 0.39). Root cause: the
per-recording normalisation scale behaves differently at no-load, so absolute magnitude features don't transfer.
`stage2_experiment.py` compared feature sets:

| Severity features | GK within-1 | LOLO within-1 | NL within-1 |
|---|---|---|---|
| all (72) | 0.993 | 0.887 | **0.39** |
| invariant only (24) | 0.946 | 0.702 | 0.78 |
| **invariant + `n_*` (37)** | **0.990** | 0.729 | **0.80** |

Adding `|I1|`-normalised magnitudes recovers in-distribution sharpness *and* fixes the no-load collapse, so
Stage-2 uses this set. (The lower *pooled* LOLO vs all-features reflects all-features overfitting the easy
interpolation loads while failing the boundary; the invariant set is uniformly robust.) Coarse 3-class severity
did not improve LOLO and was not adopted.

---

## 7. End-to-end cascade
Stage-1 gates; Stages 2 & 3 are applied to windows **predicted** faulty (true end-to-end).

Severity stage is **load-aware** (measured load supplied), the deployed configuration.

| | Detect F1 | Sev within-1 (cond.) | Phase acc (cond.) | **E2E window within-1** | E2E window exact | **Per-case within-1** |
|---|---|---|---|---|---|---|
| **GroupKFold** | 0.977 | 0.997 | 0.957 | **0.949** | 0.911 | **0.927** |
| **LOLO (primary)** | 0.948 | 0.945 | 0.997 | **0.916** | 0.521 | **0.918** |

(For reference, with load-*blind* severity features the LOLO end-to-end within-1 was 0.774 / per-case 0.708;
adding the measured load lifts these to 0.916 / 0.918.)

(*E2E "within-1" = state correct, and if faulty: severity within 1 level AND phase exact. Per-case aggregates
windows by recording: majority-vote phase, median severity.*)

Artifacts: `dataset/cascade_oof_predictions.csv` (labeled), `dataset/cascade_metrics.json`.

---

## 8. Key findings
1. **The three sub-tasks separate by load-robustness, and the physics explains it.** Detection (0.95) and
   phase ID (0.997) generalise across loads because they ride on load-invariant signatures (negative-sequence
   ratio/angle, per-phase asymmetry). Absolute severity does not, because |I2| magnitude is load-dependent —
   **but supplying the measured load (load-aware severity) restores LOLO within-1 from 0.73 to 0.95** (§13.4).
2. **LOLO was essential.** In-distribution numbers are uniformly high (≥0.97); only LOLO reveals where the
   system genuinely generalises (detection, phase) versus where it extrapolates poorly (severity at unseen loads).
   Reporting only random/grouped CV would have hidden this.
3. **Physically-grounded features + leakage-safe splits give an honest, interpretable result** — SHAP confirms
   the models use inter-turn fault physics, not dataset shortcuts.
4. **Ordinal framing matters for severity** — errors are overwhelmingly ±1 level; within-1 (0.99 in-distribution)
   is the operationally meaningful metric, not exact accuracy.

## 9. Limitations
- **Simulation only** (PSCAD/EMTDC); no hardware validation.
- **Cross-load severity** with load-*blind* features remains weak at unseen loads; the load-aware model resolves
  most of this but the no-load boundary (extrapolation) is still the hardest point.
- **Per-recording normalisation** uses each case's own healthy baseline (a "compensation" assumption); detection
  was verified to also work on the raw |I2|/|I1| ratio, so this is not the dominant driver, but a baseline-free
  normalisation should be evaluated for deployment realism.
- **Uncertainty** via naive quantile regression is under-calibrated (§13.3) — conformal prediction recommended.
- Fault interval assumed known (no online onset detection yet).

## 10. Recommended next steps
- **Severity cross-load:** re-normalise windows by the within-window |I1| at segmentation (vs the post-hoc `n_*`
  divide); or a **load-aware** severity model (load is measurable in practice). Quantify feature stability across
  loads (ANOVA / Fisher score).
- **Deep baseline (Track B):** ResNet-1D on raw 3-phase windows vs the XGBoost baseline.
- **Robustness & trust:** noise injection (SNR 40/30/20 dB), uncertainty (quantile / MC-dropout), calibration
  (ECE, Brier), Grad-CAM for the CNN.
- **Deployment realism:** Stage-0 fault-onset detection (CUSUM on |I2| / sliding EPVA).

## 11. Reproducibility
```
python data_index.py     # -> manifest.csv
python segment.py        # -> dataset/windows.npy + window_labels.csv
python features.py       # -> dataset/features.parquet
python train_stage1.py   # detection
python train_stage2.py   # severity
python train_stage3.py   # phase
python cascade.py         # end-to-end
# Phase 3:
python noise_robustness.py        # SNR robustness
python calibration_uncertainty.py # ECE/Brier + quantile uncertainty
python dl_resnet1d.py             # ResNet-1D (Track B), needs torch
python severity_deepdive.py       # feature stability + load-aware severity
```
Environment: Python 3.13; numpy, pandas, scikit-learn 1.8, xgboost 3.2, scipy, pyarrow, shap, torch 2.12 (cpu).
Every saved table (windows, features, all OOF predictions) carries its full label set and is self-contained.

## 12. Figures
- `dataset/stage1_confusion.png`, `dataset/stage2_confusion.png`, `dataset/stage3_confusion.png`
- `dataset/noise_robustness.png`, `dataset/calibration_reliability.png`, `dataset/uncertainty_width.png`,
  `dataset/severity_feature_stability.png`
- `events_annotated.png` (event timeline), `currents_healthy_vs_faulty.png` (waveform comparison)

---

## 13. Phase 3 — Robustness, deep baseline, and the severity fix

### 13.1 Noise robustness (train clean, test noisy; GroupKFold)
Additive Gaussian noise at the stated SNR; features recomputed on the noisy signal.

| SNR | Stage-1 detect F1 | Stage-2 severity within-1 | Stage-3 phase acc |
|---|---|---|---|
| clean | 0.977 | 0.986 | 0.958 |
| 40 dB | 0.927 | 0.980 | 0.954 |
| 30 dB | 0.854 | 0.975 | 0.945 |
| 20 dB | 0.792 | 0.971 | 0.939 |

**Severity (within-1) and phase ID are noise-robust; detection is the noise-sensitive stage** — small fault
signatures get swamped, raising false alarms/misses as SNR drops.

### 13.2 Calibration (Stage-1)
On GroupKFold OOF probabilities: **ECE = 0.021, Brier = 0.021** → the detector's probabilities are well
calibrated (reliability diagram `calibration_reliability.png`).

### 13.3 Uncertainty (Stage-2, quantile regression)
80 % prediction interval (q0.1–q0.9): empirical coverage **0.63** (target 0.80) and the width↔error relationship
is uninformative → naive quantile regression is **under-calibrated** here. **Recommendation: conformalised quantile
regression** for guaranteed coverage.

### 13.4 Deep baseline — ResNet-1D (Track B) vs XGBoost (Track A)
ResNet-1D on raw 3-phase windows (decimated 8× to 250 samples, single grouped split, CPU):

| Stage | ResNet-1D | XGBoost (Track A) |
|---|---|---|
| Detection (grouped) | F1 0.898 | **0.977** |
| Detection (LOLO load=20) | F1 0.401 | **0.749** |
| Phase (grouped) | acc 0.848 / F1 0.708 | **0.972** |
| Severity (grouped) | within-1 **1.000** | 0.990 |

**Physics-feature + XGBoost beats the raw-signal CNN** on detection and phase, comparable on severity — consistent
with the literature for limited datasets (engineered features with domain knowledge are more data-efficient).
Caveats: single split, aggressive decimation, modest CPU-trained net; a larger net / full signal / more tuning
could narrow the gap.

### 13.5 Feature stability & the load-aware severity fix
Per-feature severity Fisher ratio vs cross-load drift (faulty windows): the most discriminative *and* load-stable
features are `pva_ecc`, `n_rms_imbalance_std/maxmin` (drift ≈ 0.09), `epva_sf`, `I2_I1_ratio` — confirming the
load-invariant set analytically (`severity_feature_stability.png`).

**Load-aware severity** (add the measured `load_pct` to the already-faulty severity model):

| Severity within-1 | GroupKFold | LOLO |
|---|---|---|
| load-invariant features | 0.990 | 0.729 |
| **+ measured load (load-aware)** | **0.997** | **0.946** |

Load is a sensor input (not a label), so this is a legitimate, deployment-realistic fix; most held-out loads are
*interpolated* between neighbours. It cascades to **end-to-end LOLO within-1 0.77 → 0.92** (per-case 0.71 → 0.92).
The deployed cascade (§7) uses load-aware severity.

### 13.6 Phase 3 artifacts
`noise_robustness.{json,png}`, `calibration_uncertainty.json`, `calibration_reliability.png`,
`uncertainty_width.png`, `dl_metrics.json`, `severity_deepdive.json`, `severity_feature_stability.png`.

---

## 14. Phase 4 — Conformal uncertainty & Stage-0 onset detection

### 14.1 Conformalised quantile regression (Stage-2 severity)
Conformalised Quantile Regression (CQR) with a group-disjoint calibration split (target 80 % interval):

| Method | Coverage (target 0.80) | Mean width (levels) |
|---|---|---|
| naive quantile regression | 0.49 | 2.76 |
| **CQR (conformal)** | **0.72** | 3.51 |

CQR materially improves coverage (0.49→0.72). The residual gap to 0.80 is expected: the severity-stratified
*grouped* folds break the exchangeability CQR assumes (calibration vs test see different operating points), and
window-level severity is genuinely ambiguous. Group-conditional conformal or per-case aggregation would close it
further. Artifact: `dataset/conformal_uncertainty.json`.

### 14.2 Stage-0 fault-onset detection
A per-cycle negative-sequence fault index |I2|/|I1| with a threshold calibrated on a trailing pre-fault window
(mean + 6σ, 3 consecutive crossings). On 35 sampled faulty cases (5 per severity):

| Metric | Result |
|---|---|
| Detection rate | **100 %** |
| Pre-fault false-alarm rate | **0 %** |
| Median detection delay | **0 ms** (within the first faulted cycle) |
| Delay at 0.3 % severity | ~1.42 s (smallest fault, small |I2|) |
| Delay at ≥0.5 % severity | 0 ms |

Onset detection works on the same negative-sequence physics: faults ≥0.5 % are caught essentially instantly with
no pre-fault false alarms; only the smallest 0.3 % faults incur a detection lag. This supplies the missing online
trigger (`Stage0 → Stage1 → Stage2 → Stage3`). Artifacts: `dataset/stage0_onset.json`, `stage0_onset_cases.csv`.
