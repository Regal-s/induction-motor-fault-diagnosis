# Detailed ML Plan (Research-Grounded) — Hierarchical Stator Inter-Turn Fault Diagnosis

**Task:** From three-phase stator current, build a 3-stage cascade —
**(1) Detection** healthy vs faulty → **(2) Severity** % shorted turns (0.3/0.5/1/2/3/4/5) → **(3) Faulted-phase** A/B/C.
**Data:** only `Healthy case data` + `TT fault Data`. 50 Hz, 25 kHz, fault at sample 180,000 (7.2 s).

This version incorporates a literature survey (≈50 papers, 2016–2026). Citations are inline as `[KEY]`; full list in §13.

---

## 0. What the literature changed vs. the first plan
1. **Add EPVA severity factor as the headline severity feature** — amplitude of the 2f₁ (100 Hz) component of the Park's-vector modulus ÷ its DC level. It is the single most robust, load-normalised inter-turn severity scalar `[Cardoso-EPVA, EPVA-DFIG]`.
2. **Faulted phase = angle of negative-sequence current ∠I₂** — clusters into three ~120°-separated sectors, one per phase; load-insensitive (magnitude gives severity, angle gives location) `[Sensors2025, NSC-Compensation]`.
3. **Compensate the negative-sequence current** by subtracting each case's own **pre-fault healthy baseline I₂** — we already have a matched-load healthy segment in every TT file, which makes compensation trivial and removes voltage-unbalance/native-asymmetry false positives `[NSC-Phasor2022, NSC-JEET2021]`.
4. **Severity = ordinal regression, not flat 7-class** — errors are overwhelmingly between adjacent levels; report MAE in %-turns + quadratic-weighted kappa + ±1-level accuracy `[CNN-LSTM-MultiLevel, Energies2018-ANN]`.
5. **Data-leakage discipline is the #1 credibility factor** — literature shows leaky window-splits report 95–100% while group-correct splits report 66–90% on comparable problems `[HowYouSplit, Leakage-Alz2025, CWRU-bench]`. Split by **case**, never by window.
6. **Normalisation must preserve inter-phase imbalance** — shared scale across the 3 phases, per-recording; never per-channel/per-window (that erases the fault signature) `[CNN-DQ2025, SymComp-CNN2025]`.
7. **Deep model of choice = ResNet-1D** (large first-layer kernel ≈ 1 cycle) + optional TCN/attention head; classical **RF/XGBoost on engineered features** as the interpretable, leakage-resistant baseline `[Ince2016, ResNet-BiGRU2025, ITSC-1DResNet2025]`.
8. **Decimate for the feature/spectral track** — ITSC content lives < ~2–3 kHz; 25 kHz is overkill there (keep full rate only for raw-CNN track).

---

## 1. Problem framing — cascade with an MTL option
- **Primary (as requested): cascade.** Stage 1 gate (healthy/faulty) → on the faulty branch, Stage 2 (severity) and Stage 3 (phase). Cascade is the literature-recommended structure when severity/location are only *defined given a fault exists* `[Fault-MTL, KAN-Severity]`.
- **Enhancement to evaluate:** replace the two faulty-branch heads with a **multi-task block** (shared encoder, separate heads for severity + phase). MTL improves both tasks via shared representation; use uncertainty weighting to balance the ordinal (severity) and nominal (phase) losses `[MTL-GNN-Grid, Fault-MTL]`.
- **Reject:** a single flat multiclass over 1 + 7×3 = 22 labels (ignores ordinal severity, imbalances classes).

```
window ─▶ Stage1 ─ Healthy ─▶ "Healthy"
                 └ Faulty  ─▶ Stage2 severity (ordinal) ┐
                            └ Stage3 phase (A/B/C)       ┴▶ (severity, phase)
```

---

## 2. Data inventory, labels, and segment sourcing

| Source | Cases | Class / labels |
|---|---|---|
| Healthy (21 loadings, 5 s) | 21 | healthy |
| TT regular (sev×phase×load) | 126 | faulty + severity + phase + load |
| TT "Extra" (×10 inception times) | 210 | faulty + severity + phase + load + inception |

**Segment windows from (samples @ 25 kHz):**
- **Faulty:** fault interval **180,000–200,000** (0.8 s ≈ 40 cycles) of each TT file.
- **Healthy:** (a) steady region of the 21 healthy files (post-load-change ~100k–125k); **(b) the pre-fault steady region of every TT file (~110k–178k)** — genuinely healthy before 7.2 s.

> **Why (b) is pivotal (two wins):**
> 1. Gives healthy data at the **same loads** as the faulty windows → Stage 1 can't cheat on load `[load-confound: ExpSysApp2023, Sensors2025]`.
> 2. Serves as the **per-case healthy baseline** for negative-sequence/EPVA compensation (§4) — a built-in advantage of this dataset.

**Confound to respect:** healthy files use 5% load steps; TT uses NL/20/40/60/80/100%. Do **not** pair classes by load; rely on TT pre-fault windows for matched-load healthy data.

---

## 3. Preprocessing & windowing
1. Load `.out` with `np.loadtxt` (skips blank first line). Use **`_03.out` cols 2,3,4** (`Ia_ABCIM`, kA — physical magnitudes preserve imbalance). Keep `_02` normalised currents as a cross-check only.
2. Trim ±1 cycle around start/load-change/fault edges to avoid transients contaminating steady windows (keep a separate "transient" set if you later want start-up-based features `[StartupEnergy-PMC]`).
3. **Windowing:** primary length **5,000 samples = 10 cycles (200 ms)**; sweep {1, 4, 10, 20} cycles. Overlap 50% **but never across a train/test boundary**.
4. **Decimation:** for the feature/spectral track, low-pass + decimate to ~2–5 kHz (anti-alias) to cut dimensionality; raw-CNN track keeps 25 kHz (or mild decimation).
5. **Normalisation (critical):** divide all three phases by **one shared scale per recording** — recommended = fundamental (50 Hz) amplitude of the positive-sequence current, or per-recording RMS of `|i_P|`. This reduces load dependence while **retaining inter-phase imbalance** `[CNN-DQ2025]`. Fit any feature StandardScaler on the **train split only**.

Expected windows: faulty ≈ 336 × ~19 ≈ 6.4k; healthy (pre-fault) ≈ 20k+. Subsample healthy to balance Stage 1.

---

## 4. Feature engineering — Track A (classical, do first)
Ranked by literature-reported discriminative power. Compute per window.

**Tier 1 — physics-based, most discriminative**
- **EPVA severity factor** `SF = A(2f₁) / DC` of the Park's-vector modulus `|i_P| = √(i_d²+i_q²)` (Clarke transform). Headline severity feature; load-normalised `[Cardoso-EPVA]`.
- **Negative-sequence current** from Fortescue on 50 Hz phasors (1-cycle DFT/Goertzel per phase): `|I₂|`, ratio **`|I₂|/|I₁|`** (severity), and **`∠I₂`** (faulted phase). Use the **compensated** value `I₂_comp = I₂_window − I₂_baseline(case pre-fault)` `[Sensors2025, NSC-Phasor2022]`.
- **Sequence-impedance off-diagonal** `Z₂₁ = V₂/I₁` (if you also pull voltages from `_01/_02`) — inherently robust to supply unbalance `[NSC-JEET2021]`.

**Tier 2 — spectral / Park geometry**
- **MCSA** amplitudes at Penman frequencies `f_st = f₁·[k ± n(1−s)/p]` (k odd, n=1,2,…), and **3rd-harmonic ratio** (150 Hz / 50 Hz) `[Penman1994, Harmonic-CompStudy2022]`.
- **PVA ellipse features:** major/minor semi-axes, **eccentricity** `e=√(1−b²/a²)`, **tilt angle** (tilt tracks faulted phase).

**Tier 3 — statistics (fill out the bank, then prune)**
- Per-phase RMS, THD, peak, crest factor, skewness, kurtosis — and crucially their **inter-phase differences** (encode asymmetry).
- Same stats on `|i_P|` and on wavelet sub-bands (add wavelet/DWER only if transient operation matters `[DWT-ITSC, HHT-ITSC]`).

**Selection:** rank with **mRMR / ReliefF**, optionally **PCA-compress**, then feed RF/XGBoost/SVM `[FeatureSel-IET2018, PCA-SVM]`.

## 4b. Feature engineering — Track B (deep)
- **Raw track:** tensor `[3 × window]` (kA, shared-scale normalised); optionally append symmetrical-component or d-q channels `[CNN-DQ2025, SymComp-CNN2025]`.
- **Image track:** per-phase **CWT scalogram** or **STFT spectrogram** (0–~500 Hz band) → `[3×F×T]`, or **symmetrical-component feature images** / Park-vector images for a pretrained 2D-CNN `[BRB-CNN-Compare2023, SymComp-CNN2025]`.

---

## 5. Models per stage

| Stage | Track A baseline | Track B (DL) | Output / loss |
|---|---|---|---|
| 1 Detection | XGBoost / RF / SVM-RBF | ResNet-1D | binary; class-weighted BCE |
| 2 Severity | **XGBoost ordinal / GBM regression** | ResNet-1D (+TCN head) | **ordinal regression**; predict continuous %, then bin |
| 3 Phase ID | RF / XGBoost (3-class) | ResNet-1D / 2D-CNN | softmax A/B/C (or multi-label sigmoids to allow multi-phase later) |

**ResNet-1D reference config** `[Ince2016, ITSC-1DResNet2025]`: 3-channel input → conv stem (kernel ≈ 500 ≈ 1 cycle, or stacked 7–15 kernels) → 4 residual stages (16→32→64→128 ch, BN+ReLU, stride/pool downsample) → global average pool → dense → head. Optional **TCN** (dilated causal convs) or light **attention** head for long-range/temporal severity cues `[CNN-TCN-Attn2025, ImprovedCNN-Transformer2025]`. Transformers only as a small head (data-hungry).

---

## 6. Train / validation / test split — anti-leakage (most important section)
- **Group = case_id; split by case, never by window.** Windows from one `.out` are near-duplicates `[HowYouSplit, Leakage-Alz2025]`.
- Keep all **10 inception-time variants** of a given (sev, phase, load) **in the same fold** (near-duplicate cases).
- **Two protocols, both reported:**
  1. **StratifiedGroupKFold (5-fold)** by case — in-distribution headline.
  2. **Leave-One-Load-Out (LOLO)** — train on 5 loads, test on the held-out load. This is the honest generalisation test and should be the *headline* number `[load-confound, DG-PHM]`.
- Any resampling/SMOTE/GAN augmentation happens **inside the training fold only**.
- **Sanity rule:** a near-100% result is a leakage red flag — re-audit the split `[CWRU-bench]`.

---

## 7. Class imbalance & augmentation
- Stage 1: subsample healthy or use class weights / focal loss.
- Severity/phase: roughly balanced already (≈48 cases/severity, ≈112/phase); use class weights if needed.
- **Augmentation (training folds only):** jitter (additive noise), amplitude scaling, time-shift, window-slicing — **scale all 3 phases jointly** so imbalance is preserved. For DL on scarce data, consider **1D WAC-GAN / TimeGAN** synthetic 3-phase current `[WAC-GAN, TimeGAN-Motor]`. Validate synthetics don't leak into test.

---

## 8. Evaluation protocol & metrics
- **Stage 1:** accuracy, **macro-F1**, precision/recall, ROC-AUC, confusion matrix — reported **per load**.
- **Stage 2 (ordinal):** **MAE in severity levels**, quadratic-weighted kappa, ±1-level (adjacent) accuracy, confusion matrix (expect adjacent-level errors) `[CNN-LSTM-MultiLevel]`.
- **Stage 3:** 3-class accuracy + per-phase F1 + confusion matrix.
- **End-to-end cascade:** exact-match of (state, severity, phase); per-stage error attribution; Stages 2–3 evaluated **on Stage-1's faulty predictions**.
- Report **mean ± std across folds**, for **both GroupKFold and LOLO**, with per-load and per-severity breakdowns.
- **Interpretability/leakage guard:** SHAP / feature importance (Track A) — confirm the model leans on `|I₂|/|I₁|`, EPVA-SF, harmonics, not on a load proxy `[FeatureSel-IET2018]`.
- **Per-case aggregation:** majority vote (phase) / median (severity) over a case's windows for a robust per-case decision.

---

## 9. Generalisation strategy (beyond standard CV)
- LOLO is the minimum bar. If deployment loads may be unseen, add a **domain-generalisation/adaptation** baseline (adversarial feature alignment / MMD, or domain-augmented training) `[CrossDomainAug2023, DG-PHM]`.
- Physics-based features (EPVA-SF, compensated `|I₂|/|I₁|`) generalise better across loads than raw amplitudes — prefer them for the cross-load story.

---

## 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Window leakage → inflated accuracy | Group split by case; LOLO; suspect ~100% |
| Load confound in Stage 1 | TT pre-fault windows as matched-load healthy |
| Voltage-unbalance / native asymmetry false positives | Compensated NSC (subtract per-case baseline); EPVA-SF |
| Per-channel normalisation hides imbalance | Shared-scale per-recording normalisation |
| Severity adjacent-class confusion | Ordinal regression + MAE/kappa + ±1 tolerance |
| Few distinct cases → DL overfit | Classical baseline first; augmentation; case-level CV; regularise |
| Sim-only (no hardware) | State as limitation; keep features physically interpretable |
| `_02` vs `_03` current mixing | Fix `_03` (kA) as primary |

---

## 11. Implementation roadmap (file layout & milestones)
1. `data_index.py` — manifest of every case: `case_id, path_03, state, severity, phase, load, inception, n_samples, regions`.
2. `segment.py` — region rules + window params → windowed arrays (`.npy`/parquet) with labels + `group=case_id`.
3. `features.py` — Track A: Clarke/Park + EPVA-SF, Fortescue sequence comps (+compensation), MCSA lines, PVA ellipse, per-phase stats + inter-phase diffs.
4. `splits.py` — StratifiedGroupKFold + LeaveOneLoadOut.
5. `train_stage1.py` / `train_stage2.py` / `train_stage3.py` — XGBoost baselines + CV; leakage audit.
6. `cascade.py` — wire stages, end-to-end metrics, confusion matrices, SHAP, per-case vote.
7. `dl/resnet1d.py` (+ optional `tcn.py`, `cnn2d_tf.py`) — Track B; same splits/metrics.
8. `report.md` — tables (GroupKFold + LOLO) + figures.

**Milestones:** M1 manifest+windows · M2 features · M3 Stage-1 baseline + leakage/LOLO check · M4 Stages 2&3 · M5 cascade + per-case + LOLO report · M6 ResNet-1D vs baseline · M7 (opt) MTL faulty-branch + domain-generalisation.

---

## 12. Default choices (revisable)
Current source `_03` kA · window 10 cycles / 50% overlap · shared-scale per-recording norm · classical XGBoost before DL · severity = ordinal regression · ResNet-1D as DL backbone · cascade primary, MTL faulty-branch as enhancement · headline metric = LOLO macro-F1 (Stage1), MAE (Stage2), per-phase F1 (Stage3).

---

---

# 14. Additional Recommended Enhancements (Extended Review Suggestions)

The following enhancements are recommended to further strengthen the robustness, industrial relevance, and publication quality of the project.

## 14.1 Add Fault Onset Detection Stage
Current pipeline assumes the fault interval is already known. In practical deployment, the system must first determine **when the fault begins**.

### Recommended extension
Add:

```text
Stage 0 → Fault onset detection
Stage 1 → Healthy/Faulty
Stage 2 → Severity estimation
Stage 3 → Faulted phase identification
```

### Possible methods
- CUSUM on `|I₂|`
- Change-point detection
- Moving RMS divergence
- Sliding EPVA threshold
- Autoencoder reconstruction error

This significantly improves deployment realism and industrial applicability.

---

## 14.2 Add Uncertainty Estimation
Very small severity levels (0.3%, 0.5%, 1%) may overlap under varying load/noise conditions.

Instead of only predicting:

```text
Severity = 1%
```

estimate:

```text
Severity = 1% ± confidence
```

### Recommended approaches
**Classical ML**
- Probabilistic XGBoost
- Quantile regression

**Deep Learning**
- Monte Carlo dropout
- Evidential deep learning

This improves trustworthiness of predictions for practical deployment.

---

## 14.3 Temporal Consistency at Case Level
Current predictions are window-based with majority voting. A better approach is to enforce temporal consistency over sequential windows.

### Recommended methods
- Hidden Markov Models (HMM)
- Conditional Random Fields (CRF)
- Temporal smoothing
- Confidence filtering

This prevents unrealistic predictions such as:

```text
0.5 → 5 → 1 → 4
```

and improves stability of severity estimation.

---

## 14.4 Evaluate Both Fine-Grained and Coarse-Grained Severity Labels
Distinguishing among:

```text
0.3%, 0.5%, 1%
```

may be physically difficult under noisy or unseen operating conditions.

### Recommended comparison

#### Fine-grained labels
```text
0.3, 0.5, 1, 2, 3, 4, 5
```

#### Coarse-grained labels
```text
Incipient = {0.3, 0.5, 1}
Moderate = {2, 3}
Severe = {4, 5}
```

Compare:
- LOLO performance
- robustness to noise
- uncertainty overlap
- practical deployability

This can become an important discussion section in the final report.

---

## 14.5 Add Cross-Inception Generalisation Benchmark
Currently, inception-time variants are grouped together inside folds.

### Additional benchmark
**Leave-One-Inception-Out**

Train on:
- 9 inception timings

Test on:
- unseen inception timing

This evaluates robustness against variations in:
- waveform discontinuity
- transient harmonics
- fault initiation angle

A strong result here significantly improves generalisation credibility.

---

## 14.6 Robustness and Noise Experiments
Current data is simulation-clean. Industrial environments contain:
- measurement noise
- sensor offsets
- gain mismatch
- sampling jitter

### Recommended evaluation
Inject synthetic noise levels:

```text
SNR = 40 dB, 30 dB, 20 dB
```

Then compare robustness of:
- EPVA features
- compensated NSC
- classical ML
- CNN models

This strengthens practical relevance considerably.

---

## 14.7 Additional Deep Learning Regularisation
Dataset size remains limited after leakage-safe splitting. Strong regularisation is therefore essential.

### Recommended additions
- Dropout
- Early stopping
- Weight decay
- Label smoothing
- MixUp/CutMix for 1D signals

Without these, deep models may overfit despite high apparent validation accuracy.

---

## 14.8 Model Calibration Metrics
Fault diagnosis systems require trustworthy probabilities.

### Recommended calibration metrics
- Expected Calibration Error (ECE)
- Reliability diagrams
- Brier score

Example:

```text
95% confidence should correspond to ≈95% actual correctness.
```

Calibration is particularly important for Stage 1 fault detection.

---

## 14.9 Add Physics-Threshold Baseline
Alongside ML baselines, include a simple rule-based baseline.

### Example
```python
if |I2|/|I1| > threshold:
    faulty
```

Purpose:
- demonstrates whether ML is truly necessary
- shows where classical thresholds fail
- highlights advantages for incipient-fault detection

This strengthens the scientific contribution.

---

## 14.10 Feature Stability Analysis Across Loads
The current plan states that EPVA and compensated NSC generalise better across loads, but this should be quantitatively demonstrated.

### Suggested analysis
For each feature compute:
- intra-class variance across loads
- inter-class separation
- Fisher score
- ANOVA statistics

Expected outcome:
- EPVA and NSC remain stable
- raw RMS/amplitude features vary strongly with load

This becomes a strong analytical result section.

---

## 14.11 Explainability for Deep Learning Models
Current explainability focuses mainly on classical ML (SHAP).

### Recommended DL explainability tools
- Grad-CAM for 1D CNNs
- Saliency maps
- Integrated gradients

These help identify:
- important waveform regions
- harmonic bands
- temporal fault signatures

Useful for thesis quality and reviewer confidence.

---

## 14.12 Recommended Project Prioritisation
To avoid excessive project complexity, execution should remain modular.

### Phase 1 (Highest Priority)
- Segmentation
- Leakage-safe splitting
- EPVA + compensated NSC features
- XGBoost baseline
- LOLO evaluation

### Phase 2
- Severity ordinal regression
- Faulted-phase identification
- SHAP analysis

### Phase 3
- ResNet-1D implementation
- Noise robustness
- Uncertainty estimation

### Phase 4 (Optional / Publication-grade)
- Domain adaptation
- Multi-task learning
- Transformer heads
- GAN augmentation

---

## 14.13 Most Critical Risk
The largest technical risk is that the model may unintentionally learn:
- load conditions
instead of:
- true fault physics.

Even with precautions, this risk remains significant.

Therefore:

### Leave-One-Load-Out (LOLO) should be treated as the primary benchmark,
not merely a secondary evaluation protocol.

LOLO performance is the strongest indicator of genuine generalisation capability.

---

## 14.14 Final Assessment
The overall framework is:
- technically sound
- research-aligned
- physically interpretable
- suitable for high-quality academic work

If executed carefully with leakage-safe evaluation and robust cross-load testing, the work has strong potential for:
- a high-quality BTP/MTech thesis,
- conference publication,
- or journal-level extension.

