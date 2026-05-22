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

## 13. References (key)
- **[Ince2016]** Ince et al., *Real-Time Motor Fault Detection by 1-D CNNs*, IEEE TIE 2016. https://ieeexplore.ieee.org/document/7501527/
- **[ITSC-1DResNet2025]** *Early Detection of ITSC Using Derivative of Stator Current + 1D-ResNet*, Computation 13(6):140, 2025. https://www.mdpi.com/2079-3197/13/6/140
- **[Sensors2025]** Ruzimov et al., *ITSC Detection via Negative-Sequence Current Analysis (DWT+sym-comp+FDI)*, Sensors 25(15):4844, 2025. https://www.mdpi.com/1424-8220/25/15/4844
- **[NSC-Phasor2022]** *Negative Sequence Current Phasor Compensation for Stator Shorted-Turn Detection*, Energies 15(9):3100, 2022. https://www.mdpi.com/1996-1073/15/9/3100
- **[NSC-JEET2021]** *NSC Compensation & Off-Diagonal Sequence-Impedance Term as Fault Indicator*, JEET 16:2075, 2021. https://link.springer.com/article/10.1007/s42835-021-00730-8
- **[Cardoso-EPVA]** Cardoso et al., *Stator winding fault diagnosis by the Extended Park's Vector Approach*, IEEE TIA 2001. https://ieeexplore.ieee.org/document/952496/
- **[EPVA-DFIG]** *Extended Park's vector for early ITSC in DFIG stator windings*, IET GTD 2020. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-gtd.2020.0127
- **[Penman1994]** Penman et al., *Detection of inter-turn short circuits in stator windings*, IEEE TEC 1994 (characteristic-frequency formula).
- **[SymComp-CNN2025]** *CNN Fault Classification Using Feature-Vector Images of Symmetrical Components*, Electronics 14(8):1679, 2025. https://www.mdpi.com/2079-9292/14/8/1679
- **[CNN-DQ2025]** *CNN Input via D-Axis Current in D-Q Frame (load-robust)*, Applied Sciences 15(15):8380, 2025. https://www.mdpi.com/2076-3417/15/15/8380
- **[CNN-LSTM-MultiLevel]** *Multi-Level Short-Circuit Fault Detection using CNN-LSTM*, Eng 7(2):94, 2026. https://www.mdpi.com/2673-4117/7/2/94
- **[Energies2018-ANN]** *Efficient Stator Inter-Turn Fault Diagnosis Tool (ANN severity)*, Energies 11(3):653, 2018. https://www.mdpi.com/1996-1073/11/3/653
- **[ExpSysApp2023]** *IM short-circuit diagnosis under voltage unbalance & load variation*, ESWA 2023. https://www.sciencedirect.com/science/article/abs/pii/S0957417423005006
- **[Harmonic-CompStudy2022]** *Comparative study: ITSC detection via harmonic analysis*, Math. & Comp. in Simulation 2022. https://www.sciencedirect.com/science/article/abs/pii/S0378475422000325
- **[ResNet-BiGRU2025]** *HV 3-phase async motor diagnosis: ResNet+Bi-GRU*, Eng. Appl. AI 2025. https://www.sciencedirect.com/science/article/abs/pii/S0952197625020573
- **[CNN-TCN-Attn2025]** *Motor Fault Diagnosis: CNN-TCN-Attention*, CAICE 2025. https://dl.acm.org/doi/10.1145/3727648.3727797
- **[ImprovedCNN-Transformer2025]** *Improved CNN-Transformer (multi-scale cross-attention)*, IJDC 2025. https://link.springer.com/article/10.1007/s40435-025-01938-6
- **[BRB-CNN-Compare2023]** *Comparative Analysis of CNN Architectures for Broken Rotor Bars*, Sensors 23(19):8196, 2023. https://www.mdpi.com/1424-8220/23/19/8196
- **[Fault-MTL]** *Fault-MTL: Multi-task DL for fault classification + localization in power systems*, 2024. https://www.researchgate.net/publication/383046116
- **[MTL-GNN-Grid]** *Heterogeneous Graph Multi-Task Learning for Smart-Grid fault diagnosis*, arXiv 2309.09921, 2023. https://arxiv.org/html/2309.09921v2
- **[KAN-Severity]** *Explainable Fault Classification & Severity via Kolmogorov-Arnold Networks*, 2024/25. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12025949/
- **[HowYouSplit]** *How You Split Matters: Data Leakage & Subject Characteristics*, arXiv 2309.00350, 2023. https://arxiv.org/abs/2309.00350
- **[Leakage-Alz2025]** *Data Leakage in DL for Alzheimer's (scoping review)*, Diagnostics 15(18):2348, 2025. https://www.mdpi.com/2075-4418/15/18/2348
- **[CWRU-bench]** *Benchmarking DL for bearing fault diagnosis (leakage pitfall)*, arXiv 2407.14625, 2024. https://arxiv.org/html/2407.14625v1
- **[CrossDomainAug2023]** *Adversarial domain-augmented generalization for unseen conditions*, RESS 2023. https://www.sciencedirect.com/science/article/abs/pii/S0951832023000868
- **[DG-PHM]** Curated domain-generalization fault-diagnosis repo. https://github.com/CHAOZHAO-1/DG-PHM
- **[WAC-GAN]** *WAC-GAN augmentation for inverter fault diagnosis (1D conv, 3-phase)*, RESS 2023. https://www.sciencedirect.com/science/article/abs/pii/S0951832023002740
- **[TimeGAN-Motor]** *Time-Frequency Conditional Transformer-TimeGAN for Motor Fault Augmentation*, Machines 13(10):969, 2025. https://www.mdpi.com/2075-1702/13/10/969
- **[FeatureSel-IET2018]** Haroun et al., *Multiple feature extraction & selection for stator winding faults*, IET EPA 2018. https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-epa.2017.0457
- **[PCA-SVM]** *Non-invasive IM fault diagnosis: PCA-optimized FFT + SVM*, Measurement 2026. https://www.sciencedirect.com/science/article/pii/S0263224126001107
- **[DWT-ITSC]** *IM stator ITSC detection using DWT*; **[HHT-ITSC]** *Fourier/HHT/DWT for ITSC detection*; **[StartupEnergy-PMC]** *Start-up current envelope energy*, Sensors/PMC10611028.

> **Source caveat:** several quantitative figures come from search summaries (publisher PDFs were paywalled to the research agents). Verify exact accuracies/conventions (esp. Penman frequency set and Park scaling) against primary PDFs before citing in a paper.

---

# 14. Additional Recommended Enhancements (Extended Review — folded in from `Updated_Fault_Detection_Readme`)
Reviewer suggestions to strengthen robustness, industrial relevance, and publication quality. **Adopted into the roadmap (see revised prioritisation §14.12).**

**14.1 Stage 0 — Fault-onset detection.** Pipeline currently assumes the fault window is known. For deployment, first detect *when* the fault starts. Methods: CUSUM on `|I₂|`, change-point detection, moving-RMS divergence, sliding-EPVA threshold, autoencoder reconstruction error. New chain: `Stage0 onset → Stage1 healthy/faulty → Stage2 severity → Stage3 phase`.

**14.2 Uncertainty estimation.** Low severities (0.3/0.5/1%) overlap under load/noise → predict `severity ± confidence`. Classical: probabilistic XGBoost, quantile regression. DL: MC-dropout, evidential DL.

**14.3 Temporal consistency at case level.** Beyond per-window majority vote, enforce smoothness over sequential windows (HMM, CRF, temporal smoothing, confidence filtering) to avoid jumpy sequences like `0.5→5→1→4`.

**14.4 Fine- vs coarse-grained severity.** Compare fine `{0.3,0.5,1,2,3,4,5}` vs coarse `Incipient{0.3,0.5,1} / Moderate{2,3} / Severe{4,5}` on LOLO performance, noise robustness, uncertainty overlap, deployability. Good discussion-section material.

**14.5 Leave-One-Inception-Out (LOIO) benchmark.** In addition to grouping inception variants within folds, run train-on-9-inception / test-on-unseen-inception to test robustness to fault initiation angle / transient harmonics / waveform discontinuity.

**14.6 Noise / robustness experiments.** Data is sim-clean; inject noise at **SNR = 40/30/20 dB** (plus sensor offset, gain mismatch, sampling jitter) and compare robustness of EPVA, compensated NSC, classical ML, and CNN.

**14.7 Stronger DL regularisation.** Dataset is small after leakage-safe splitting → dropout, early stopping, weight decay, label smoothing, MixUp/CutMix for 1D signals.

**14.8 Model calibration.** Report ECE, reliability diagrams, Brier score (esp. Stage 1) — confidence should match accuracy.

**14.9 Physics-threshold baseline.** Add a rule-based baseline (`if |I₂|/|I₁| > thr: faulty`) alongside ML to show whether ML is truly needed and where simple thresholds fail (incipient faults).

**14.10 Feature-stability analysis across loads.** Quantify (don't just claim) load-robustness: intra-class variance across loads, inter-class separation, Fisher score, ANOVA. Expect EPVA/NSC stable, raw RMS/amplitude load-sensitive.

**14.11 DL explainability.** Grad-CAM (1D-CNN), saliency maps, integrated gradients to surface important waveform regions / harmonic bands (complements SHAP on Track A).

**14.12 REVISED prioritisation (modular execution):**
- **Phase 1 (highest):** segmentation · leakage-safe splitting · EPVA + compensated-NSC features · XGBoost baseline · **LOLO evaluation**.
- **Phase 2:** severity ordinal regression · faulted-phase ID · SHAP.
- **Phase 3:** ResNet-1D · noise robustness · uncertainty estimation.
- **Phase 4 (publication-grade, optional):** domain adaptation · multi-task learning · transformer heads · GAN augmentation · Stage-0 onset.

**14.13 Most critical risk:** model may learn *load* instead of *fault physics*. Therefore **LOLO is the PRIMARY benchmark, not a secondary one** — it is the strongest indicator of genuine generalisation. (This supersedes the "headline" framing in §6/§12.)

**14.14 Assessment:** framework is technically sound, research-aligned, physically interpretable; with leakage-safe + cross-load evaluation it suits a high-quality BTP/MTech thesis, conference paper, or journal extension.
