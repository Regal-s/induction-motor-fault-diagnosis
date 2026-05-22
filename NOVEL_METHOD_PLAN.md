# Novel Method Plan — IEEE-Transactions-Grade Stator Inter-Turn Fault Diagnosis

**Goal.** Convert the current applied cascade into a *method paper* with **genuine novelty in five axes**
(feature engineering, feature reduction, ML/DL model, transfer learning, data augmentation), validated on
**experimental public data** + our PSCAD simulation, benchmarked against **16 models**, under leakage-safe,
cross-load, and **sim-to-real** protocols. Grounded in a 2023–2026 literature survey (see
`papers/PAPERS_KNOWLEDGE_BASE.md` + research notes). Target venues: **IEEE TIM / TIA / TIE / T-PEL**.

---

## 0. One-sentence contribution
> *A physics-informed, load-invariant framework for induction-motor stator inter-turn diagnosis that (i)
> represents 3-phase current as a sequence-component Stockwell tensor, (ii) selects load-invariant features by
> an invariance-regularized criterion, (iii) classifies fault/type/phase and regresses calibrated severity with a
> phase-coupled, load-conditioned Mamba–KAN multi-task network, (iv) augments scarce regimes with a
> physics-constrained generative model, and (v) transfers simulation→experiment via physics-anchored domain
> adaptation — validated cross-load and sim-to-real on public experimental datasets.*

Each axis below is a *standalone* contribution; together they form a coherent system. The literature confirms each
is **unexplored for 3-phase stator inter-turn current** (most prior art lives on bearing vibration / PMSM / power flow).

---

## 1. NOVELTY AXIS 1 — Feature engineering: **Sequence-Component Stockwell Tensor (SCST)**
**Idea.** Decompose the 3-phase current into instantaneous **positive / negative / zero symmetrical-component**
streams, apply the **Stockwell (S-) transform** to each, and stack the three time-frequency maps as a 3-channel
tensor (RGB) input for 2-D models.
**Why novel (research-backed).** S-transform→deep-learning and *image-encoding of symmetrical-component channels*
are both essentially absent for stator inter-turn faults; the negative-sequence channel carries the
**load-invariant** inter-turn signature, and S-transform's frequency-dependent resolution tracks the 2f₀ EPVA line
and slip sidebands better than fixed STFT. (Gap confirmed by feature-engineering survey.)
**Variants to ablate:** (a) raw-phase Stockwell tensor (control), (b) SCST (proposed), (c) SCST with **δc/δv
normalization** (divide by the load-invariant complex unbalance coefficient before imaging → bakes load-robustness
into the input), (d) EPVA-modulus scalogram channel added.
**Deliverable:** `feat_scst.py` producing `(3, F, T)` tensors + the existing 72 scalar physics features for the ML track.

## 2. NOVELTY AXIS 2 — Feature reduction: **Load-Invariance-Regularized Selection (LIR-mRMR + neural gate)**
**Idea.** Replace plain mRMR/ReliefF with a three-term criterion that builds load-robustness into selection:
`J(f) = Relevance(f; label) − λ·Redundancy(f) − γ·LoadInstability(f)`,
where **LoadInstability(f)** = variance of the feature's class-conditional means across load bins (or inter-load MMD).
Also a **differentiable analogue**: an L1/attention feature-gate trained with a HSIC/MMD inter-load stability penalty
("invariance-regularized sparse gate").
**Why novel.** No named "load-invariance-regularized feature selection" exists; cross-load robustness is currently
handled only by opaque adversarial back-ends. LIR-mRMR yields a **sparse, interpretable, physics-traceable** subset —
a strong selling point for safety-critical monitoring. Directly extends our own finding that ratio/`n_*` features are
load-stable while magnitudes drift (we already have the Fisher-vs-drift analysis as preliminary evidence).
**Validation:** leave-load-out accuracy AND **selected-feature stability** vs mRMR/ReliefF/PCA/adversarial-DA.
**Deliverable:** `select_lir.py` (LIR-mRMR) + `gate_invariance.py` (neural gate).

## 3. NOVELTY AXIS 3 — Model: **PCM-Net (Phase-Coupled, Conditioned Mamba–KAN multi-task network)**
**Architecture (the core contribution):**
- **Stem:** multi-scale 1-D conv on raw 3-phase current (+ optional sequence-component / Park channels) → tokens.
- **Backbone:** **bidirectional selective state-space (Mamba) blocks** — linear-time long-sequence modeling, fresh for
  this domain (Mamba currently only on bearing vibration).
- **Phase-coupling module:** a small **graph/attention** over the three phase-streams to encode inter-phase
  asymmetry — physically the inter-turn signature (phases-as-nodes is unexploited here).
- **Load conditioning:** **FiLM layers driven by the measured operating point** (load/speed/supply) — *inject* the
  condition rather than adversarially erasing it. This is the key model novelty; "rethinking operating conditions"
  (arXiv 2506.17740) shows erasure is suboptimal, yet FiLM-conditioning is unused in motor fault diagnosis.
- **Multi-task heads:** (a) detection (binary), (b) fault-type (Healthy/TT/HRC/LL), (c) faulted-phase (A/B/C),
  (d) **severity via an interpretable KAN regression head** (learnable splines → transparency vs black-box MLP),
  with **uncertainty-based task weighting**.
- **Physics regularizer (PINN term):** loss penalizing violation of the sequence-component fault relation
  (`Isp/Isn` vs severity) and Kirchhoff current balance `‖i_a+i_b+i_c‖²` — ties the severity head to physics.
**Why novel.** The integration *SSM backbone + phase-graph prior + FiLM load-conditioning + multi-task KAN severity
head + PINN residual, on 3-phase stator inter-turn current* is unpublished (each ingredient exists separately, mostly
on vibration). Ablate every block to prove necessity.
**Deliverable:** `pcmnet.py` (PyTorch); ablations `pcmnet_ablate.py`.

## 4. NOVELTY AXIS 4 — Transfer learning: **PADA (Physics-Anchored Domain Adaptation), sim→real**
**Idea.** Train on our **PSCAD simulation** (rich, labeled: severity×phase×load×inception) and adapt to
**experimental** current via DANN/MMD/CORAL, but with **physics invariants (δc/δv, negative-sequence ratio,
sequence components) as domain-invariant anchors** in the alignment objective (anchor loss pulls source/target onto
the same physics manifold).
**Why novel.** Existing IM stator sim-to-real (Sci.Rep. 2025) uses hand-crafted features with *naïve* transfer — no
learned alignment; physics-anchored adversarial/MMD alignment for stator ITSC is open. Also do **cross-load** (LOLO),
**cross-severity**, and **cross-motor** (different HP) adaptation.
**Validation:** target-domain accuracy/F1, severity MAE, phase accuracy vs (a) no-transfer, (b) naïve fine-tune,
(c) plain DANN/MMD without physics anchor.
**Deliverable:** `pada.py` + `domain_datasets.py`.

## 5. NOVELTY AXIS 5 — Data augmentation: **PC-Diff (Physics-Constrained Conditional Generator)**
**Idea.** A **conditional diffusion (DDPM) or conditional-TimeGAN** generator of 3-phase current windows,
conditioned on `(load, slip, severity, phase)`, with physics enforced three ways:
1. **Hard current-balance** via hyperplane projection at each denoising step (transplant KCLNet idea) → generated
   `(i_a,i_b,i_c)` satisfy topology exactly.
2. **Soft sequence-component loss** — penalize deviation of generated **negative-sequence magnitude** from the
   physics-expected monotone function of severity (operate in αβ/sequence coords, **never per-phase independently**).
3. **Slip/harmonic consistency** — tie injected fault-harmonic positions to load-dependent slip → **physics-guided
   interpolation/extrapolation to unseen (load, severity) cells.**
**Why novel.** Physics-constrained generation exists only for *power-flow* data; the nearest motor competitor (SGDA,
arXiv 2506.08412) is heuristic FFT-peak injection with amplitude-as-severity, no generative model, no calibration.
PC-Diff is the **first sequence-component-consistent generative augmenter for motor current**, and it directly
addresses the classical-augmentation pitfall (independent per-phase jitter manufactures fake imbalance).
**Validation:** does PC-Diff (vs classical-αβ, WGAN-GP/WAC-GAN, conditional TimeGAN, plain DDPM, SGDA) raise
accuracy AND tighten calibrated severity intervals at **unseen** load×severity cells?
**Deliverable:** `pcdiff.py`.

## 5b. BONUS NOVELTY — Calibrated severity intervals: **Mondrian-CQR**
Conformalized Quantile Regression with **group-conditional (Mondrian) calibration per load bin** → distribution-free
severity intervals with coverage valid *within each operating condition*. Unexplored for ITSC severity (most output
point estimates). Fixes our Phase-4 conformal under-coverage. **Deliverable:** `mondrian_cqr.py`.

---

## 6. Experimental data (download & integrate — addresses the "simulation-only" reviewer red flag)
| Dataset | Role | Specs | Access |
|---|---|---|---|
| **github.com/ibarram/ITSC** | **Primary experimental** | 0.75 hp IM, 3-phase current, **13 classes = healthy + ITSC 10/20/30/40% per phase**, 1 kHz, no-load; **per-phase + severity labels** | MIT, direct download |
| **Kaggle MIT (Cunha)** | Secondary (current+flux) | 3-phase current + flux, 7 severity (HI/LI impedance), ~14.7k samples | free Kaggle |
| **Our PSCAD set** | **Simulation source** for sim-to-real | severity×phase×load×inception, 25 kHz | local (122 GB) |
| Mendeley KAIST PMSM (opt.) | cross-machine transfer | 3φ current 100 kHz + vibration, 8 ITSC severities | CC-BY |
| IEEE-DataPort BRB (Treml, opt.) | DA transfer baseline | 3φ current+volt+vib, 50 kHz, multi-load | open |

**Note resolution mismatch:** ibarram is 1 kHz (no slip sidebands above 500 Hz) — handle by decimating sim to a
common rate for sim-to-real, and report it. Use `ilkersahin78` Simulink ITSC model only if more sim variety is needed.

## 7. Benchmark — **16 models** (≥10 required; mix ML + DL), identical leakage-safe splits
**Classical ML (6):** Logistic Regression · SVM-RBF · Random Forest · XGBoost · LightGBM · k-NN (+ shallow MLP).
**Deep learning (9):** 1D-CNN · multi-scale/wide-kernel 1D-CNN · TCN · BiLSTM · CNN-LSTM+attention · Transformer
encoder · Deep Residual Shrinkage Net (DRSN) · 1D-CNN-KAN · Mamba/SSM.
**Proposed (1): PCM-Net.**
Report per model: accuracy, macro-F1, **severity MAE/within-1/QWK**, **per-phase F1**, params, FLOPs, latency;
**mean ± std over folds + paired Wilcoxon significance** vs PCM-Net. (We already have XGBoost numbers as the anchor.)

## 8. Evaluation protocol (publication-grade rigor)
- **Leakage-safe** grouping by operating point (done) + **StratifiedGroupKFold**.
- **Cross-load Leave-One-Load-Out (PRIMARY)** + **Leave-One-Inception-Out** + **cross-dataset (sim→ibarram/Kaggle)**.
- **Voltage-unbalance robustness** — simulate VUF 1–3% cases (the classic ITSC confounder we currently never test)
  and show the detector rejects supply unbalance (δv disentangles it).
- **Noise** — SNR 40/30/20/10 dB + sensor offset/gain/quantization.
- **Calibration** — ECE/Brier/NLL (classification), coverage + width + per-load conditional coverage (severity).
- **Ablations** — window length {1,4,10,20 cyc}; feature families; SCST vs raw; LIR-mRMR vs mRMR; each PCM-Net block;
  augmentation matrix; transfer with/without physics anchor.
- **Interpretability** — SHAP (ML) + KAN spline visualization + Grad-CAM (SCST images) confirming physics, not shortcuts.

## 9. Automated pipeline, milestones, repo structure
All stages scripted, checkpointed to `PROJECT_LOG.md`, artifacts stored *with labels* (existing convention).
```
P1 data_index/segment/features  (DONE)            P6 pcdiff.py            (Axis 5 augmentation)
P2 feat_scst.py                 (Axis 1)          P7 pada.py + domain_datasets.py (Axis 4 transfer)
P3 select_lir.py / gate_invariance.py (Axis 2)    P8 benchmark_all.py     (16-model comparison)
P4 pcmnet.py (+ablate)          (Axis 3 model)    P9 mondrian_cqr.py      (calibrated severity)
P5 download_experimental.py     (data §6)         P10 paper_figures.py + report_v2.md
```
**Milestones (each = a checkpoint + a paper subsection):**
M1 experimental data ingested & unified loader; M2 SCST features; M3 LIR-mRMR + stability study;
M4 16-model benchmark (ML+DL) under LOLO; M5 PCM-Net + ablations; M6 PC-Diff augmentation study;
M7 PADA sim→real; M8 Mondrian-CQR calibration; M9 voltage-unbalance + noise robustness; M10 figures + manuscript draft.

**MVP path (fastest route to a defensible novel result, if time-boxed):** M1 → M2(SCST) → M3(LIR-mRMR) →
M4(benchmark) → M5(PCM-Net core, FiLM+Mamba+KAN, skip graph/PINN first) → M7(PADA sim→ibarram). Axes 1–4 alone are
already a strong Transactions paper; PC-Diff (Axis 5) and Mondrian-CQR are high-value extensions.

## 10. Paper skeleton (where each novelty lands)
I. Intro & gap (load-robustness + sim-to-real + no public-code, per the review). II. Related work. III. Problem &
data (sim + experimental). IV. **SCST representation**. V. **LIR-mRMR selection**. VI. **PCM-Net architecture**.
VII. **PC-Diff augmentation**. VIII. **PADA transfer**. IX. Experiments: 16-model benchmark, ablations, cross-load,
sim→real, unbalance/noise robustness, calibration, interpretability. X. Discussion & limitations. XI. Conclusion.
Public code + dataset splits released (directly answers the field's reproducibility gap).

## 11. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Mamba/KAN/diffusion = heavy on CPU | small configs, decimation, run long jobs in background; GPU optional |
| ibarram 1 kHz ≠ our 25 kHz | common-rate resampling; report; use it as the *hard* low-res regime |
| "too-good" sim results | lead with sim→real + LOLO + unbalance/noise; honest degradation curves |
| Novelty overlap (someone published it) | each axis chosen as a *confirmed gap*; ablate to prove contribution; re-survey before submission |
| Scope too large | MVP path (Axes 1–4) is a complete paper; 5 & calibration are extensions |

---
**Bottom line.** Five confirmed-novel components, an experimental sim-to-real validation, a 16-model benchmark, and
publication-grade rigor — turning an application into a method. Recommended first step: **M1 (ingest the experimental
ibarram/ITSC dataset + unified loader)**, then **M2 (SCST features)**.
