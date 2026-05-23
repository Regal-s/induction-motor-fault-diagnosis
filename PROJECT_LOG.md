# PROJECT LOG — Master Continuity / Handoff File

> **START HERE after any restart.** This file is the single source of truth for resuming work.
> It records the goal, dataset facts, everything done so far, every file produced, research
> findings, current status, and the next action. Read this top-to-bottom, then open the
> referenced docs. **This file is checkpointed periodically (≈ every 30 min of active work
> and at every milestone) — see the Changelog at the bottom for the latest state.**

- **Last updated:** 2026-05-22
- **Working directory:** `D:\Naveen`
- **OS / shell:** Windows 11, PowerShell (also Bash available)
- **Python:** 3.13.0 — installed: numpy 2.2.1, matplotlib 3.10.0, pandas 2.3.3
  (still needed for ML: scikit-learn, xgboost, scipy; DL optional: torch)
- **Persistent memory dir:** `C:\Users\shash\.claude\projects\D--Naveen\memory\` (MEMORY.md + notes)

---

## 1. PROJECT GOAL
Build an **ML diagnosis pipeline** for **three-phase induction-motor stator winding turn-to-turn
(inter-turn) faults**, as a **3-stage cascade**:
1. **Detection** — Healthy vs Faulty (binary)
2. **Severity** — if faulty: % shorted turns ∈ {0.3, 0.5, 1, 2, 3, 4, 5}
3. **Faulted phase** — if faulty: A / B / C

**Use ONLY** `TT fault Data` + `Healthy case data` (exclude `HRC fault data` and `LL`).
**Use three-phase current data.**

---

## 2. DATASET FACTS (full details in `DATASET_KNOWLEDGE.md`)
- Source: **PSCAD/EMTDC** simulation. Fundamental **50 Hz**, sampling **25 kHz** (40 µs step,
  500 samples/cycle). Fault runs = **10 s / 250,000 samples**; healthy runs = **5 s / 125,000 samples**.
- File trio per case: `.inf` (channel legend) + `.infx` (XML meta) + **3 × `.out`** (data).
  `.out` = whitespace columns, **first line blank**, col 1 = time. 10 traces per `.out`:
  `_01`=PGB1–10, `_02`=PGB11–20, `_03`=PGB21–24.
- **Three-phase current location:** `_02.out` cols 2,3,4 = `Ia_ABCIM_s` (normalised ±2);
  **`_03.out` cols 2,3,4 = `Ia_ABCIM` in kA** ← use this as primary.
- **Folder counts:** Healthy 21 cases (105 files) · TT 336 cases (1,680 files; 126 regular +
  210 "Extra" inception-time variants) · HRC 663 · LL 1.
- TT tree: `<severity%>\Phase_<a|b|c>_<sev>\<sev>_<load>\` ; loads = NL/20/40/60/80/100%.
  "Extra TT Fault Cases" add 10 fault-inception times (7.202–7.22 s) per (sev,phase,load).

### Event timeline (verified from current RMS envelope) — KEY
| Event | Sample | Time |
|---|---:|---:|
| Motor energised (start, big inrush) | 50,000 | 2.0 s |
| Inrush settled | ~67,000 | ~2.7 s |
| **Load applied / changed** | 100,000 | 4.0 s (NONE for NL cases) |
| **Fault inception** | **180,000** | **7.2 s** (fixed in ALL fault cases) |
| Fault cleared | ~200,000 | ~8.0 s (≈0.8 s duration) |
| End (fault files) | 250,000 | 10.0 s |
| End (healthy files) | 125,000 | 5.0 s |
> HRC files use a DIFFERENT schedule (load 3 s, fault 4 s) — don't reuse TT timing for HRC.

---

## 3. WORK DONE SO FAR (chronological)
1. Explored full folder tree; characterised all 4 datasets and file format.
2. Confirmed dataset semantics with user: induction-motor stator inter-turn fault; 50 Hz; 25 kHz; 10 s.
3. Wrote **`DATASET_KNOWLEDGE.md`** — complete dataset reference (verified facts).
4. Plotted 3-phase current (cols 2,3,4 of `_02.out`) **healthy vs faulty** → `currents_healthy_vs_faulty.png`.
   Demo case chosen by user: **TT 5% severity, phase A, 20% load** vs **Healthy 5% load**.
5. Analysed transition points via per-cycle RMS envelope → produced the event timeline above and
   annotated figure `events_annotated.png`. Confirmed user's statement that fault = sample 180,000.
6. Wrote first ML plan **`ML_PLAN.md`**.
7. Ran **3 parallel web-research agents** (≈50 papers) on: (a) ML/DL models + severity, (b) signal
   processing / feature engineering, (c) DL architectures + methodology + phase ID.
8. Synthesised into research-grounded **`ML_PLAN_DETAILED.md`** (with citations + bibliography).
9. Wrote this **`PROJECT_LOG.md`** continuity file and set up periodic checkpointing.

---

## 4. FILES PRODUCED (in `D:\Naveen`)
| File | Purpose |
|---|---|
| `DATASET_KNOWLEDGE.md` | Full dataset reference: structure, channel map, timeline, caveats |
| `ML_PLAN.md` | First-pass ML plan |
| `ML_PLAN_DETAILED.md` | **Research-grounded detailed plan** (primary plan to follow) |
| `PROJECT_LOG.md` | **This file** — master continuity/handoff |
| `plot_currents.py` | Plot `_02` cols 2,3,4 (3-phase current); healthy-vs-faulty or single case |
| `plot_events.py` | Annotated phase-A current + RMS envelope marking start/load/fault/clear |
| `analyze_events.py` | Detect motor-start / load-change / fault from RMS envelope |
| `data_index.py` | **M1** — builds the case manifest (walk Healthy+TT, parse labels, regions) |
| `manifest.csv` | **M1 output** — 357 cases × labels/regions/paths (input to M2) |
| `segment.py` | **M2** — windows the regions, normalises, stores labeled windows |
| `dataset/windows.npy` | **M2 output** — (18033,3,2000) float32 current windows |
| `dataset/window_labels.csv` | **M2 output** — 18,033 labeled window rows (aligned to windows.npy) |
| `features.py` | **M3** — Track-A feature extraction (vectorized + per-case NSC compensation) |
| `dataset/features.csv` | **M3 output** — 18,033 × (~59 features + all labels) |
| `splits.py` | **M4** — leakage-safe splitters (StratifiedGroupKFold + Leave-One-Load-Out) |
| `train_stage1.py` | **M4** — Stage-1 Healthy/Faulty XGBoost + SHAP audit |
| `dataset/stage1_*.{csv,json,png}` | **M4 output** — OOF preds (labeled), metrics, confusion fig |
| `train_stage2.py` | **Phase 2** — severity ordinal regression (rank) |
| `train_stage3.py` | **Phase 2** — faulted-phase A/B/C classifier |
| `dataset/stage2_*.{csv,json,png}` | Stage-2 OOF preds (labeled), metrics, confusion |
| `dataset/stage3_*.{csv,json,png}` | Stage-3 OOF preds (labeled), metrics, confusion |
| `feature_sets.py` | shared feature-group defs (all vs load-invariant severity set) |
| `stage2_experiment.py` | Stage-2 feature/granularity comparison (fine/coarse × all/invariant/I1norm) |
| `cascade.py` | end-to-end Stage1→Stage2+Stage3 evaluation (GroupKFold + LOLO) |
| `dataset/cascade_*.{csv,json}` | cascade OOF preds (labeled) + end-to-end metrics |
| `report.md` | **consolidated results report** (all stages + cascade, thesis/paper-ready) |
| `noise_robustness.py` | **Phase 3a** — SNR 40/30/20 dB robustness (train clean/test noisy) |
| `calibration_uncertainty.py` | **Phase 3a** — Stage-1 ECE/Brier + Stage-2 quantile uncertainty |
| `dataset/noise_robustness.*`, `calibration_uncertainty.json`, `*reliability.png`, `uncertainty_width.png` | Phase 3a outputs |
| `dl_resnet1d.py` | **Phase 3b** — ResNet-1D Track B (torch) |
| `severity_deepdive.py` | **Phase 3c** — feature stability + load-aware severity |
| `dataset/dl_metrics.json`, `severity_deepdive.json`, `severity_feature_stability.png` | Phase 3b/3c outputs |
| `conformal_uncertainty.py` | **Phase 4** — CQR conformal severity intervals |
| `stage0_onset.py` | **Phase 4** — fault-onset detection (per-cycle |I2|/|I1|) |
| `dataset/conformal_uncertainty.json`, `stage0_onset.{json,csv}` | Phase 4 outputs |
| `dataset/features.parquet` | **M3 output (now 72 feats + labels; csv removed, pyarrow installed)** |
| `currents_healthy_vs_faulty.png` | Output plot |
| `events_annotated.png` | Output plot |

Memory notes: `MEMORY.md`, `dataset-overview.md`, `ml-project.md` in the memory dir.

---

## 5. RESEARCH FINDINGS — key decisions (full detail + refs in `ML_PLAN_DETAILED.md`)
1. **Anti-leakage is #1 priority:** split by **CASE, never by window** (StratifiedGroupKFold);
   keep all 10 inception variants of a case in one fold; **Leave-One-Load-Out (LOLO)** is the
   headline generalisation metric. Treat ~100% accuracy as a leakage alarm.
2. **Healthy data from the TT pre-fault region (~110k–178k)** of every fault file → matched-load
   healthy samples (removes load confound) AND a per-case baseline for NSC compensation.
3. **Top features (ranked):**
   - **EPVA severity factor** = amplitude(2f₁=100 Hz of Park modulus |i_P|) / DC(|i_P|) — best severity scalar.
   - **Compensated negative-sequence current**: `|I₂|/|I₁|` (severity), **`∠I₂` (faulted phase, 3 sectors ~120° apart)**.
   - MCSA Penman lines `f₁[k±n(1−s)/p]`, 3rd-harmonic ratio; PVA ellipse (eccentricity/tilt);
     per-phase stat **differences** (RMS/THD/kurtosis/skew).
4. **Severity = ordinal regression** (predict continuous %, then bin). Metrics: **MAE in levels**,
   quadratic-weighted kappa, ±1-level accuracy. (Adjacent-level confusion is the dominant error.)
5. **Normalisation:** one **shared scale across the 3 phases, per recording** (e.g. ÷ positive-seq
   fundamental amplitude). NEVER per-channel/per-window — that erases the imbalance signature.
6. **Models:** Track A = **XGBoost/RF on engineered features** (baseline, interpretable, leakage-resistant);
   Track B = **ResNet-1D** (large first-layer kernel ≈ 1 cycle) + optional TCN/attention head.
   Cascade primary; **multi-task faulty-branch** (shared encoder, severity+phase heads) as enhancement.
7. **Decimate to ~2–5 kHz** for the feature/spectral track (ITSC content < ~2–3 kHz); raw-CNN keeps full rate.
8. **Windowing:** primary 10 cycles (5,000 samples, 200 ms), 50% overlap; sweep {1,4,10,20} cycles.
9. **Augment** (training folds only): jitter/scale/shift with all 3 phases scaled jointly; optional 1D-GAN.
10. **Evaluate:** macro-F1 + per-load confusion (Stage1); MAE/kappa (Stage2); per-phase F1 (Stage3);
    end-to-end exact-match; report mean±std over GroupKFold AND LOLO; SHAP as leakage guard.

---

## 5b. EXTENDED ENHANCEMENTS (folded in from `Updated_Fault_Detection_Readme (1).md` → now §14 of `ML_PLAN_DETAILED.md`)
- **LOLO (Leave-One-Load-Out) is now the PRIMARY benchmark** (not secondary) — the #1 risk is the model learning *load* instead of *fault physics*.
- **Stage 0 — fault-onset detection** (CUSUM on |I₂| / change-point / sliding-EPVA / AE recon error) — for deployment realism (Phase 4).
- **Uncertainty estimation** (probabilistic XGBoost / quantile reg; MC-dropout / evidential DL) — esp. for 0.3/0.5/1% overlap.
- **Temporal consistency** at case level (HMM/CRF/smoothing) instead of plain majority vote.
- **Fine vs coarse severity** comparison: fine {0.3..5} vs coarse {Incipient/Moderate/Severe}.
- **Leave-One-Inception-Out (LOIO)** benchmark (robustness to fault initiation angle).
- **Noise robustness:** inject SNR 40/30/20 dB (+offset/gain/jitter); compare EPVA/NSC/ML/CNN.
- **DL regularisation** (dropout, early stop, weight decay, label smoothing, MixUp/CutMix-1D).
- **Calibration metrics** (ECE, reliability diagram, Brier) — esp. Stage 1.
- **Physics-threshold baseline** (`|I₂|/|I₁| > thr`) to justify ML.
- **Feature-stability analysis across loads** (intra-class var, Fisher score, ANOVA) — prove EPVA/NSC robustness.
- **DL explainability** (Grad-CAM, saliency, integrated gradients) alongside SHAP.
- **Revised modular prioritisation (Phases 1–4)** — see roadmap §7.

## 6. CURRENT STATUS
- **CONVENTION (user instruction):** every stored artifact — windows AND features — must be saved
  to disk *with its full label set* (state, severity, phase, load, inception, variant) + case_id + group_id,
  so each table is self-contained and traceable. Applied in M2; carry through M3+.
- **Phase:** Phase 1 + 2 DONE. **Phase 3 DONE** (3a robustness, 3b ResNet-1D, 3c severity fix) — autonomous run.
- **Phase 3b — ResNet-1D (`dl_resnet1d.py`, torch 2.12 cpu):** decimated 8×, single grouped split. Track B vs Track A:
  detect F1 0.898 vs 0.977; detect LOLO-20 0.401 vs 0.749; phase 0.848 vs 0.972; severity within-1 1.000 vs 0.990.
  → **physics+XGBoost beats raw-signal CNN** on detect/phase (data-efficiency); comparable on severity. `dl_metrics.json`.
- **Phase 3c — severity deep-dive (`severity_deepdive.py`):** (1) feature-stability: ratios/`n_*` are load-stable
  (drift~0.09) & severity-discriminative — validates the invariant set. (2) **LOAD-AWARE severity** (add measured
  `load_pct`): LOLO within-1 **0.729→0.946**, GK 0.990→0.997. Made `feature_sets.severity_features` load-aware by default.
  Re-ran Stage-2 + cascade: **cascade LOLO end-to-end within-1 0.774→0.916, per-case 0.708→0.918; GK within-1 0.949 (exact 0.911).**
  → cross-load severity bottleneck largely RESOLVED. `report.md` updated with full Phase 3 section (§13).
- **Phase 3a — Robustness & trust DONE** (`noise_robustness.py`, `calibration_uncertainty.py`):
  - Refactored `features.py` to expose `build_feature_frame(X, lab)` (reused for noisy-signal features).
  - **Noise (train clean / test noisy, GroupKFold):** detect F1 0.977→0.927(40dB)→0.854(30)→0.792(20);
    severity within-1 0.986→0.971; phase acc 0.958→0.939. → severity & phase noise-robust; **detection is noise-sensitive.**
  - **Calibration (Stage-1):** ECE=0.021, Brier=0.021 → well-calibrated.
  - **Uncertainty (Stage-2 quantile reg):** 80% interval coverage only 0.63 (under-covered), width-vs-error
    uninformative → naive quantile reg poorly calibrated; **recommend conformal prediction** (noted as future work).
  - Outputs: `dataset/noise_robustness.{json,png}`, `dataset/calibration_uncertainty.json`,
    `calibration_reliability.png`, `uncertainty_width.png`.
- **Phase 4 (partial) DONE:** **Conformal uncertainty** (`conformal_uncertainty.py`) — CQR coverage 0.49→0.72
  (group folds break exchangeability; group-conditional conformal would close it). **Stage-0 onset detection**
  (`stage0_onset.py`) — per-cycle |I2|/|I1| threshold: 100% detection, 0% pre-fault false alarm, 0 ms median delay
  (≥0.5% sev instant; 0.3% ~1.42s). report.md §14 added.
- **STOPPING POINT (autonomous run end):** Phases 1–3 complete + Phase 4 conformal & Stage-0. Remaining Phase 4
  (open-ended, not done): domain adaptation, multi-task learning, transformer heads, GAN augmentation, hardware validation.
- **Stage-2 fix DONE:** root cause = per-recording norm scale behaves differently at no-load → magnitude feats
  don't transfer. Fix = **load-invariant feature set** (dimensionless ratios/shapes/angles + `n_*` = magnitudes ÷ within-window |I1|).
  Added `n_*` features to `features.py` (features now 72; `features.parquet`, csv removed). Stage-2 uses `feature_sets.severity_features`.
  Result: GK within-1 0.990 (≈ all-feats 0.993), **NL within-1 0.39→0.80**, QWK 0.62→0.74. (`stage2_experiment.py` has the comparison.)
- **CASCADE DONE (`cascade.py`):** Stage1(all feats)→{Stage2(invariant), Stage3(all)}; Stages 2/3 scored on Stage-1's pred-faulty.
  - GroupKFold: detect F1 0.977 | cond sev within-1 0.986, phase 0.957 | **end-to-end window within-1 0.942 (exact 0.775)** | per-case within-1 0.916.
  - LOLO (PRIMARY): detect F1 0.948 | cond sev within-1 0.721, **phase 0.997** | **end-to-end window within-1 0.774 (exact 0.414)** | per-case within-1 0.708.
  - KEY FINDING: detection + phase generalise across loads; **absolute severity is the cross-load bottleneck.**
  - LOLO eval bug fixed: evaluate only over covered windows (healthy files at non-TT loads aren't held out).
- **Phase 2 (`train_stage2.py`, `train_stage3.py`) DONE** — faulty windows only (label==1).
  - **Stage-2 severity (ordinal regression on rank):** GroupKFold MAE=0.127 levels, QWK=0.975, exact=0.889,
    within-1=0.993. **LOLO (PRIMARY): MAE=1.076, within-1=0.887, QWK=0.622 — NL hold-out collapses (within-1 0.39).**
    SHAP: top feats are MAGNITUDE features (I1_abs, pva_minor, park_dc) → load-dependent. **Improvement: use
    load-invariant ratio feats (I2/I1, epva_sf) and/or load-aware modeling for cross-load severity** (matches §14.10).
  - **Stage-3 phase A/B/C:** GroupKFold acc/F1=0.972 (minor A↔B confusion, C perfect); **LOLO acc/F1=0.997**
    (load-robust — NSC angle signature). Per-case majority vote 0.973. Essentially solved.
  - Outputs: `dataset/stage2_*.{csv,json,png}`, `dataset/stage3_*.{csv,json,png}` (OOF preds labeled).
- **Deps installed:** scikit-learn 1.8.0, xgboost 3.2.0, scipy 1.17.1, pyarrow 24.0.0, shap 0.51.0.
- **M4 (`splits.py` + `train_stage1.py`) DONE** — Stage-1 Healthy-vs-Faulty XGBoost.
  - StratifiedGroupKFold (group=group_id): **macro-F1 0.977, ROC-AUC 0.994** (honest: fold0=0.898).
  - **Leave-One-Load-Out (PRIMARY): pooled macro-F1 0.948, mean-over-loads 0.939.** Loads 40/60/80/100 = 1.0;
    **weak at unseen NL (0.885) and load-20 (0.749)** = realistic incipient-at-light-load difficulty.
  - SHAP leakage audit PASSES — top features physical (pva_major, pva_ecc, I0_I1_ratio, I2_I1_ratio, epva_sf), no load proxy.
  - Outputs: `dataset/stage1_oof_predictions.csv` (labeled), `stage1_metrics.json`, `stage1_confusion.png`.
  - Improvement ideas for later: address load-20/NL weak spot (more low-severity windows, threshold tuning, the
    normalization revisit). Detection relies on physical features, so the normalization caveat is not dominating.
- **M3 (`features.py`) DONE** → `dataset/features.csv` (18,033 rows, ~59 features + all label cols).
  Validated physics: detection separation strong (I2_I1_ratio healthy 0.0004 vs faulty 0.193, pva_ecc 0.039 vs 0.675,
  and detection works on RAW |I2|/|I1| without the compensation trick); severity monotonic in |I2c|/|I1|
  (0.063→0.322 over 0.3→5%); faulted-phase NSC angle clusters ~120° apart (A -65°, B 58°, C 175°, via circular mean).
  Features incl: epva_sf, sequence comps |I1|/|I2|/|I0| + ratios + angles(+sin/cos), compensated I2c (abs/ratio/angle),
  PVA ellipse, 3rd-harm ratio, THD, per-phase rms/peak/crest/std/skew/kurt + inter-phase diffs, argmax_rms_phase.
  NOTE: pyarrow not installed → saved CSV (not parquet); `pip install pyarrow` for typed/faster load if wanted.
- **M2 (`segment.py`) DONE** → `dataset/windows.npy` (18,033 × 3 × 2000, 433 MB) + `dataset/window_labels.csv`.
  Window=2000 (4 cyc); step faulty=500 (75% ovl), healthy=4000. Per-recording norm = RMS over case's
  healthy region (preserves imbalance, removes load amplitude). 11,760 faulty (balanced 1680/sev, 3920/phase)
  + 6,273 healthy (6,138 matched-load pre-fault + 135 steady).
  **CAVEAT:** per-recording scale uses the case's own healthy-region RMS even for faulty files (a "compensation"
  assumption). Fine for this dataset + the plan's philosophy, but makes Stage-1 detection easier than a
  baseline-free deployment. Robust path = M3 scale-free ratio features (|I2|/|I1|, EPVA-SF) + LOLO. Revisit
  normalization (e.g. within-window positive-seq, or per-load reference) when building Stage 1.
- **M1 (`data_index.py`) DONE** → `manifest.csv` (357 cases) built & validated.
- **Planning** COMPLETE (+ extended review folded in).
- **M1 result:** 336 faulty + 21 healthy; severity balanced (48×7); phase balanced (112×3);
  147 CV groups (inception variants bound into size-11 groups). All file paths + region bounds validated PASS.
  Manifest columns: case_id, group_id, state, severity_pct, phase, load_label/pct, inception_s, variant,
  fs, n_rows, path_03/02/inf/infx, motor_start, load_change, fault_start/end, hreg_start/end, freg_start/end.
- **Gotcha found & handled:** `.inf/.infx` use a different base name than `.out` (e.g. `TT25kphasea_0.3%_load100.infx`
  vs `...loading100%_03.out`) → resolve by globbing the leaf dir, not name-derivation.
- **Decision/defaults locked (revisable):** current source `_03.out` kA · window 10 cycles/50% ·
  shared-scale per-recording norm · XGBoost before DL · severity ordinal regression · ResNet-1D backbone ·
  cascade primary · headline metric LOLO.
- **No code for the ML pipeline written yet** (only the plotting/analysis scripts above).

---

## 7. NEXT ACTION (resume here) — revised modular roadmap
**Phase 1 (highest priority, do first):**
- ✅ **M1 `data_index.py`** DONE → `manifest.csv` (357 cases, validated).
- ✅ **M2 `segment.py`** DONE → `dataset/windows.npy` + `window_labels.csv` (18,033 labeled windows).
- ✅ **M3 `features.py`** DONE → `dataset/features.csv` (18,033 × ~59 features + labels).
- ✅ **M4 `splits.py` + `train_stage1.py`** DONE — Stage-1 GroupKFold macro-F1 0.977 / LOLO 0.948 (leakage audit passed).
- ✅ **`train_stage2.py`** + **`train_stage3.py`** + **Stage-2 load-invariant fix** + **`cascade.py`** ALL DONE.
**Phase 2 COMPLETE.** Phase 1 + 2 of the project goal (detect→severity→phase cascade) delivered & evaluated.
**NEXT — pick one:**
- ✅ **`report.md`** DONE — consolidated results report written.
- **Phase 3:** ResNet-1D (Track B) to compare vs XGBoost; noise robustness (SNR 40/30/20); uncertainty (quantile/MC-dropout);
  calibration (ECE/Brier); feature-stability ANOVA (§14.10). See ML_PLAN_DETAILED §14.12.
- **Severity deep-dive:** try within-window |I1| RE-NORMALISATION at segmentation (vs the n_* post-hoc divide) to
  recover cross-load severity further; or load-aware severity (load is measurable in deployment).
  · M4 `splits.py` (StratifiedGroupKFold + **LOLO primary** + LOIO) · Stage-1 XGBoost baseline + LOLO eval.
**Phase 2:** severity ordinal regression · faulted-phase ID · SHAP · fine-vs-coarse severity.
**Phase 3:** ResNet-1D · noise robustness (SNR 40/30/20) · uncertainty estimation · calibration.
**Phase 4 (optional, publication-grade):** Stage-0 onset · domain adaptation · MTL · transformer heads · GAN aug.
(Full detail: §11 + §14 of `ML_PLAN_DETAILED.md`.)

Open question to confirm before Phase 3: classical-first (recommended) is locked for Phase 1–2; confirm whether to proceed to ResNet-1D after.

---

## 8. HOW TO RESUME (quick start for a new session)
1. Read this file, then `ML_PLAN_DETAILED.md` and `DATASET_KNOWLEDGE.md`.
2. `cd D:\Naveen`; verify Python libs (§ top); `pip install scikit-learn xgboost scipy` if missing.
3. Continue from **§7 NEXT ACTION**.
4. Quick data sanity: `python plot_currents.py "<path to a _02.out>"`.

---

## NOVEL-METHOD IMPLEMENTATION (from NOVEL_METHOD_PLAN.md)
- **M1 DONE — experimental data ingested + unified pipeline.** Cloned `github.com/ibarram/ITSC`
  (gitignored `experimental/`). Built `experimental/load_ibarram.py` → unified schema windows
  (1235: 1140 faulty/95 healthy; balanced phase 380 ea, severity 285 ea; fs=1000, f0=60, no-load,
  W=500=30cyc). **Parameterized `features.py` (chunk_features/build_feature_frame take fs,f0)** so one
  extractor serves sim (25kHz/50Hz) and experimental (1kHz/60Hz); sim behavior unchanged.
  `experimental/features_ibarram.py` → features.parquet. **Physics VALIDATED on real data:** |I2|/|I1|
  monotonic with severity (0.061→0.149 over 10-40%), healthy/faulty separates (I2/I1 0.029→0.109);
  phase-angle clustering noisier than sim (A94/B175/C147°) → real phase-ID harder. Note: severity scales
  differ (exp 10-40% vs sim 0.3-5%) so only detection/phase transfer directly; no load variation in exp.
- **Sim→real naive-transfer baseline DONE** (`experimental/sim2real_baseline.py`): train on full sim,
  test on real ibarram, NO adaptation. **Detection macro-F1 0.48** (degenerate: predicts all faulty, because
  real healthy I2/I1=0.029 >> sim healthy 0.0004 → covariate shift); **phase macro-F1 0.32** (collapses to
  phase A; 50→60 Hz angle frame differs). → strong motivation for PADA. Physics features transfer (monotonic
  trends) but naive model transfer fails — the gap to close.
- **NEXT:** M2 (SCST sequence-component Stockwell tensor representation); then PADA (domain adaptation:
  per-domain standardization / DANN / MMD with physics anchors) to recover sim→real performance.

## CHANGELOG / CHECKPOINTS
- **2026-05-22 — Checkpoint 1:** Created this log. State = planning complete, implementation not started.
  Dataset fully characterised; timeline verified; research done; `ML_PLAN_DETAILED.md` written.
  Next = M1 `data_index.py`.
- **2026-05-22 — Checkpoint 2:** Folded reviewer file `Updated_Fault_Detection_Readme (1).md` into
  `ML_PLAN_DETAILED.md` as §14 (14 enhancements). Key change: **LOLO is now the PRIMARY benchmark**.
  Added §5b (extended enhancements) and revised §7 roadmap into Phases 1–4. Still pre-implementation; next = M1.
- **2026-05-22 — Checkpoint 3:** **M1 complete.** `data_index.py` → `manifest.csv` (357 cases),
  fully validated (all paths exist, regions sane, balanced labels, 147 CV groups). Handled `.inf/.infx`
  name-mismatch via globbing. Next = **M2 `segment.py`** (windowing + normalisation).
- **2026-05-22 — Checkpoint 4:** **M2 complete.** `segment.py` → `dataset/windows.npy` (18,033×3×2000)
  + `window_labels.csv`, all windows labeled (per user instruction). Balanced faulty classes. Logged a
  normalization caveat to revisit at Stage 1. Next = **M3 `features.py`** (labeled feature table).
- **2026-05-22 — Checkpoint 5:** **M3 complete.** `features.py` → `dataset/features.csv` (18,033 × ~59
  features + all labels). Physics validated: strong detection separation, monotonic severity in |I2c|/|I1|,
  phase NSC angle ~120° apart. Added sin/cos angle features for wrap-free phase ID. pyarrow missing → CSV.
  Next = **M4 splits + Stage-1 XGBoost baseline** under StratifiedGroupKFold + LOLO.
- **2026-05-22 — Checkpoint 6:** **M4 complete + deps installed** (sklearn/xgboost/scipy/pyarrow/shap).
  `splits.py` + `train_stage1.py`. Stage-1 GroupKFold macro-F1 0.977 (ROC-AUC 0.994); LOLO PRIMARY 0.948 pooled
  (weak at NL/load-20). SHAP audit passed (physical features). **Phase 1 COMPLETE.** Next = Phase 2:
  `train_stage2.py` (severity ordinal) + `train_stage3.py` (phase).
- **2026-05-22 — Checkpoint 7:** **Stages 2 & 3 baselined.** Stage-2 severity: GroupKFold within-1=0.993/QWK=0.975,
  LOLO within-1=0.887 (NL collapses; load-dependent magnitude feats → improve w/ ratio feats). Stage-3 phase:
  GroupKFold F1=0.972, **LOLO F1=0.997 (load-robust, ~solved)**. All OOF preds saved labeled. Next = Stage-2 fix + cascade.
- **2026-05-22 — Checkpoint 8:** **Phase 2 fully COMPLETE.** Stage-2 fix (load-invariant + n_* I1-normalised feats):
  NL within-1 0.39→0.80, GK within-1 still 0.990. Added n_* to features.py (72 feats, parquet; csv removed). Built
  `cascade.py` + `feature_sets.py` + `stage2_experiment.py`. **End-to-end: GK within-1 0.942 / LOLO within-1 0.774**
  (per-case 0.916 / 0.708). Finding: detect+phase load-robust, severity is cross-load bottleneck. Next = report.md or Phase 3.
- **2026-05-22 — Checkpoint 9:** **`report.md` written** — consolidated thesis/paper-ready results report (dataset,
  pipeline, per-stage tables + confusions, Stage-2 fix experiment, end-to-end cascade, findings, limitations, repro).
  Phases 1 & 2 fully delivered & documented. Next (optional) = Phase 3 (ResNet-1D / noise / uncertainty / calibration).
- **2026-05-22 — Checkpoint 10 (AUTONOMOUS, user asleep w/ full permission):** **Phase 3a robustness & trust DONE.**
  Refactored features.py (build_feature_frame). Noise: detect noise-sensitive (F1 0.79@20dB), severity/phase robust.
  Calibration Stage-1 ECE 0.021. Uncertainty: quantile intervals under-calibrated (cov 0.63) → recommend conformal.
  Next = Phase 3b ResNet-1D (torch install + train).
- **2026-05-22 — Checkpoint 11 (AUTONOMOUS):** **Phase 3 COMPLETE.** 3b ResNet-1D (torch cpu): physics+XGBoost beats
  CNN on detect/phase. 3c severity deep-dive → **load-aware severity fix**: cascade LOLO end-to-end within-1 0.77→0.92,
  per-case 0.71→0.92; GK within-1 0.949/exact 0.911. feature_sets severity now load-aware. report.md §13 added.
  Continuing autonomously to optional Phase 4 (conformal uncertainty + Stage-0 onset).
- **2026-05-22 — Checkpoint 12 (AUTONOMOUS, run end):** **Phase 4 conformal + Stage-0 DONE.** CQR coverage 0.49→0.72.
  Stage-0 onset: 100% detection / 0% false alarm / 0 ms median delay (0.3% sev ~1.42s). report.md §14 added.
  **AUTONOMOUS RUN COMPLETE — clean stopping point. Phases 1–3 + Phase 4 (conformal, Stage-0) all delivered & documented.**
  Open (not done): domain adaptation, MTL, transformer heads, GAN aug, hardware validation.
- **2026-05-23 — Checkpoint 13:** **M1 (experimental ingestion) COMPLETE.** Cloned ibarram/ITSC,
  built unified loader (1235 real windows) + parameterized features.py (fs/f0-aware). Physics validated
  on real data (|I2|/|I1| monotonic 0.061->0.149 over 10-40%). Naive sim->real transfer baseline:
  detection F1 0.48 / phase F1 0.32 (large gap -> motivates PADA). Next = M2 SCST representation.
- **2026-05-23 — Checkpoint 14:** **PADA (sim->real domain adaptation) DONE** (`experimental/pada.py`).
  Ladder vs naive (macro-F1 sim->real): detection 0.48->0.59 (unsup quantile) ->0.98 (few-shot +2 reps);
  phase 0.32->0.50 (unsup) ->0.67 (few-shot). Physics-anchored subset (42 domain-stable feats) matches
  full-feature quantile alignment (parsimony win). Few-shot calibration with a handful of real reps is the
  decisive lever; phase the hard case unsupervised (50->60Hz angle frame). Next = M2 (SCST representation).
- **2026-05-23 — Checkpoint 15:** **M2 (SCST representation) DONE** (`scst.py`, `scst_eval.py`, `scst_lolo.py`).
  Instantaneous sequence components (Hilbert+Fortescue) + band-limited Stockwell transform -> 3-channel
  tensor (pos/neg/zero). Generated 2151 sim windows. Findings: in-distribution SCST ties raw-phase on
  detection (0.972) & severity (within-1 0.917) but FAILS phase (0.20 vs 1.0 raw) — sequence MAGNITUDE is
  phase-agnostic by design (angle, discarded by |S|, carries phase). **Cross-load (held-out load): SCST
  severity within-1 1.0 vs raw 0.75** — the load-invariant negative-seq channel generalizes severity to
  unseen loads. Design principle: SCST for load-robust detect/severity + neg-seq ANGLE for phase. Next = PCM-Net (M5).
- **2026-05-23 — Checkpoint 16:** **M3 (LIR-mRMR feature selection) DONE** (`select_lir.py`).
  Criterion J(f)=Rel - lambda*Redundancy - gamma*LoadInstability (gamma=0 -> plain mRMR). On load-blind
  severity, LOLO within-1: k=5 mRMR 0.741 vs LIR 0.834; k=10 mRMR 0.740 vs LIR 0.848 (near all-72=0.887 with
  10 feats); selected-feature mean load-drift ~halved (0.18 vs 0.38). LIR picks dimensionless stable feats
  (pva_ecc, neg-seq angle sin/cos, n_rms_imbalance, skew); mRMR pulls load-drifting magnitudes (I1_abs,
  I1_angle_deg). Validates building load-invariance into the selection criterion. (k=15 dip = greedy noise.)
  Plan progress: M1+PADA+M2+M3 done. Remaining: PCM-Net (M5, heaviest) + PC-Diff (augmentation).
- **2026-05-23 — Checkpoint 17:** **M5 (PCM-Net, compact) DONE** (`dl_pcmnet.py`). Phase-coupled, FiLM-load-
  conditioned multi-task net (detect/phase/severity) + current-balance physics regularizer; Mamba/KAN approximated
  by dilated-temporal backbone (CPU). FiLM ablation: **LOLO severity within-1 0.932 (no FiLM) -> 0.982 (FiLM)** —
  best cross-load severity in project (> XGBoost load-aware 0.946). In-distribution: detect F1 0.898, severity
  within-1 1.0; FiLM HURTS in-dist phase (0.85->0.70) — sensible, phase is load-invariant so load-conditioning is
  irrelevant there -> FiLM should gate severity not phase. Detection/phase still stronger with feature XGBoost
  (data efficiency). **Validates core model novelty: inject measured load via FiLM.** Remaining: PC-Diff (augmentation).
- **2026-05-23 — Checkpoint 18:** **PC-Diff (physics-constrained augmentation) DONE — MIXED/honest result**
  (`pcdiff.py`). Conditional VAE, vanilla vs PC (hard Kirchhoff current-balance projection + soft neg-seq-vs-
  severity loss). PHYSICS ENFORCEMENT WORKS: balance residual 0.000 (PC) vs 0.991 (vanilla). BUT compact MLP-CVAE
  failed to learn severity-controllable generation (corr(gen |I2|/|I1|, severity) ~0 for both -> posterior
  collapse / weak conditioning on CPU). Downstream scarce-severity augmentation did NOT help: no-aug within-1
  0.985, +PC 0.845 (hurt, mislabeled samples), +vanilla 0.982 (neutral); scarce baseline already saturated.
  HONEST CONCLUSION: physics-constraint mechanism validated, but augmentation value needs a stronger conditional
  generator (diffusion/cGAN) + genuinely data-starved regime -> the one under-delivering axis (future work).
  **All 5 novel axes attempted: SCST(M2) +, LIR-mRMR(M3) +, PCM-Net/FiLM(M5) +, PADA +, PC-Diff ~(physics ok, aug no).**
- **2026-05-23 — Checkpoint 19:** **Manuscript revised to novel-method paper** (`manuscript/main.tex`, 9 pages).
  Folded all 5 novelty axes into main experiments: new Section "Load-Robustness Mechanisms" (SCST, LIR-mRMR,
  PCM-Net/FiLM, PADA, PC-Diff honest) with real LOLO results + summary table; added experimental ibarram dataset
  to Section III; updated abstract + 5 contributions; trimmed Future Work to genuine remaining (stronger
  generator, full Mamba/KAN, voltage unbalance, Mondrian conformal, full benchmark). Compiles clean, citations OK.
- **2026-05-23 — Checkpoint 20:** **Manuscript Word export + SVG result figures.** `manuscript.docx` (pandoc).
  10 editable SVGs in `figures_svg/` (svg.fonttype=none): confusion (detect/severity/phase), ROC (overall+per-load,
  AUC 0.994), per-stage precision/recall/F1/accuracy, baseline model comparison (5-fold: feature-ML ~0.98 vs
  deep ResNet/PCM-Net 0.898), XGBoost convergence, 1D-CNN learning curves, inference-time/efficiency (feat+XGB
  0.34 ms/window), noise robustness. `make_paper_figures.py` generates all from real data/OOF.
- **2026-05-23 — Checkpoint 21:** **Format match to IEEE Trans (double-column, Times) + IEEE-sized SVGs.**
  Reference Final_Submission.pdf = IEEE Trans Magnetics (Word, double-column) -> format only. Manuscript already
  IEEEtran [journal] (double-column); added newtxtext/newtxmath so it renders in Times (8 pages). Two-column Word
  version `manuscript_2col.docx` (pandoc + python-docx 2-col section). Regenerated 10 result SVGs at IEEE column
  widths (single 3.45in / double 7.16in, 8pt fonts) in `figures_svg/`. Final_Submission.pdf gitignored.
- **2026-05-23 — Checkpoint 22:** **Manuscript restructured to reference (Final_Submission.pdf) outline.**
  Used the reference (IEEE Trans Magnetics) for FORMAT/structure only. New section/subtopic outline (verified to
  match): I Introduction (A Motivation, B Literature Review, C Key Contributions); II Proposed Methodology
  (A Overview, B Preprocessing & Symmetrical Components, C Stockwell/SCST, D Time-Domain & Spectral, E Statistical
  Features, F Data Augmentation & Feature Selection, G XGBoost Model, H Transfer Learning); III Experimental Setup
  & Dataset Prep (A Setup, B Datasets); IV Performance Evaluation (A Implementation & Training, B Results &
  Discussion, C Comparative Evaluation); V Conclusion. Double-column IEEE Times, 6 pages, compiles clean. Two-column
  Word `manuscript_ieee.docx` regenerated. All my technical content/results retained (reference content NOT used).
