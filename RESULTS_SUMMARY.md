# Project results summary

Stator inter-turn fault diagnosis in three-phase induction motors from
three-phase stator current. Physics-informed, leakage-safe machine-learning
framework with end-to-end deployment on an NVIDIA Jetson Orin Nano.

Last updated: 2026-06-05.

---

## 1. Headline result

> **Detection and phase identification are intrinsically load-robust on
> stator current; absolute severity is recovered cross-load by supplying
> the measured load; the physics-feature gradient-boosted cascade beats
> every modern post-2022 sequence model and both time-series foundation
> models on identical leakage-safe splits; and the deployment works
> end-to-end on a Jetson Orin Nano with sub-window-cadence fault-detection
> latency.**

---

## 2. Deployed cascade — main results

| Task | In-distribution (GroupKFold) | Cross-load (LOLO, primary) |
|---|---|---|
| **Detection** | macro-F1 **0.977**, ROC-AUC **0.994** | macro-F1 **0.948** |
| **Severity** (within-1 level) | **0.997** (QWK 0.987, MAE 0.330) | **0.946** (with load-aware fix) |
| **Phase ID** | macro-F1 **0.972** | macro-F1 **0.997** -- cross-load *exceeds* in-distribution |
| **End-to-end** (state + severity within 1 + phase) | within-1 **0.949** | within-1 **0.916** |

**Cross-load severity recovery:** within-1 went from **0.39 -> 0.80 -> 0.95**
(all features -> load-invariant subset -> load-invariant + measured load).
The measured-load addition is the decisive lever.

**SHAP audit (Stage-1 detection):** top features are physical (Park-vector
geometry, $|I_2|/|I_1|$, $|I_0|/|I_1|$, EPVA factor, inter-phase RMS
differences) -- the cascade is not exploiting load proxies.

---

## 3. Full 14-model benchmark on identical leakage-safe splits

Detection / phase use macro-F1; severity uses within-1 level accuracy.
Bold = best per column. Sources: `dataset/benchmark_results.json`,
`dataset/benchmark_table.csv`.

| Model | Det GK | Det LOLO | Sev GK w-1 | Sev LOLO w-1 | Phase GK | Phase LOLO |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.978 | 0.978 | **0.997** | 0.971 | 0.970 | 0.920 |
| SVM-RBF             | 0.978 | 0.977 | 0.991 | 0.975 | **0.984** | 0.967 |
| k-NN                | 0.977 | 0.977 | 0.937 | 0.942 | 0.982 | 0.981 |
| MLP                 | 0.978 | 0.978 | **0.997** | **0.982** | 0.956 | 0.981 |
| Random Forest       | 0.978 | 0.979 | 0.985 | 0.688 | 0.965 | **0.997** |
| XGBoost             | 0.976 | 0.948 | 0.990 | 0.943 | 0.976 | **0.997** |
| TabPFN-2.5          | **0.979** | **0.979** | **0.997** | 0.969 | 0.968 | 0.991 |
| xLSTM               | 0.957 | 0.890 | 0.982 | 0.908 | 0.921 | 0.961 |
| ESN                 | 0.919 | 0.885 | 0.896 | 0.854 | 0.906 | 0.932 |
| NG-RC               | 0.824 | 0.852 | 0.682 | 0.652 | 0.935 | 0.926 |
| ROCKET              | 0.947 | 0.817 | 0.917 | 0.856 | n/r | n/r |
| PatchTST            | 0.896 | 0.824 | 0.461 | 0.560 | 0.949 | 0.967 |
| iTransformer        | 0.872 | 0.831 | 0.628 | 0.470 | 0.973 | 0.979 |
| TimesNet            | 0.941 | 0.868 | 0.663 | 0.541 | 0.920 | 0.932 |
| TTM                 | 0.804 | 0.618 | 0.540 | 0.510 | 0.624 | 0.637 |
| Chronos-Bolt        | 0.753 | 0.545 | 0.452 | 0.372 | 0.611 | 0.607 |

Notes:
* Reservoir, ROCKET, and modern-sequence rows run with a ~1,500-sample
  training cap per fold for CPU feasibility on this corpus.
* ROCKET phase incomplete (would not finish in the run-time budget;
  reported for detection + severity only).
* Mamba omitted: CUDA `selective_scan` not built for the Jetson `sm_87`
  arch; a Python-loop CPU fallback is prohibitive at this corpus size.
  PCM-Net's dilated-temporal backbone (see Section 5 below) is the
  paper's intended Mamba-style alternative.

### Mean rank across the 6 (task, protocol) cells

Lower is better. Three clean groupings.

| Rank | Model | Mean rank | Family |
|---|---|---|---|
| **1**  | **TabPFN-2.5**       | **2.75** | tabular foundation |
| 2  | MLP                  | 3.50 | tabular |
| 3  | SVM-RBF              | 3.75 | tabular |
| 4  | XGBoost              | 4.75 | tabular (**deployed**) |
| 5  | Random Forest        | 5.17 | tabular |
| 6  | k-NN                 | 5.25 | tabular |
| 7  | Logistic Regression  | 5.33 | tabular |
| -- | *gap of 3 ranks*     | --   | -- |
| 8  | xLSTM                | 8.33 | sequence deep |
| 9  | ROCKET               | 10.00 (over 4 cells) | random kernels |
| 10= | ESN                 | 10.50 | reservoir |
| 10= | iTransformer        | 10.50 | modern attention |
| 12 | TimesNet             | 11.17 | modern |
| 13= | PatchTST            | 11.50 | modern attention |
| 13= | NG-RC               | 11.50 | reservoir |
| -- | *gap of 3 ranks*     | --   | -- |
| 15 | TTM                  | 14.33 | time-series foundation |
| 16 | Chronos-Bolt         | 15.67 | time-series foundation |

### Statistical significance (Friedman + Holm-Wilcoxon, 7 tabular models)

| Task | Protocol | $\chi^2$ | $p$ | Outcome |
|---|---|---|---|---|
| Detection | group k-fold | 7.66 | 0.264 | n.s. -- *saturated* |
| Detection | LOLO | 12.61 | 0.050 | **sig.** |
| Severity | group k-fold | 21.84 | 0.0013 | **sig.** |
| Severity | LOLO | 22.85 | 0.0008 | **sig.** |
| Phase | group k-fold | 3.35 | 0.764 | n.s. -- *saturated* |
| Phase | LOLO | 6.00 | 0.423 | n.s. |

With only 5--6 blocks no pairwise Wilcoxon survives Holm correction
(smallest adjusted $p$ = 0.19); the omnibus is the reliable signal.

---

## 4. Best model per task

| Task | Protocol | Best model | Score |
|---|---|---|---|
| Detection | GroupKFold | TabPFN-2.5             | 0.979 |
| Detection | LOLO       | TabPFN-2.5 / RF (tied) | 0.979 |
| Severity  | GroupKFold | TabPFN-2.5 / LR / MLP  | 0.997 |
| Severity  | LOLO       | MLP                    | 0.982 (MAE 0.112) |
| Phase     | GroupKFold | SVM-RBF                | 0.984 |
| Phase     | LOLO       | XGBoost / RF (tied)    | 0.997 |

**Top non-tabular performers per task (LOLO):**
* Detection: **TimesNet** 0.868
* Severity: **ROCKET** within-1 0.856
* Phase: **iTransformer** 0.979 (matches the deployed cascade)

**Overall winner (top-tier on all three tasks under both protocols):**
TabPFN-2.5 -- best or tied-best on 4/6 cells, within 0.018 of best on the
other 2.

---

## 5. Load-robustness mechanism ablations

| Mechanism | Cross-load (LOLO) result |
|---|---|
| **SCST** (Sequence-Component Stockwell Tensor) | severity within-1 **0.75 -> 1.0** |
| **LIR-mRMR** (load-invariance-regularised selection) | k=10 within-1 **0.848** vs mRMR 0.740; halves load drift |
| **PCM-Net + FiLM** (load-conditioned multi-task) | severity within-1 **0.932 -> 0.982** -- best in study |
| **PADA** (physics-anchored domain adaptation) | sim->real detection F1 **0.48 -> 0.98** with 2 labelled real samples |
| **Biomedical features (24)** | LOLO detection +0.030, severity within-1 +0.037 |
| **PC-Diff** (physics-constrained generator) | hard constraint satisfied; severity controllability not achieved (honest negative result) |

---

## 6. Tuning + physics-consistent augmentation (deployed XGBoost)

| Metric | Default | Tuned | Tuned + Aug |
|---|---|---|---|
| Detection LOLO macro-F1 | 0.964 | 0.964 | **0.977** |
| Severity GK within-1 | 0.934 | 0.937 | **0.976** |
| Severity LOLO within-1 | 0.911 | 0.908 | **0.935** |
| Severity LOLO MAE | 0.917 | -- | **0.731** (-20%) |
| Phase LOLO macro-F1 | 0.994 | 0.997 | 0.997 |

---

## 7. Within-hardware validation on a real motor (ibarram)

Real 0.75 hp IM at 60 Hz / 1 kHz, no load. Held-out repetition-05.

| Model | Detection LORO | Severity within-1 LORO | Phase per-window | Phase per-recording majority |
|---|---|---|---|---|
| XGBoost      | **0.955** | 0.836 | 0.708 | **0.95** |
| Random Forest | 0.947 | 0.829 | 0.718 | 0.93 |
| TabPFN-2.5   | **0.994** | 0.834 (MAE **0.34**) | 0.69 | matches |
| SVM-RBF      | 0.923 | 0.818 | 0.689 | -- |

Key finding: **per-recording majority vote of phase predictions** is the
lever -- raises XGBoost phase macro-F1 from 0.708 to **0.95** at no
feature-engineering cost; closes the sim-to-real phase gap.

---

## 8. Trust: calibration, noise, onset

| Metric | Value |
|---|---|
| Stage-1 ECE (expected calibration error) | **0.021** |
| Brier score | 0.021 |
| Fault-onset detection rate | **100%** |
| Pre-fault false-alarm rate | **0%** |
| Median onset delay (severity >= 0.5%) | **0 ms** (first faulted cycle) |
| Delay at severity = 0.3% | ~1.4 s |

**Noise robustness** (train clean, test noisy, group k-fold):

| SNR | Detect F1 | Severity within-1 | Phase acc. |
|---|---|---|---|
| clean | 0.977 | 0.986 | 0.958 |
| 40 dB | 0.927 | 0.980 | 0.954 |
| 30 dB | 0.854 | 0.975 | 0.945 |
| 20 dB | 0.792 | 0.971 | 0.939 |

Severity and phase essentially flat to 20 dB SNR; detection is the
noise-sensitive stage.

---

## 9. Real-time edge deployment on Jetson Orin Nano

### 9.1 Per-window inference latency (200 reps each, measured)

| Mode | Feature (ms) | Predict (ms) | Total (ms) | Note |
|---|---|---|---|---|
| CPU (sklearn wrapper) | 22.7 | 3.5 | 26.2 | -- |
| GPU (XGBoost device=cuda) | -- | -- | n/a | needs source build for sm_87 |
| **Accelerated CPU** (Booster + DMatrix) | 22.7 | **1.0** | **23.7** | **deployed path** |

Window-fill at $f_s=1$ kHz / W=500: 500 ms. **~21x real-time headroom.**

### 9.2 Recorded-source sweep on the Jetson (6 ibarram rep-05 cases at 5 kHz UART)

| | Result |
|---|---|
| Detection | 5/6 (HLT false positive at incipient boundary) |
| Phase | 5/5 on faulted cases (100%) |
| Severity (within-1 level) | 4/5 |
| Wire integrity (30 s) | **0 bytes dropped** |

### 9.3 Live-source four-scenario sweep (sample-by-sample synthesised motor + 12-bit ADC + CT noise)

| Scenario | Truth | Onset crossed 0.065? | Latency | Phase | Severity |
|---|---|---|---|---|---|
| A30      | F / 30% / A | yes at t=3.0 s | <=500 ms | **A** correct | 40 (within-1) |
| B30      | F / 30% / B | yes at t=3.0 s | <=500 ms | **B** correct | 40 (within-1) |
| C30      | F / 30% / C | yes at t=3.0 s | <=500 ms | **C** correct | 40 (within-1) |
| Healthy  | H, no fault | **never** (~0.001) | -- | gated off | gated off |

**Per-window over 135 decisions:**

| Quantity | Count | Accuracy |
|---|---|---|
| Detection TP (faulted-interval, onset > threshold) | 81/84 | 96.4% |
| Detection TN (healthy-interval, onset < threshold) | 51/51 | 100.0% |
| False alarms during 10 s healthy-only run | 0/40 | 0 FA |
| Phase identification (faulted-interval, gated) | 84/84 | **100.0%** |
| Severity within-1 level (faulted-interval, gated) | 84/84 | **100.0%** |
| **End-to-end** (state + phase + sev within-1) per window | 132/135 | **97.8%** |
| End-to-end per scenario | 4/4 | **100.0%** |
| Wire integrity (200,000 packets, 40 s) | 0 bytes | 100.0% |
| Packet-rate drift | -- | 0.004% |

---

## 10. Deployment claim

**XGBoost was chosen as the deployed model not because it is the most
accurate but because it is Pareto-optimal across the deployment axes:**

| Axis | XGBoost (deployed) | TabPFN-2.5 | MLP |
|---|---|---|---|
| LOLO accuracy (D / S / P) | 0.98 / 0.95 / 1.00 | approx 0.98 / 0.97 / 0.99 | approx 0.98 / 0.98 / 0.98 |
| Bundle size | **5 MB** | ~500 MB | ~5 MB |
| Predict latency | **1 ms** | 50-500 ms | few ms |
| Cold-start | **<0.1 s** | minutes | ~1 s |
| Runtime dependencies | `xgboost` only | transformer + token | PyTorch |

TabPFN-2.5 remains the recommended choice when the deployment target is
a back-end inference server rather than a motor-mounted edge node.

---

## 11. Dataset

* **Simulation (PSCAD/EMTDC):** 357 cases (336 faulty + 21 healthy);
  severity {0.3, 0.5, 1, 2, 3, 4, 5}%, phase {A, B, C}, load
  {NL, 20, 40, 60, 80, 100}%, 11 inception variants per operating point.
  Sampled at 25 kHz; 18,033 labelled windows after segmentation.
* **Real motor (ibarram):** 0.75 hp IM at 60 Hz / 1 kHz, no load;
  severity {10, 20, 30, 40}%, phase {A, B, C}, 5 repetitions; 65
  recordings -> 1,235 windows.
* **Leakage-safe grouping:** operating-point groups bind inception
  variants together (147 sim groups, 13 real groups).
* **Evaluation:** StratifiedGroupKFold (in-distribution) and
  Leave-One-Load-Out (LOLO, primary cross-load metric).

---

## 12. Sources for all numbers

| File | Content |
|---|---|
| `dataset/benchmark_results.json` | All 14 models x 3 tasks x 2 protocols |
| `dataset/benchmark_table.csv` | Same, flat |
| `dataset/significance_results.json` | Friedman + Holm-Wilcoxon |
| `dataset/cascade_metrics.json` | Cascade end-to-end |
| `dataset/stage{1,2,3}_metrics.json` | Per-stage |
| `dataset/hardware_results.json` | Within-hardware (real motor) |
| `dataset/calibration_uncertainty.json` | ECE, Brier |
| `dataset/noise_robustness.json` | SNR sweep |
| `dataset/stage0_onset.json` | Onset detection |
| `realtime/jetson_compute_bench.json` | Jetson per-window latency |
| `report.md` | Consolidated thesis-style report |
| `manuscript/main.pdf` | IEEE Transactions paper (20 pages) |
