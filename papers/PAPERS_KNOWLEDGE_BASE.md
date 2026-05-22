# Literature Knowledge Base — Induction-Motor Fault Detection & Classification

Synthesis of the 18 papers in `D:\Naveen\papers\` (read in full, 2026-05-22), focused on
**induction-motor (IM) stator inter-turn / turn-to-turn short-circuit (ITSC/TSCF)** detection,
severity estimation, and faulted-phase identification from three-phase stator current — i.e. the
exact problem of this project. Each entry: relevance, method, results, code/data.

**Triage:** of 18 PDFs, **~6 are directly relevant** to IM stator faults, **~4 are motor-adjacent**
(PMSM / vibration / inverter), and **8 are off-topic** (computer-vision or general anomaly-detection
theory, apparently included by mistake). The off-topic set still yields a few transferable ideas.

---

## A. DIRECTLY RELEVANT — IM stator inter-turn faults

### A1. ★ Turn-to-Turn SC & High-Resistance Connection Diagnosis during Start-Up Transient
Mazzoletti, Bossio, Donolo, Donolo — **IEEE Trans. Industry Applications 61(2), 2025**. *(No code; custom 7.5 hp prototype data.)*
**Most relevant paper — it is the analytical method for OUR exact fault set (TT + HRC).**
- Soft-starter-fed 3-φ squirrel-cage IM; diagnoses **TSCF vs HRC** (must be decoupled: TSCF→shutdown, HRC→monitor) during **start-up transient**, 8 kS/s, only 2 current sensors.
- **Core feature:** positive-sequence component of the **5th-harmonic stator current `Isp5`**, extracted via αβ→dq demodulation at ±5ωe + low-pass filter (lighter than FFT/Kalman).
- **Physics:** for TSCF, `Isp5 = (1/3)·μqd·If` (μqd = shorted-turn fraction = severity; If = short-circuit current) → severity indicator. For HRC, `Isp5 = (1/3)·(ra/Zsp5)·Isn5` → depends on contact resistance & negative-sequence current, **independent of fault current** → decouples HRC from TSCF.
- **Faulted-phase ID** encoded in per-phase direction vectors nq,nd: a=[1,0], b=[−½,−√3/2], c=[−½,+√3/2].
- **Classification rule:** TSCF keeps Isp5 elevated through the whole ramp; HRC only spikes in t=0–3 s then drops. Threshold = 1.2× healthy mean.
- Detects ≥3 shorted turns (~2%) at rf=0; robust to supply unbalance (VUF 1%, 3%).
- **Caveat for us:** relies on harmonic distortion (soft-starter/inverter); a clean DOL/PSCAD sinusoidal supply has weak 5th harmonic, so the negative-sequence-fundamental route (which we already use, |I2|/|I1|) is the better fit. The TSCF-vs-HRC decoupling logic is directly applicable to extending our cascade to fault-TYPE classification.

### A2. ★ Predicting Stator Winding Short-Circuit Faults using ML: A Comparative Study
Aasar, Sultan, Khan, Czanner, Abbasi, Ahmad — **Discover Computing 29:103, 2026** (open access). *(Public Kaggle dataset; no model code.)*
- 3-φ IM ITSC; **binary detection + 7-class severity** (healthy + 6 ITSC levels). Signals: stator current (4 winding points) + leakage flux, variable load (NL/half/full).
- Dataset: **Kaggle "MIT short circuit flux and current signals"** (Cunha 2021), 14,000 samples (balanced 7×2000).
- Models: ANN (74% 7-class), **LSTM (89% 7-class, best)**, PINN (75%). Binary: **LSTM 100%, PINN 99.3%, ANN 99%** across loads.
- **PINN** adds physics losses: Kirchhoff current balance `‖I1+I2+I3‖²` + flux-consistency `‖φ−f(I)‖²` → regularizer, best in small-data regime.
- **Key finding (matches ours):** mid-severity classes overlap and are the hard part; temporal model (LSTM) helps. **No faulted-phase ID** (gap).
- Related dataset found: **github.com/ibarram/ITSC** — 3-φ current, 13 classes = per-phase ITSC at 10/20/30/40% + healthy (DOES label faulted phase — useful for phase-ID benchmarking).

### A3. ★ Incipient Fault Detection in Stator Windings using Stockwell Transform + SVM
Singh & Shaik (IIT Jodhpur) — **IEEE Trans. Instrumentation & Measurement 69(12), 2020**. *(No public code/data.)*
- 3-stage scheme on 3-φ stator current (0.75 kW IM, **6.4 kHz**, no-load): (i) detect, (ii) discriminate **inter-turn vs phase-to-ground**, (iii) locate **faulty phase A/B/C**.
- **Feature:** standard deviation of **Stockwell-transform magnitude** at turn-short sidebands **125/175/225 Hz** (avoids 25/50/75/100 Hz which alias with eccentricity/asymmetry). Type discrimination via **zero-sequence current** I0=(Ia+Ib+Ic)/3 (large for ground, small for turn).
- Two RBF-SVMs, grid-searched (C,γ), heuristic feature subset selection (best 6 of 9).
- **Results: faulted-phase ID 100% for turn faults**, 91.7% for ground; ~96% average. **No severity estimation; single 1.6% severity, no-load only.**

### A4. ML for Inverter-Fed Motors Monitoring & Fault Detection: An Overview
García-Pérez et al. (Univ. Oviedo) — **IEEE Access 12, 2024** (open access). *(Code: github.com/gsdpi/MLforInvertedFedMotors, MIT; no public raw data.)*
- Inverter-fed IM **stator insulation degradation** (precursor of turn shorts) + winding-temperature estimation. 3-φ current+voltage, 750 kHz→resampled 20 kHz; 28 run-to-failure tests.
- **Feature:** Clarke αβ → complex FFT → **direct & inverse (negative) sequence impedance** `Z_h = U(h)/I(h)`, `Z_h⁻ = U(h)/I(−h)` (negative-seq impedance = classical asymmetry/turn-fault descriptor). Harmonics −5, 7, 13.
- Models: time-domain (LSTM, CNN, TCN, ESN, ROCKET) vs frequency-domain (MLR, SVR, MLP) for temperature; **residual-based detection** (train on healthy, flag deviation). CNN/LSTM best (time), MLP/SVR best (freq). UMAP to visualize degradation.
- Relevance: residual scheme suits incipient detection without labeled faults; negative-seq impedance front-end reusable for phase-ID.

### A5. Various Faults Classification of IM using Supervised ML: A Comprehensive Review
Hussain, Saleh, Refaat (Texas A&M) — **IEEE Access 13, 2025**. *(Review; no code — and it criticizes the field for this.)*
- Best literature map for our task. Fault prevalence: bearing 40%, **stator winding 38%**, rotor 10%.
- ~17 signal-processing techniques (DWT most popular; RQA used to "image" turn faults) × shallow ML (SVM most cited).
- **Key inter-turn results cited:** SVM+Akima envelope-energy **100%** (20 kHz, early-vs-severe); DWT(db2)+SVM **99.99%** (6 severity classes, 10 kHz); **RQA images + soft-voting ensemble 98%** turn-severity at **25 kHz** (Sharma et al. [72] — same fs as us); ANN with negative-sequence-voltage input separates **ITSC from supply unbalance** [89]; wavelet-packet+RF 98.67%.
- **Identified field gaps (we address several):** no standardized public dataset; suspicious 100% accuracies (overfitting / test≈train); missing train/test split details; **almost no published code**; few real-time validations. Calls for physics-informed NNs, GitHub releases, IoT.

### A6. ML-Based Diagnosis for Single- & Multi-Faults in IMs using Stator Currents & Vibration
(Group-D paper) — *(No public repo.)*
- 3-φ stator current + vibration; matching-pursuit (MP) / DWT + statistical features; ~100% accuracy for top classifiers; load-interpolation curve-fitting (<8% error). Processes essentially one current component (I2) → **cannot identify faulted phase as-is**; would need per-phase / sequence-component features. Useful as feature-pipeline reference.

---

## B. MOTOR-ADJACENT (PMSM / vibration / different fault) — method transfer only

### B1. ML-Based Stator-Current PMSM Stator Winding Fault Diagnosis
Pietrzak & Wolkiewicz — **Sensors 22:9668, 2022**. *(No code/data; "Data Availability: Not applicable".)*
- **PMSM** ITSC (0–5 shorted turns ≈ ≤2%), severity classification. STFT of **positive & negative instantaneous symmetrical components** of 3-φ current; **best indicator = f_s amplitude in the negative-sequence spectrogram (load-robust)**; the positive-seq 3·f_s indicator saturates at rated load.
- SVM (Gaussian, σ=0.4) 96–97%, MLP (3-7-15-1) **99%**, NB 79% (poor; correlated features). Online verified.
- Transfer: confirms negative-sequence current as the severity-sensitive, load-robust ITSC feature (PMSM has no zero-seq path; IM needs slip-aware recalibration). Faulted phase fixed to A → no phase-ID.

### B2. An Improved Fault Diagnosis Strategy using Weighted Probability Ensemble DL (WPEDL)
Ali, Ramzan, Ali, Al-Jaafari — **IEEE Access 13, 2025** (arXiv:2412.18249). *(No code; 3 public datasets.)*
- Unified bearing+rotor+stator. **STFT spectrograms → 6 fine-tuned CNNs (ResNet18/GoogleNet/DenseNet/MobileNetV2/ShuffleNetV2/SqueezeNet) → metric-weighted probability ensemble.** ~98.9–99.6% accuracy; STFT-CNN > DWT features.
- Stator data is **PMSM** (ITSC vs ICSC vs healthy), not IM; rotor data is true IM (BRB). Fault-TYPE classification, not severity/phase.
- Public datasets: IEEE DataPort BRB (Treml, doi 10.21227/fmnm-bn95); Mendeley PMSM stator (Jung); MaFaulDa bearing.

### B3. Multi-fault Diagnosis of Rotating Machine under Uncertain Speed (STFNet)
*(No code; proprietary data.)* Vibration-only, broken-rotor-bar focus. STFT-segmentation + LSTM ("STFNet") 92–98% across uncertain speeds; **95%-overlap windowing as augmentation**; speed/load-independent training strategy. Transfer: windowing-augmentation + temporal model under varying operating conditions.

---

## C. OFF-TOPIC (computer-vision / general anomaly-detection theory) — included by mistake

| File | What it actually is | Code | Transferable idea (only) |
|---|---|---|---|
| `2412.04780` SEKA | Knowledge-graph anomaly detection (ν-SVM on path features) | github.com/AsaraSenaratne/SEKA | one-class SVM framing |
| `2505.02626` VELM | Visual industrial anomaly *classification* with MLLMs (GPT-4o) | github.com/Sassanmtr/VELM | two-stage "fast detector + downstream classifier"; anomaly-vs-defect (severity) framing |
| `2309.00379` rAD | Semi-supervised AD via risk estimators (PU-learning theory) | github.com/LeThiKhanhHien/rAD | learn detector from few labels + abundant unlabeled |
| `2511.06644` UniADC | Unified visual AD+classification (CLIP/DINOv3 + diffusion synth) | github.com/cnulab/UniADC | few-shot/zero-shot; class-aware head ≈ phase-ID |
| `2212.12092` ECET | Ensemble + Dempster-Shafer evidence theory (Tennessee Eastman) | none (TE data public) | evidence-theory fusion of per-phase classifiers; uncertainty→OOD flag for unseen severities |
| `2602.16182` World-Model | Robot inspection failure detection (Cosmos tokenizer + world models) | autoinspection-classification.github.io | **conformal-prediction calibrated thresholds**; predictive-residual scoring; "OOD only if both models fire" |
| `2403.14213` MINT-AD | Multi-class visual AD (transformer + INR class-aware query) | none | class-aware modeling to avoid inter-class interference across phases/severities |
| `2411.10150` | Outlier-resistant assembly-line image classification | none | metric learning + distance-threshold OOD |

---

## D. Cross-cutting insights & actionable takeaways for THIS project

**Validated consensus (multiple papers):**
1. **Negative-sequence current is THE inter-turn indicator** — `|I2|`, `|I2|/|I1|`, negative-seq impedance, or negative-seq f_s STFT amplitude. (A1, A4, B1, A5[89].) We already use `|I2|/|I1|` — well-supported.
2. **Faulted phase is recoverable from per-phase / sequence-component features** — Stockwell SD gives 100% (A3); per-phase direction vectors (A1). Matches our 0.997 LOLO phase-ID.
3. **Severity mid-levels overlap** — the universal hard problem (A2 LSTM 89% 7-class; B1; A5[33]). Our within-1 framing + load-aware fix is the right response.
4. **Time-frequency + temporal models help** — STFT/Stockwell/RQA + SVM/CNN/LSTM dominate; RQA-images+ensemble hit 98% severity at **25 kHz** (our exact fs).

**Ideas worth adopting (Phase-4+ extensions):**
- **Fault-TYPE discrimination (TT vs HRC vs LL)** using A1's decoupling physics — our dataset HAS all three; this is a natural, novel extension (our cascade currently does TT-only severity/phase).
- **PINN current-balance loss** `‖I1+I2+I3‖²` (A2) and **zero-sequence-current index** (A3) as extra physics features/constraints.
- **Conformal-prediction thresholds** (C/World-Model) — we already started conformal (Phase 4).
- **Residual-based detection** trained on healthy only (A4) — alternative to our supervised Stage-1.
- **RQA imaging + soft-voting ensemble** (A5[72]) as a Track-B alternative to ResNet-1D.

**Where our project already leads the field (per A5's gap analysis):**
- We use **leakage-safe splits + Leave-One-Load-Out** (most papers report suspicious 100% with unclear splits).
- We **published code + a results report** publicly (the review notes almost no code is released).
- We do **detection + severity + faulted-phase in one cascade** with honest cross-load evaluation — few papers do all three.

**Public datasets/repos to reuse for external validation:**
- Kaggle MIT ITSC (current+flux, 7 severity): kaggle.com/datasets/rebecacunha/mit-short-circuit-flux-and-current-signals
- github.com/ibarram/ITSC (per-phase ITSC, 13 classes — for phase-ID benchmarking)
- IEEE DataPort BRB (Treml): doi 10.21227/fmnm-bn95 · Mendeley PMSM stator (Jung) · MaFaulDa bearing
- Method code: github.com/gsdpi/MLforInvertedFedMotors (sequence-impedance + residual ML)
