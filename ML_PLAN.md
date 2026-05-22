# ML Plan — Hierarchical Stator Turn-to-Turn Fault Diagnosis

**Goal:** From three-phase stator current, build a **3-stage cascade**:
1. **Stage 1 — Detection:** Healthy vs. Faulty (binary)
2. **Stage 2 — Severity:** if faulty, classify % shorted turns (0.3 / 0.5 / 1 / 2 / 3 / 4 / 5 %)
3. **Stage 3 — Phase ID:** if faulty, identify the faulted phase (A / B / C)

**Data scope:** ONLY `Healthy case data` + `TT fault Data`. (HRC and LL excluded.)
**Signals:** three-phase stator currents = `_02.out` cols 2,3,4 (`Ia_ABCIM_s`) — or `_03.out` cols 2,3,4 (`Ia_ABCIM`, kA). Pick one consistently; recommend `_03` (kA) so magnitudes are physical and phase imbalance is preserved.

---

## 1. Problem framing
Hierarchical/cascade rather than one flat multi-class model, because the three questions are nested and have different decision boundaries:
- A flat model mixing 1 healthy + (7 severities × 3 phases) classes wastes the structure and imbalances classes.
- Cascade lets each stage specialise; errors are interpretable per stage.
- Severity is **ordinal** (0.3 < 0.5 < … < 5) → can be classification or ordinal regression.

---

## 2. Data inventory & labelling

| Source | Cases | Use |
|---|---|---|
| Healthy (21 loadings, 5 s) | 21 | Healthy class |
| TT regular (sev × phase × load) | 126 | Faulty class + severity + phase labels |
| TT "Extra" (× 10 inception times) | 210 | Faulty class + severity + phase labels |

**Labels per case:** `state ∈ {healthy, faulty}`, `severity ∈ {0.3…5}` (faulty only), `phase ∈ {a,b,c}` (faulty only), plus metadata `load`, `inception_time`.

**Key timeline (samples @ 25 kHz):** motor start 50,000 (2.0 s) · load change 100,000 (4.0 s) · **fault 180,000 (7.2 s)** · clear ~200,000 (8.0 s). Healthy files end at 125,000 (5 s).

### Where to take "healthy" and "faulty" segments
- **Faulty windows:** from the fault interval **180,000 – 200,000** (0.8 s, 40 cycles) of each TT file.
- **Healthy windows — two sources:**
  1. Steady region of the 21 Healthy files (after load change, ~100,000 – 125,000).
  2. **Pre-fault steady region of every TT file (~110,000 – 178,000).** Before 7.2 s the simulated motor is genuinely healthy.
- **Why source 2 matters:** it gives abundant healthy data **at the exact same loads as the faulty windows**, so Stage 1 cannot cheat by learning "load level" instead of "fault." This removes the biggest confound.

> **Confound warning:** healthy files use 5 % load steps; TT uses NL/20/40/60/80/100 %. If healthy came only from the 21 files, the model could separate classes by load, not by fault. Using TT pre-fault regions fixes this.

---

## 3. Preprocessing & windowing
1. Load `.out` with `np.loadtxt` (skips the blank first line). Extract t + 3 current columns.
2. (Optional) drop transients: ignore ±1 cycle around start/load-change/fault edges.
3. **Window** each steady region into fixed segments:
   - Window length: **2,000 samples = 4 cycles (0.08 s)** (start here; also try 500 = 1 cycle and 5,000 = 0.2 s).
   - Overlap: 50 %.
   - Each window → one ML sample, inheriting its case labels.
4. **Normalisation:** scale all three phases by a **single common factor per window** (e.g., positive-sequence RMS or rated current) — NOT per-channel — so inter-phase imbalance (the fault signature, and the phase-ID cue) is preserved. Standardise engineered features with a scaler fit on the **train split only**.

**Expected window counts (rough):** faulty ≈ 336 × ~19 ≈ 6.4k; healthy (pre-fault) ≈ 336 × ~60 ≈ 20k+. Plenty for both classical and deep models; subsample healthy to balance Stage 1.

---

## 4. Train / validation / test split — anti-leakage (critical)
- **Split by simulation CASE, never by window.** Windows from one `.out` must not appear in both train and test (they are near-duplicates). Use **GroupKFold / StratifiedGroupKFold** with `group = case_id`.
- **Generalisation tests (report both):**
  - *Random group split* (5-fold): in-distribution performance.
  - *Leave-One-Load-Out*: train on some loads, test on a held-out load → does it generalise to unseen operating points? (the realistic deployment question.)
- For the "Extra" inception-time cases, keep all 10 inception variants of a given (sev, phase, load) **in the same fold** to avoid near-duplicate leakage.

---

## 5. Feature engineering (Track A — classical ML)
Compute per window, from the 3-phase current:

**Per-phase (×3):** RMS, peak, crest factor, std, skewness, kurtosis, fundamental (50 Hz) magnitude, THD, magnitudes of 3rd/5th/7th harmonics.

**Three-phase / cross-phase (the discriminative ones):**
- **Symmetrical components** I0, I1, I2 (Fortescue from 50 Hz phasors) → **negative-sequence ratio I2/I1** (classic inter-turn indicator; scales with severity).
- **Current unbalance** (NEMA), max phase-RMS deviation.
- **Park's vector** (Clarke→ d-q): id, iq; ellipse features — major/minor axis, eccentricity, tilt angle. Extended Park's Vector Approach (EPVA) spectral peak.
- **Phase-ID cues:** index of phase with max RMS / max harmonic content; **angle of the negative-sequence current** (points to the faulted phase).

Output: a tabular feature matrix `[n_windows × n_features]` + label columns.

## 5b. Feature engineering (Track B — deep learning, raw)
- Input tensor `[3 channels × window_len]` (3-phase current), common-factor normalised.
- Optionally add a **time-frequency image** per phase (STFT spectrogram or CWT scalogram) → `[3 × F × T]` for a 2D-CNN.

---

## 6. Models per stage
Run **Track A first** (fast, interpretable baseline), then Track B if needed.

| Stage | Track A (baseline) | Track B (if more accuracy) | Notes |
|---|---|---|---|
| 1 Detection | RandomForest / XGBoost / SVM-RBF | 1D-CNN on raw current | Likely near-perfect; check it isn't load-leakage |
| 2 Severity | XGBoost (multiclass) **or ordinal regression** | 1D-CNN / CNN-LSTM | Report MAE in severity levels; expect adjacent-class confusion (0.3 vs 0.5) |
| 3 Phase ID | RandomForest / XGBoost (3-class) | 1D-CNN | Negative-seq angle + per-phase RMS should make this strong |

- Severity as **ordinal regression** (predict continuous % then bin) is recommended since 0.3↔0.5 confusion is "less wrong" than 0.3↔5.
- Use class weights / balanced sampling where needed (esp. Stage 1 healthy vs faulty).

---

## 7. Cascade integration & inference
```
window → Stage1
   ├─ Healthy → output "Healthy"
   └─ Faulty  → Stage2 (severity)  +  Stage3 (phase)  → output (severity, phase)
```
- Stages 2 & 3 are trained **only on faulty windows** (their natural domain), but **evaluated on Stage-1's faulty predictions** to get true end-to-end numbers.
- Optionally aggregate multiple windows per case by majority vote for a per-case decision (more robust than per-window).

---

## 8. Evaluation protocol
- **Stage 1:** accuracy, precision/recall/F1, ROC-AUC, confusion matrix.
- **Stage 2:** macro-F1, confusion matrix (inspect adjacent-severity errors), **MAE in #severity levels** (ordinal).
- **Stage 3:** 3-class accuracy + confusion matrix.
- **End-to-end cascade:** exact-match accuracy of (state, severity, phase); per-stage error attribution.
- **Always report:** GroupKFold (by case) AND Leave-One-Load-Out results; per-load and per-severity breakdowns.
- Track-A bonus: feature-importance / SHAP to confirm the model uses physically-meaningful features (I2/I1, harmonics) — guards against leakage.

---

## 9. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Window leakage (same case in train+test) | Group split by case |
| Load confound in Stage 1 | Use TT pre-fault windows as healthy (matched loads) |
| Healthy/TT load grids differ (no 5 % in TT) | Don't pair by load; rely on pre-fault healthy |
| Severity adjacent-class confusion | Ordinal regression + report MAE, not just accuracy |
| Few distinct cases (overfitting DL) | Prefer classical baseline; heavy windowing; regularise; case-level CV |
| Per-channel normalisation hides phase imbalance | Common-factor normalisation |
| `_02` (normalised) vs `_03` (kA) mixing | Fix one current source |

---

## 10. Implementation roadmap
1. **`data_index.py`** — walk Healthy + TT folders, build a manifest table: `case_id, path_02/03, state, severity, phase, load, inception, n_samples`.
2. **`segment.py`** — given manifest + region rules + window params → produce windowed arrays (or save to `.npy`/parquet) with labels and `group=case_id`.
3. **`features.py`** — Track A feature extraction (per-phase + sequence + Park + phase-ID cues).
4. **`split.py`** — StratifiedGroupKFold + Leave-One-Load-Out splitters.
5. **`train_stage1/2/3.py`** — train + CV each stage (XGBoost baseline).
6. **`cascade.py`** — wire stages, end-to-end evaluation, confusion matrices, plots.
7. **(opt) `cnn.py`** — Track B 1D-CNN for comparison.
8. **`report.md`** — metrics tables + figures.

**Milestones:** (M1) manifest + windows → (M2) features → (M3) Stage-1 baseline + leakage check → (M4) Stages 2&3 → (M5) cascade + Leave-One-Load-Out → (M6) optional CNN + final report.

---

## 11. Open design choices (defaults chosen, can revisit)
- Window length 4 cycles / 50 % overlap — *default, will sweep.*
- Current source `_03` kA — *default.*
- Classical (XGBoost) first, CNN optional — *default.*
- Severity = ordinal regression — *default.*
- Per-case majority-vote aggregation for final decision — *recommended.*
