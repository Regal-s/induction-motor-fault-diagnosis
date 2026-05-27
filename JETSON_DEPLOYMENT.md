# NVIDIA Jetson Edge-Deployment Plan — Stator Inter-Turn Fault Diagnosis

Real-time, on-motor deployment of the diagnosis cascade on an NVIDIA Jetson "controller".
Target: continuously diagnose a running induction motor from its three-phase current —
onset → detection → severity → faulted phase — at the edge, no cloud.

The deployed model is tiny (feature extraction + XGBoost cascade ≈ **0.34 ms/window on a desktop CPU**),
so a Jetson is comfortably real-time with large headroom; the GPU is spare capacity for the deep /
SCST track and future models.

---

## 1. Hardware

| Option | When | Notes |
|---|---|---|
| **Jetson Orin Nano (8 GB)** | recommended default | ample for the feature+XGBoost cascade + a small CNN; ~7–15 W |
| Jetson Orin NX (8/16 GB) | if running the deep/SCST track or many motors | 2× DLA cores for INT8 inference |
| Jetson AGX Orin | multi-motor / high-rate / research headroom | overkill for one motor |

**Signal chain:** 3× current transducers (LEM/CT, ±, isolated) → anti-alias filter → ADC.
- Match the training rate where possible. Sim model trained at 25 kHz; the real (ibarram) path at 1 kHz.
  For a deployed motor pick a fixed `fs` (e.g. 10–25 kHz), and **retrain/recalibrate the feature
  pipeline at that rate** (the extractor is already `(fs,f0)`-parameterised, so this is a config change).
- ADC options: USB DAQ (e.g. NI/MCC), an SPI/I²S ADC HAT, or the drive's own current feedback if exposed.
- Time-sync the 3 channels (simultaneous-sampling ADC) — inter-phase phase is the fault signature.

---

## 2. Software stack
- **JetPack 6.x** (L4T, CUDA 12, cuDNN, **TensorRT 10**). Python 3.10/3.11.
- `numpy`, `scipy`, `pandas` (feature extraction); `xgboost` (CPU inference is plenty), or **Treelite**/
  **ONNX Runtime** for the tree cascade; **onnxruntime-gpu** + **TensorRT** for any deep model.
- `antropy`/`EntropyHub` for the biomedical features (pure-Python; fine on Jetson CPU at the window rate).
- Containerise with the **`nvcr.io/nvidia/l4t-*`** base image; ship as a Docker image + `systemd` service.

---

## 3. Model export & optimisation

### 3.1 Deployed cascade (primary — feature + XGBoost)
1. **Train on the host** (CPU/GPU) → 3 boosters: Stage-1 detect, Stage-2 severity (load-aware), Stage-3 phase.
2. **Export** each booster:
   - `Treelite` → compiled shared lib (fastest CPU tree inference), **or**
   - `onnxmltools`/`onnxconverter-common` → ONNX → **ONNX Runtime** (CPU or CUDA EP).
3. **Feature extractor**: keep the vectorised NumPy `build_feature_frame` (incl. the `n_*` and biomedical
   blocks). At one decision per 4-cycle window (~80 ms at 50 Hz) the per-window cost (<1 ms) is negligible;
   only optimise (Numba/Cython/C) if running many motors per Jetson.
4. **Stage-0 onset trigger**: the per-cycle `|I2|/|I1|` threshold runs continuously as a cheap C/NumPy
   loop and gates the cascade (Stage0 → Stage1 → Stage2/3), exactly as in `stage0_onset.py`.

### 3.2 Deep / SCST track (optional, uses the GPU/DLA)
- Export xLSTM / PCM-Net / 1-D CNN (or the SCST-tensor CNN) `torch → ONNX → TensorRT` engine.
- **FP16** (default on Orin) ≈ 2–3× over FP32; **INT8** with calibration ≈ 4× and runs on the **DLA**,
  freeing the GPU. Build the engine once on the target (`trtexec --onnx=model.onnx --fp16 --saveEngine=...`).
- Only needed if the deep models are shown (on a GPU host) to beat the feature cascade; today they do not.

---

## 4. Real-time inference pipeline (on the Jetson)
```
              ┌─ ADC (3φ current, simultaneous-sampling) ─┐
acquire ─────▶│  ring buffer (continuous)                 │
              └───────────────────────────────────────────┘
   │  every cycle: Stage-0 onset index |I2|/|I1| (cheap)  ──▶ if > threshold for 3 cycles → ARM
   │  every 4-cycle window (≈80 ms @50 Hz, 50–75% overlap):
   ▼
  preprocess (per-recording / running normalisation)
   ▼
  feature extraction (72 physics + biomedical block)        [<1 ms]
   ▼
  Stage-1 detect ── healthy ─▶ idle / log
        │ faulty
        ▼
  Stage-2 severity (load-aware: read measured load) + Stage-3 phase
   ▼
  per-recording aggregation (median severity, majority-vote phase)
   ▼
  output: {state, severity%, phase, onset time, calibrated prob, confidence interval}
          → MQTT / Modbus / OPC-UA / relay / dashboard
```
**Latency budget:** a decision is needed only once per window (~80 ms); the whole chain is ≪1 ms of
compute, so a single Jetson can monitor **many** motors or run the heavier deep/SCST track in parallel.

---

## 5. On-target validation & calibration
- **Per-machine calibration** (the PADA few-shot result): collect a handful of labelled healthy +
  known-fault windows on the target motor and fine-tune / re-standardise — recovers detection to ~0.98
  and improves phase (the sim→real gap).
- **Onset threshold**: calibrate `mean+6σ` of `|I2|/|I1|` on a trailing healthy window of the target.
- **Probability calibration**: keep the Stage-1 calibration (ECE 0.021); re-check on target data.
- **Phase caveat**: per-window phase is noisy on real hardware (macro-F1 ~0.72) — use **per-recording
  majority vote** (recovers to ~0.93–0.95), which is the realistic on-device decision granularity.

---

## 6. Ops / productionisation
- **Packaging**: Docker (`l4t-base`) image, `systemd` unit, watchdog; models + scaler + thresholds in a
  versioned artifact bundle.
- **OTA model updates**: pull a new artifact bundle (boosters + ONNX/TRT engines + calibration) and
  hot-swap; keep a rollback.
- **Logging/telemetry**: ring-buffer raw current on alarm for post-hoc analysis; stream diagnoses + the
  interpretable drivers (negative-sequence ratio, EPVA factor) to the SCADA/MQTT layer.
- **Power/thermal**: Orin Nano 7–15 W; passive/active cooling per enclosure; set the `nvpmodel` power mode.
- **Safety/EMC**: isolated current sensing, surge protection; the Jetson is monitoring-only (advisory) unless
  wired into a protection relay, which needs functional-safety review.

---

## 7. Milestones
1. **M1 — Bench bring-up:** Jetson + ADC acquiring synchronous 3φ current; verify waveforms.
2. **M2 — Cascade port:** Treelite/ONNX export of the 3 boosters + feature extractor running on-device;
   reproduce host accuracy on recorded windows.
3. **M3 — Real-time loop:** ring buffer + Stage-0 onset + windowed cascade + per-recording aggregation;
   measure end-to-end latency and CPU/GPU load.
4. **M4 — On-target calibration:** few-shot per-machine calibration + onset/probability calibration.
5. **M5 — Deep/SCST track (optional):** ONNX→TensorRT FP16/INT8 engine on GPU/DLA; only if it beats the
   cascade on a GPU host.
6. **M6 — Productionise:** Docker + systemd + OTA + MQTT/Modbus integration; field trial.

---

## 8. What is ready vs. needed
- **Ready now:** the full training/feature/cascade code (host), GPU-ready (`gpu.py`, `FAULT_DEVICE=cuda`),
  Stage-0 onset, calibration, per-recording aggregation, the `(fs,f0)`-parameterised extractor.
- **Needed for Jetson:** the export step (Treelite/ONNX/TensorRT), the acquisition/ring-buffer service,
  the real-time loop wrapper, and an on-target calibration pass. None require model changes — they wrap
  the existing artifacts.
- **Caveat:** diagnosis accuracy is validated in simulation + one public real dataset; a Jetson field
  deployment is the bridge to the full instrumented campaign noted as future work.
