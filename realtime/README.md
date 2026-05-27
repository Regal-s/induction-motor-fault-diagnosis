# Real-time streaming diagnosis: laptop → Ethernet → NVIDIA Jetson

Replays saved **real-machine** three-phase current from a laptop over TCP/Ethernet to a Jetson
(or any host), which extracts features and runs the deployed cascade in real time:

```
onset (|I2|/|I1|)  →  detect (healthy/faulty)  →  severity (%)  →  faulted phase
```

Diagnosis is per-window plus a per-recording aggregated verdict (median severity, majority-vote
phase — the realistic decision granularity).

## Components
| File | Runs on | Role |
|---|---|---|
| `train_deploy.py` | host (once) | trains the cascade on the real ibarram data (reps 01–04; rep 05 held out) → `deploy/` bundle |
| `sender.py` | **laptop** | replays a recording's 3-phase samples over TCP at the real rate |
| `receiver.py` | **Jetson** | TCP server: buffer → window → features → cascade → verdict |
| `streamio.py` | both | recording reconstruction + `CascadeInfer` (loads `deploy/`) |
| `deploy/` | Jetson | `detect.json`, `severity.json`, `phase.json` (XGBoost) + `meta.json` |

## Run
On the **Jetson** (receiver):
```bash
FAULT_DEVICE=cuda python -u realtime/receiver.py --port 9009     # FAULT_DEVICE=cpu on a CPU box
```
On the **laptop** (sender), point at the Jetson's IP:
```bash
python realtime/sender.py --host <JETSON_IP> --port 9009 --case ibarram_ITSC_B30_Repetition05
python realtime/sender.py --host <JETSON_IP> --all-rep 05        # stream every held-out rep-05 recording
# --speed N  : playback speed (1 = real time, e.g. 60 = 60x faster for a quick demo)
```

## Protocol (TCP, one connection per recording)
1. Sender → newline-terminated JSON header: `{"case_id","fs","n","channels":3,"truth":{...}}`
2. Sender → little-endian `float32` stream, packets of `packet` samples (`packet*3` floats, interleaved
   `[t0:a,b,c, t1:a,b,c, …]`), paced at `fs`.
3. Receiver windows at `W=500` (30 cycles @ 60 Hz), `stride=250`, classifies each window, prints a
   per-window line, and on connection close prints the aggregated `VERDICT` vs `truth`.

## Deployed model & notes
- **XGBoost cascade trained on the real data** (fast, exportable, no token). The sim-trained model is
  NOT used for the real stream (it needs PADA domain adaptation).
- **Context-dependent compensated features (`I2c_*`) are excluded** from the deployed model: they need a
  per-recording healthy baseline that a live single-window stream lacks. The plain `|I2|/|I1|` features
  carry the same signal and are fully self-contained. (On-target baseline calibration could re-enable
  `I2c_*` — see `JETSON_DEPLOYMENT.md`.)
- **Held-out rep-05 (window-level):** detection macro-F1 ≈ 0.95, severity within-1 ≈ 0.73, phase ≈ 0.65;
  **per-recording aggregation** (the live verdict) is markedly stronger on phase — in the loopback test
  all streamed recordings got the correct state, severity (±1 level) and phase.
- Sampling: the bundle is trained at the data's `fs=1000 Hz, f0=60 Hz`; for a different motor/rig fix `fs`
  and retrain (`build_feature_frame` is `(fs,f0)`-parameterised). GPU is auto-detected via `gpu.py`.
