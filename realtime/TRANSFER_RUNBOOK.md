# Real-time transfer runbook — laptop → Ethernet → Jetson

Step-by-step to perform the live data transfer + on-Jetson diagnosis later. Read top to bottom.
See `realtime/README.md` for the protocol and `JETSON_DEPLOYMENT.md` for the broader edge plan.

```
 LAPTOP (sender)                         JETSON (receiver + inference)
 saved real 3φ current  ──TCP/Ethernet──▶ buffer → window → features →
 sender.py replays @1kHz                  cascade (onset→detect→severity→phase)
                                          → per-recording VERDICT
```

---

## 0. What moves where (one-time)
- The trained model bundle `realtime/deploy/` (detect.json, severity.json, phase.json, meta.json) and the
  code in `realtime/` go **onto the Jetson**.
- The data to replay stays on the **laptop** (the sender reads it). For this repo the replay source is the
  saved real recordings under `experimental/ibarram/` (windows.npy + window_labels.csv), so the laptop
  needs those too.

---

## 1. Physical / network setup (Ethernet)
1. Connect laptop ↔ Jetson with an Ethernet cable (direct, or via a switch/router).
2. Give both a static IP on the same subnet (direct cable example):
   - Jetson:  `sudo ip addr add 192.168.50.1/24 dev eth0`  (or set via nmcli/Network settings)
   - Laptop:  set the Ethernet adapter to `192.168.50.2`, mask `255.255.255.0`
3. Verify connectivity: from the laptop `ping 192.168.50.1`.
4. Open the port on the Jetson firewall if enabled: `sudo ufw allow 9009/tcp` (default port 9009).

---

## 2. Jetson software setup (one-time)
```bash
# JetPack already provides CUDA/cuDNN/TensorRT. Then:
sudo apt-get install -y python3-pip
git clone https://github.com/Regal-s/induction-motor-fault-diagnosis.git
cd induction-motor-fault-diagnosis
pip3 install numpy scipy pandas scikit-learn xgboost antropy EntropyHub
# (torch only needed for the optional deep/SCST track, not the XGBoost cascade)
```
Adjust the hard-coded `ROOT = r"D:\Naveen"` paths in `realtime/*.py` and `gpu.py`/`features.py` to the
Jetson clone path (e.g. set an env var or edit `ROOT`), since they currently point at the Windows dev path.

---

## 3. Start the receiver (on the Jetson)
```bash
FAULT_DEVICE=cuda python3 -u realtime/receiver.py --port 9009      # use cpu if no CUDA
```
Expected startup:
```
loading deployed cascade ...
ready. cascade metrics (held-out rep): {"detection": ... }
listening on 0.0.0.0:9009  (W=500, stride=250)
```
Leave it running (one TCP connection == one recording; it keeps serving).

---

## 4. Stream from the laptop (sender)
```bash
# one recording:
python realtime/sender.py --host 192.168.50.1 --port 9009 --case ibarram_ITSC_B30_Repetition05
# all held-out rep-05 recordings, back to back:
python realtime/sender.py --host 192.168.50.1 --port 9009 --all-rep 05
#   --speed 1   real time (default);  --speed 60  fast demo;  --packet 50  samples/packet
```
The Jetson prints per-window lines and, when each recording ends, a verdict:
```
>>> stream START ibarram_ITSC_B30_Repetition05 (fs=1000.0, n=5000, truth=1/B)
   t=0.50s win#01 faulty sev=40% phase=B (onset_idx=0.030)
   ...
<<< VERDICT ...: {'state':'FAULTY','severity_pct':40.0,'phase':'B',...} | truth=... [OK]
```

---

## 5. Verify it works
- Stream one healthy (`ibarram_ITSC_HLT_Repetition05`) and one faulty (`ibarram_ITSC_A40_Repetition05`)
  recording; confirm the verdicts read HEALTHY and FAULTY respectively, with plausible severity/phase.
- Loopback test (no Jetson needed): run receiver with `--host 127.0.0.1` and sender `--host 127.0.0.1`.

---

## 6. Retrain / re-deploy the model (when data or rig changes)
```bash
python realtime/train_deploy.py     # retrains on experimental/ibarram, writes realtime/deploy/
```
Then copy the new `realtime/deploy/` to the Jetson and restart the receiver.

---

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `ConnectionRefused` | receiver not running / wrong IP / port blocked → check §1.3–1.4 and §3 |
| No output on Jetson | run the receiver with `python3 -u` (unbuffered) as shown |
| `KeyError: feature` | model bundle out of date → re-run `train_deploy.py` and recopy `deploy/` |
| Wrong/odd predictions on a NEW rig | see §8 calibration; also confirm the rig's `fs`/`f0` match `meta.json` |
| Path errors (`D:\Naveen`) | edit `ROOT` in `realtime/*.py` to the Jetson clone path |

---

## 8. Going from "replay saved data" to a LIVE motor (important)
This runbook replays **saved** real recordings (already in the training normalization), so the streamed
window normalization matches the model exactly. On a **live** motor you must add an on-target step:
1. **Per-recording normalization**: compute the normalising scale from a short healthy baseline window of
   the target motor (the pipeline normalises by the recording's 3-phase RMS); apply it to live windows.
2. **Onset threshold**: recalibrate `mean+6σ` of `|I2|/|I1|` on that healthy baseline (stored in `meta.json`).
3. **Few-shot calibration** (optional, recommended): collect a handful of labelled windows on the target
   motor and fine-tune — recovers detection and improves phase (the PADA sim→real result).
4. **Faulted-phase**: trust the per-recording majority-vote phase, not single windows (per-window phase is
   noisy on real hardware).
These are the same points in `JETSON_DEPLOYMENT.md` §5; none require model-architecture changes.
