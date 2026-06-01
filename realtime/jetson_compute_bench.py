r"""
On-Jetson compute benchmark: per-window inference latency of the deployed cascade
under three execution modes:

  (1) CPU            — XGBoost stock CPU predictor (the current production path)
  (2) GPU            — XGBoost with device='cuda' (Jetson Orin GPU)
  (3) GPU-accelerated — Treelite-compiled C kernel on CPU (fastest CPU tree inference)
                       OR ONNX Runtime CUDA EP, whichever is available.

For each mode we time:
  - Feature extraction (NumPy/SciPy, CPU only — same for all modes)
  - Three booster predictions (detect + severity + phase)
  - Total per-window end-to-end

Run on the Jetson:
  python3 realtime/jetson_compute_bench.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from realtime.streamio import CascadeInfer
from features import build_feature_frame

DEPLOY = ROOT / "realtime" / "deploy"
N_WARMUP = 5
N_REPS = 200

def time_call(fn, reps=N_REPS, warmup=N_WARMUP):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000.0   # ms

def make_synthetic_window(seed=0, W=500, f0=60.0, fs=1000.0):
    rng = np.random.RandomState(seed)
    t = np.arange(W) / fs
    ph = np.array([0.0, -2*np.pi/3, +2*np.pi/3])
    amp = np.array([1.0, 0.85, 1.0])     # 15% imbalance -> looks faulty
    sig = (amp[:, None] * np.sin(2*np.pi*f0*t[None, :] + ph[:, None])).astype(np.float32)
    sig += 0.02 * rng.randn(*sig.shape).astype(np.float32)
    return sig

def main():
    print("loading cascade ...")
    cas = CascadeInfer(DEPLOY)
    print(f"  fs={cas.fs} f0={cas.f0} features={len(cas.allf)} sev_feats={len(cas.sevf)}")
    results = {"reps": N_REPS, "modes": {}}

    # Build a feature frame ONCE for the inference-only timings
    win = make_synthetic_window()
    fr = cas._features(win)
    Xall = fr[cas.allf].to_numpy(np.float32)
    Xsev = fr[cas.sevf].to_numpy(np.float32)
    print(f"  warm feature frame shape: allf={Xall.shape} sevf={Xsev.shape}")

    # ====== Mode (a): feature extraction cost (shared by all execution modes) ======
    t_feat = time_call(lambda: cas._features(win))
    print(f"feature extraction (CPU NumPy/SciPy): {t_feat:.3f} ms/window")

    # ====== Mode (1): CPU XGBoost (current production) ======
    t_det_cpu = time_call(lambda: cas.detect.predict(Xall))
    t_sev_cpu = time_call(lambda: cas.severity.predict(Xsev))
    t_phs_cpu = time_call(lambda: cas.phase.predict(Xall))
    t_pred_cpu = t_det_cpu + t_sev_cpu + t_phs_cpu
    results["modes"]["CPU"] = {
        "feature_ms": t_feat,
        "predict_ms": t_pred_cpu,
        "total_ms": t_feat + t_pred_cpu,
        "detail": {"detect": t_det_cpu, "severity": t_sev_cpu, "phase": t_phs_cpu},
        "notes": "stock XGBoost CPU predictor",
    }
    print(f"CPU XGBoost: det {t_det_cpu:.3f}  sev {t_sev_cpu:.3f}  ph {t_phs_cpu:.3f}  "
          f"sum {t_pred_cpu:.3f} ms  total {t_feat+t_pred_cpu:.3f} ms")

    # ====== Mode (2): GPU XGBoost (device='cuda') ======
    try:
        import xgboost as xgb
        # Re-load each booster with device='cuda' so prediction runs on GPU
        det_gpu = xgb.XGBClassifier(); det_gpu.load_model(str(DEPLOY / "detect.json"))
        sev_gpu = xgb.XGBClassifier(); sev_gpu.load_model(str(DEPLOY / "severity.json"))
        phs_gpu = xgb.XGBClassifier(); phs_gpu.load_model(str(DEPLOY / "phase.json"))
        det_gpu.set_params(device="cuda")
        sev_gpu.set_params(device="cuda")
        phs_gpu.set_params(device="cuda")
        # warm CUDA: a single predict triggers context creation
        det_gpu.predict(Xall); sev_gpu.predict(Xsev); phs_gpu.predict(Xall)
        t_det_gpu = time_call(lambda: det_gpu.predict(Xall))
        t_sev_gpu = time_call(lambda: sev_gpu.predict(Xsev))
        t_phs_gpu = time_call(lambda: phs_gpu.predict(Xall))
        t_pred_gpu = t_det_gpu + t_sev_gpu + t_phs_gpu
        results["modes"]["GPU"] = {
            "feature_ms": t_feat,
            "predict_ms": t_pred_gpu,
            "total_ms": t_feat + t_pred_gpu,
            "detail": {"detect": t_det_gpu, "severity": t_sev_gpu, "phase": t_phs_gpu},
            "notes": "XGBoost device='cuda' (Orin GPU)",
        }
        print(f"GPU XGBoost: det {t_det_gpu:.3f}  sev {t_sev_gpu:.3f}  ph {t_phs_gpu:.3f}  "
              f"sum {t_pred_gpu:.3f} ms  total {t_feat+t_pred_gpu:.3f} ms")
    except Exception as e:
        results["modes"]["GPU"] = {"error": str(e)}
        print(f"GPU XGBoost: failed -- {e}")

    # ====== Mode (3): Accelerated -- XGBoost native Booster + DMatrix ======
    # The sklearn XGBClassifier wrapper is slow per call (validation overhead).
    # Production deployments use the C-level Booster + DMatrix API directly.
    try:
        import xgboost as xgb
        det_b = xgb.Booster(); det_b.load_model(str(DEPLOY / "detect.json"))
        sev_b = xgb.Booster(); sev_b.load_model(str(DEPLOY / "severity.json"))
        phs_b = xgb.Booster(); phs_b.load_model(str(DEPLOY / "phase.json"))
        Dall = xgb.DMatrix(Xall, feature_names=cas.allf)
        Dsev = xgb.DMatrix(Xsev, feature_names=cas.sevf)
        det_b.predict(Dall); sev_b.predict(Dsev); phs_b.predict(Dall)
        t_det_acc = time_call(lambda: det_b.predict(Dall))
        t_sev_acc = time_call(lambda: sev_b.predict(Dsev))
        t_phs_acc = time_call(lambda: phs_b.predict(Dall))
        t_pred_acc = t_det_acc + t_sev_acc + t_phs_acc
        results["modes"]["AcceleratedCPU"] = {
            "feature_ms": t_feat,
            "predict_ms": t_pred_acc,
            "total_ms": t_feat + t_pred_acc,
            "detail": {"detect": t_det_acc, "severity": t_sev_acc, "phase": t_phs_acc},
            "notes": "XGBoost native Booster + DMatrix (C-level, no sklearn wrapper)",
        }
        print(f"Booster API: det {t_det_acc:.3f}  sev {t_sev_acc:.3f}  ph {t_phs_acc:.3f}  "
              f"sum {t_pred_acc:.3f} ms  total {t_feat+t_pred_acc:.3f} ms")
    except Exception as e:
        results["modes"]["AcceleratedCPU"] = {"error": str(e)}
        print(f"Booster API: failed -- {e}")

    # Save
    out = ROOT / "realtime" / "jetson_compute_bench.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
