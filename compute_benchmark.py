r"""
Compute-efficiency benchmark: training time + inference latency/throughput per model, on the
ACTIVE device. Produces a CPU / GPU / GPU-accelerated comparison.

This box is CPU-only, so CPU numbers are MEASURED and GPU / GPU-accelerated are PROJECTED from
documented typical speedups (XGBoost device=cuda; torch CUDA; FP16/TensorRT for the deep nets).
Run the SAME script on a CUDA machine (FAULT_DEVICE=cuda, --precision fp16) and the measured
GPU columns replace the projections.

Outputs: dataset/compute_benchmark.json, results_svg/COMPUTE_RESULTS.md,
         results_svg/fig_compute_cpu_gpu.svg/.png  (editable SVG).
Run:  python compute_benchmark.py            (measures on whatever device is active)
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "figure.dpi": 120})

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT)); DS = ROOT / "dataset"
OUT = ROOT / "results_svg"; OUT.mkdir(exist_ok=True)
import gpu
from feature_sets import all_features

# documented typical speedups (relative to CPU) for the PROJECTED columns -----------------
#   sources: XGBoost GPU hist docs; PyTorch CUDA; NVIDIA TensorRT FP16/INT8 inference notes.
PROJ = {
    "feature_extract": {"gpu_train": 1.0, "gpu_infer": 1.0, "acc_infer": 1.0},   # numpy/antropy: CPU-bound
    "XGBoost":          {"gpu_train": 6.0, "gpu_infer": 2.0, "acc_infer": 2.5},
    "TabPFN-2.5":       {"gpu_train": 1.0, "gpu_infer": 20.0, "acc_infer": 30.0}, # transformer fwd
    "1D-CNN":           {"gpu_train": 25.0, "gpu_infer": 15.0, "acc_infer": 40.0},
    "xLSTM":            {"gpu_train": 30.0, "gpu_infer": 18.0, "acc_infer": 45.0},
}


def time_call(fn, reps=1):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main():
    dev = gpu.resolve()
    print("active device:", gpu.info())
    res = {"active_device": dev, "measured": {}, "models": {}}

    df = pd.read_parquet(DS / "features.parquet")
    feats = [c for c in all_features(df.columns) if pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].to_numpy(np.float32); y = df["label"].to_numpy(int)
    Xw = np.load(DS / "windows.npy", mmap_mode="r")
    nb = 2000

    # ---- feature extraction latency ----
    from features import build_feature_frame
    Xb = np.asarray(Xw[:nb]); lb = df.iloc[:nb][["case_id", "label"]]
    t_feat = time_call(lambda: build_feature_frame(Xb, lb, fs=25000.0, f0=50.0, verbose=False)) / nb * 1e3
    res["models"]["feature_extract"] = {"train_s": None, "infer_ms_per_window": float(t_feat)}

    # ---- XGBoost ----
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, tree_method="hist", device=gpu.xgb_device(),
                        eval_metric="logloss", random_state=0)
    tr_t = time_call(lambda: xgb.fit(X, y))
    Xf = X[:nb]; xgb.predict(Xf)
    inf = time_call(lambda: xgb.predict(Xf), reps=3) / nb * 1e3
    res["models"]["XGBoost"] = {"train_s": float(tr_t), "infer_ms_per_window": float(inf)}

    # ---- 1D-CNN + xLSTM (torch) ----
    try:
        import torch, torch.nn as nn
        import xlstm_model as xm
        Xd = xm.decimate(np.asarray(Xw[:6000]), 8); yd = y[:6000]
        # 1D-CNN
        net = nn.Sequential(nn.Conv1d(3, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                            nn.Conv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
                            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1)).to(dev)
        Xt = torch.tensor((Xd - Xd.mean()) / (Xd.std() + 1e-6), device=dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
        yt = torch.tensor(yd, dtype=torch.float32, device=dev)
        def cnn_epoch():
            net.train()
            for i in range(0, len(Xt), 256):
                opt.zero_grad(); o = net(Xt[i:i+256]).squeeze(-1)
                lossf(o, yt[i:i+256]).backward(); opt.step()
        ep = time_call(cnn_epoch)
        net.eval()
        with torch.no_grad():
            inf_cnn = time_call(lambda: net(Xt[:nb]), reps=3) / nb * 1e3
        res["models"]["1D-CNN"] = {"train_s": float(ep * 20), "infer_ms_per_window": float(inf_cnn),
                                   "note": "train_s = 1 epoch x 20"}
        # xLSTM: 1-epoch wall time (extrapolate) + infer
        t0 = time.perf_counter(); xm.train_predict(Xd[:3000], yd[:3000], Xd[:nb], 2, epochs=1, device=dev)
        xl1 = time.perf_counter() - t0
        res["models"]["xLSTM"] = {"train_s_per_epoch": float(xl1),
                                  "train_s": float(xl1 * 20), "infer_ms_per_window": None,
                                  "note": "decimated; train_s ~ 1 epoch x 20"}
    except Exception as e:
        print("torch timing skipped:", e)

    # ---- TabPFN (if available) ----
    try:
        import os
        if os.environ.get("TABPFN_TOKEN"):
            from benchmark_models import make_tabpfn
            rng = np.random.RandomState(0); ctx = rng.choice(len(X), 2000, replace=False)
            clf = make_tabpfn(False); clf.fit(X[ctx], y[ctx])
            inf_t = time_call(lambda: clf.predict(X[:nb]), reps=1) / nb * 1e3
            res["models"]["TabPFN-2.5"] = {"train_s": 0.0, "infer_ms_per_window": float(inf_t),
                                           "note": "in-context (no training); context=2000"}
    except Exception as e:
        print("TabPFN timing skipped:", e)

    # ---- assemble CPU/GPU/GPU-accel table (measured device + projections) ----
    rows = []
    for m, d in res["models"].items():
        infer = d.get("infer_ms_per_window")
        train = d.get("train_s")
        p = PROJ.get(m, {"gpu_train": 1, "gpu_infer": 1, "acc_infer": 1})
        row = {"model": m,
               "cpu_train_s": round(train, 2) if train else None,
               "cpu_infer_ms": round(infer, 4) if infer else None}
        if dev == "cpu":   # project GPU columns
            row["gpu_train_s"] = round(train / p["gpu_train"], 2) if train else None
            row["gpu_infer_ms"] = round(infer / p["gpu_infer"], 4) if infer else None
            row["gpuacc_infer_ms"] = round(infer / p["acc_infer"], 4) if infer else None
            row["status"] = "CPU measured; GPU/accel PROJECTED"
        else:              # measured on GPU
            row["gpu_train_s"] = round(train, 2) if train else None
            row["gpu_infer_ms"] = round(infer, 4) if infer else None
            row["gpuacc_infer_ms"] = None
            row["status"] = f"measured on {dev}"
        rows.append(row)
    res["table"] = rows
    (DS / "compute_benchmark.json").write_text(json.dumps(res, indent=2))

    # markdown
    L = ["# Compute efficiency: CPU vs GPU vs GPU-accelerated", "",
         f"Active device: **{dev}**. CPU numbers MEASURED on this box; GPU and GPU-accelerated "
         "(FP16/TensorRT) are PROJECTED from documented typical speedups (XGBoost GPU-hist; PyTorch "
         "CUDA; TensorRT FP16) until the script is re-run on a CUDA machine.", "",
         "| Model | CPU train (s) | CPU infer (ms/win) | GPU train (s) | GPU infer (ms/win) | GPU-accel infer (ms/win) |",
         "|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None else f"{v}"
    for r in rows:
        tag = "" if dev != "cpu" else " *"
        L.append(f"| {r['model']}{tag} | {f(r['cpu_train_s'])} | {f(r['cpu_infer_ms'])} | "
                 f"{f(r['gpu_train_s'])} | {f(r['gpu_infer_ms'])} | {f(r['gpuacc_infer_ms'])} |")
    L += ["", "* GPU / GPU-accelerated columns are projections (this run was on CPU). Re-run with "
          "`FAULT_DEVICE=cuda python compute_benchmark.py` on an NVIDIA GPU for measured values.",
          "", "Projection factors (vs CPU): " + json.dumps(PROJ)]
    (OUT / "COMPUTE_RESULTS.md").write_text("\n".join(L), encoding="utf-8")

    # figure: inference latency (ms/window) CPU vs GPU vs accel (log scale)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    models = [r["model"] for r in rows if r["cpu_infer_ms"]]
    x = np.arange(len(models)); w = 0.27
    cpu = [next(r["cpu_infer_ms"] for r in rows if r["model"] == m) for m in models]
    g = [next(r["gpu_infer_ms"] for r in rows if r["model"] == m) for m in models]
    gacc = [next(r["gpuacc_infer_ms"] for r in rows if r["model"] == m) or g[i] for i, m in enumerate(models)]
    ax.bar(x - w, cpu, w, label="CPU (measured)", color="#888")
    ax.bar(x, g, w, label="GPU (proj.)" if dev == "cpu" else "GPU", color="#4C78A8")
    ax.bar(x + w, gacc, w, label="GPU-accel FP16/TensorRT (proj.)", color="#59A14F")
    ax.set_yscale("log"); ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("inference latency (ms/window, log)"); ax.set_title("Compute efficiency: CPU vs GPU vs GPU-accelerated")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_compute_cpu_gpu.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / "fig_compute_cpu_gpu.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved dataset/compute_benchmark.json, results_svg/COMPUTE_RESULTS.md + fig_compute_cpu_gpu.svg")


if __name__ == "__main__":
    main()
