# Compute efficiency: CPU vs GPU vs GPU-accelerated

Active device: **cpu**. CPU numbers MEASURED on this box; GPU and GPU-accelerated (FP16/TensorRT) are PROJECTED from documented typical speedups (XGBoost GPU-hist; PyTorch CUDA; TensorRT FP16) until the script is re-run on a CUDA machine.

| Model | CPU train (s) | CPU infer (ms/win) | GPU train (s) | GPU infer (ms/win) | GPU-accel infer (ms/win) |
|---|---|---|---|---|---|
| feature_extract * | — | 0.3388 | — | 0.3388 | 0.3388 |
| XGBoost * | 0.54 | 0.0007 | 0.09 | 0.0004 | 0.0003 |
| 1D-CNN * | 2.98 | 0.004 | 0.12 | 0.0003 | 0.0001 |
| xLSTM * | 94.5 | — | 3.15 | — | — |
| TabPFN-2.5 * | — | 8.595 | — | 0.4298 | 0.2865 |

* GPU / GPU-accelerated columns are projections (this run was on CPU). Re-run with `FAULT_DEVICE=cuda python compute_benchmark.py` on an NVIDIA GPU for measured values.

Projection factors (vs CPU): {"feature_extract": {"gpu_train": 1.0, "gpu_infer": 1.0, "acc_infer": 1.0}, "XGBoost": {"gpu_train": 6.0, "gpu_infer": 2.0, "acc_infer": 2.5}, "TabPFN-2.5": {"gpu_train": 1.0, "gpu_infer": 20.0, "acc_infer": 30.0}, "1D-CNN": {"gpu_train": 25.0, "gpu_infer": 15.0, "acc_infer": 40.0}, "xLSTM": {"gpu_train": 30.0, "gpu_infer": 18.0, "acc_infer": 45.0}}