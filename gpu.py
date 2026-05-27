r"""
GPU/CPU device resolution so the pipeline runs unchanged on an NVIDIA GPU.

Auto-detects CUDA via torch. Override with the FAULT_DEVICE env var (cpu|cuda|auto) or by
passing prefer= explicitly. On this CPU-only box everything resolves to 'cpu' (no behavior
change); on a CUDA machine the deep models, XGBoost and TabPFN move to the GPU automatically.

Usage:
    import gpu
    dev = gpu.resolve()              # 'cuda' or 'cpu'  (for torch .to(dev))
    XGBClassifier(..., tree_method="hist", device=gpu.xgb_device())
    TabPFNClassifier(device=gpu.resolve())
Set FAULT_DEVICE=cuda to force GPU, FAULT_DEVICE=cpu to force CPU.
"""
from __future__ import annotations
import os


def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve(prefer: str | None = None) -> str:
    """Return 'cuda' or 'cpu'. prefer in {cpu,cuda,gpu,auto,None}; FAULT_DEVICE env overrides None."""
    p = (prefer or os.environ.get("FAULT_DEVICE", "auto")).lower()
    if p == "cpu":
        return "cpu"
    if p in ("cuda", "gpu"):
        return "cuda" if _torch_cuda() else "cpu"
    return "cuda" if _torch_cuda() else "cpu"


def xgb_device(prefer: str | None = None) -> str:
    """Device string for XGBoost's `device=` (use with tree_method='hist')."""
    return resolve(prefer)


def info() -> str:
    dev = resolve()
    try:
        import torch
        name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    except Exception:
        name = "CPU"
    return f"device={dev} ({name})"
