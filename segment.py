r"""
Phase 1 / M2 — Segment cases into labelled, normalised current windows.

Reads `manifest.csv`, and for each case reads ONLY the needed regions of `_03.out`
(3-phase current in kA, cols 2,3,4), windows them, normalises by a per-recording
scale (RMS over the case's healthy/steady region across all 3 phases — removes load
amplitude, preserves inter-phase imbalance), and stores EVERY window WITH its full
label set.

Outputs (in D:\Naveen\dataset\):
  windows.npy         float32 array, shape (N, 3, WINDOW)  [phases x time]
  window_labels.csv   N rows aligned to windows.npy, fully labelled

Label columns: window_id, case_id, group_id, state, label (0=healthy,1=faulty),
  severity_pct, phase, load_label, load_pct, inception_s, variant, region,
  src_start (sample offset in the .out), scale (per-recording norm factor).

Run:  python segment.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Naveen")
OUTDIR = ROOT / "dataset"
MANIFEST = ROOT / "manifest.csv"

WINDOW = 2000          # 4 cycles @ 25 kHz
STEP_FAULTY = 500      # 75% overlap in the short 0.8 s fault region
STEP_HEALTHY = 4000    # sparser sampling of the long pre-fault / steady region
CUR_COLS = [1, 2, 3]   # _03.out: col0=time, cols 1,2,3 = Ia_ABCIM:1/2/3 (kA)


def read_region(path: str, start, end) -> np.ndarray:
    """Read .out data rows [start, end) -> (n, 3) float32. Row 0 of data is the
    t=0 sample; the file's first physical line is blank (skip it)."""
    start, end = int(start), int(end)
    n = end - start
    if n <= 0:
        return np.empty((0, 3), np.float32)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c",
                     skiprows=1 + start, nrows=n, usecols=CUR_COLS)
    return df.to_numpy(dtype=np.float32)


def windowize(arr: np.ndarray, win: int, step: int):
    """(T,3) -> list of (3,win) windows + their start offsets."""
    T = arr.shape[0]
    starts = list(range(0, T - win + 1, step))
    return [arr[s:s + win].T.copy() for s in starts], starts


def main():
    OUTDIR.mkdir(exist_ok=True)
    man = pd.read_csv(MANIFEST)

    all_windows, rows = [], []
    wid = 0
    for i, c in man.iterrows():
        # --- healthy/steady region: also defines the per-recording scale ---
        H = read_region(c.path_03, c.hreg_start, c.hreg_end)
        scale = float(np.sqrt(np.mean(H ** 2)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        hwins, hstarts = windowize(H, WINDOW, STEP_HEALTHY)
        for w, s in zip(hwins, hstarts):
            all_windows.append(w / scale)
            rows.append(dict(
                window_id=wid, case_id=c.case_id, group_id=c.group_id,
                state="healthy", label=0,
                severity_pct=0.0, phase="", load_label=c.load_label,
                load_pct=c.load_pct, inception_s=c.inception_s, variant=c.variant,
                region="prefault" if c.state == "faulty" else "steady",
                src_start=int(c.hreg_start) + s, scale=scale))
            wid += 1

        # --- fault region (faulty cases only) ---
        if c.state == "faulty":
            F = read_region(c.path_03, c.freg_start, c.freg_end)
            fwins, fstarts = windowize(F, WINDOW, STEP_FAULTY)
            for w, s in zip(fwins, fstarts):
                all_windows.append(w / scale)
                rows.append(dict(
                    window_id=wid, case_id=c.case_id, group_id=c.group_id,
                    state="faulty", label=1,
                    severity_pct=c.severity_pct, phase=c.phase, load_label=c.load_label,
                    load_pct=c.load_pct, inception_s=c.inception_s, variant=c.variant,
                    region="fault",
                    src_start=int(c.freg_start) + s, scale=scale))
                wid += 1
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(man)} cases, {wid} windows so far")

    X = np.stack(all_windows).astype(np.float32)   # (N, 3, WINDOW)
    labels = pd.DataFrame(rows)
    np.save(OUTDIR / "windows.npy", X)
    labels.to_csv(OUTDIR / "window_labels.csv", index=False)

    print(f"\nSaved {OUTDIR / 'windows.npy'}  shape={X.shape}  "
          f"({X.nbytes / 1e6:.0f} MB)")
    print(f"Saved {OUTDIR / 'window_labels.csv'}  ({len(labels)} rows)")
    print("\nby label (0=healthy,1=faulty):\n", labels["label"].value_counts().to_string())
    print("\nhealthy windows by region:\n", labels[labels.label == 0]["region"].value_counts().to_string())
    print("\nfaulty windows by severity:\n",
          labels[labels.label == 1]["severity_pct"].value_counts().sort_index().to_string())
    print("\nfaulty windows by phase:\n", labels[labels.label == 1]["phase"].value_counts().to_string())
    print("\nwindows per group (describe):\n",
          labels.groupby("group_id").size().describe().to_string())
    # sanity: no NaN/Inf in stored windows
    print(f"\nfinite windows: {np.isfinite(X).all()}  | min={X.min():.3f} max={X.max():.3f}")


if __name__ == "__main__":
    main()
