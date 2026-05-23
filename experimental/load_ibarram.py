r"""
M1 (experimental ingestion) — load the ibarram/ITSC experimental dataset into the SAME
window/label schema as our PSCAD simulation, so both can feed one feature/model interface.

Source: experimental/ITSC/dataset/RAWData_ITSC.mat  (13 classes x 5 reps x 5000x3 @ 1 kHz, 60 Hz, no-load)
  classes: ITSC_{A,B,C}{10,20,30,40} (phase x severity%) + ITSC_HLT (healthy).

Windows the raw 5 s signals (W=500 = 30 cycles at 60 Hz -> 60 Hz lands on an integer FFT bin),
normalises per case by the shared 3-phase RMS (as in the sim pipeline), and writes:
  experimental/ibarram/windows.npy        (N,3,W) float32
  experimental/ibarram/window_labels.csv  N rows, schema-compatible with sim window_labels.csv

Run:  python experimental/load_ibarram.py
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(r"D:\Naveen")
SRC = ROOT / "experimental" / "ITSC" / "dataset" / "RAWData_ITSC.mat"
OUT = ROOT / "experimental" / "ibarram"
FS, F0 = 1000.0, 60.0          # experimental sampling / fundamental
W, STEP = 500, 250             # 30 cycles @ 60 Hz (integer bin), 50% overlap
CLASS_RE = re.compile(r"^ITSC_([ABC])(\d+)$")


def windowize(arr, win, step):
    T = arr.shape[0]
    starts = list(range(0, T - win + 1, step))
    return [arr[s:s + win].T.copy() for s in starts], starts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    itsc = loadmat(str(SRC), simplify_cells=True)["itsc"]
    classes = list(itsc.keys())

    wins, rows, wid = [], [], 0
    for cls in classes:
        m = CLASS_RE.match(cls)
        if m:
            state, label, phase, sev = "faulty", 1, m.group(1), float(m.group(2))
        elif cls == "ITSC_HLT":
            state, label, phase, sev = "healthy", 0, "", 0.0
        else:
            print("skip unknown class", cls); continue
        for rep in itsc[cls].keys():
            sig = np.asarray(itsc[cls][rep], dtype=np.float32)      # (5000,3)
            scale = float(np.sqrt(np.mean(sig ** 2))) or 1.0
            ws, starts = windowize(sig, W, STEP)
            case_id = f"ibarram_{cls}_{rep}"
            for w, s in zip(ws, starts):
                wins.append(w / scale)
                rows.append(dict(
                    window_id=wid, case_id=case_id, group_id=f"ibarram_{cls}",
                    state=state, label=label, severity_pct=sev, phase=phase,
                    load_label="NL", load_pct=0, inception_s=np.nan, variant=np.nan,
                    region=("fault" if label else "steady"), src_start=s, scale=scale,
                    dataset="ibarram", fs=FS, f0=F0))
                wid += 1

    X = np.stack(wins).astype(np.float32)
    lab = pd.DataFrame(rows)
    np.save(OUT / "windows.npy", X)
    lab.to_csv(OUT / "window_labels.csv", index=False)

    print(f"Saved {OUT/'windows.npy'}  shape={X.shape}")
    print(f"Saved {OUT/'window_labels.csv'}  ({len(lab)} rows)")
    print("\nby label:\n", lab["label"].value_counts().to_string())
    print("\nfaulty by phase:\n", lab[lab.label == 1]["phase"].value_counts().to_string())
    print("\nfaulty by severity_pct:\n",
          lab[lab.label == 1]["severity_pct"].value_counts().sort_index().to_string())
    print("\ngroups:", lab["group_id"].nunique(), " windows/case:",
          int(lab.groupby("case_id").size().mean()))
    print(f"finite: {np.isfinite(X).all()}  min={X.min():.2f} max={X.max():.2f}")


if __name__ == "__main__":
    main()
