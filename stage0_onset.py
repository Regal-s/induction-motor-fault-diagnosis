r"""
Phase 4 — Stage-0 fault-onset detection (deployment realism).

On full recordings, compute a per-cycle negative-sequence fault index |I2|/|I1| and detect
the inception instant by a threshold calibrated on a trailing pre-fault reference window.
Reports detection delay vs the true onset (sample 180000 / 7.2 s) and pre-fault false alarms,
broken down by severity. Runs on a sampled subset of faulty cases for speed.

Run:  python stage0_onset.py  -> dataset/stage0_onset.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Naveen"); DS = ROOT / "dataset"
FS, NCYC = 25000.0, 500
A = np.exp(2j * np.pi / 3)
BASIS = np.exp(-2j * np.pi * np.arange(NCYC) / NCYC)   # 50 Hz bin of a 500-sample cycle
TRUE_ONSET = 180000
REF_S, REF_E = 4.5, 6.5      # reference (pre-fault) window for threshold
SCAN_S = 6.6                 # start scanning for onset here
K, CONSEC = 6.0, 3           # threshold = mean + K*std of ref; need CONSEC consecutive crossings


def read_full_current(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", skiprows=1, usecols=[1, 2, 3])
    return df.to_numpy(np.float32)


def fault_index(cur):
    ncyc = len(cur) // NCYC
    c = cur[:ncyc * NCYC].reshape(ncyc, NCYC, 3)
    P = (2.0 / NCYC) * np.tensordot(c, BASIS, axes=([1], [0]))   # (ncyc,3) complex phasors
    I1 = (P[:, 0] + A * P[:, 1] + A**2 * P[:, 2]) / 3.0
    I2 = (P[:, 0] + A**2 * P[:, 1] + A * P[:, 2]) / 3.0
    fidx = np.abs(I2) / (np.abs(I1) + 1e-9)
    tc = (np.arange(ncyc) + 0.5) * NCYC / FS
    return tc, fidx


def detect(tc, fidx):
    ref = fidx[(tc >= REF_S) & (tc <= REF_E)]
    thr = ref.mean() + K * ref.std()
    scan = np.where(tc >= SCAN_S)[0]
    run = 0
    for i in scan:
        if fidx[i] > thr:
            run += 1
            if run >= CONSEC:
                onset_cyc = i - CONSEC + 1
                return int((onset_cyc) * NCYC), thr, ref.mean()
        else:
            run = 0
    return None, thr, ref.mean()


def main():
    man = pd.read_csv(ROOT / "manifest.csv")
    fa = man[man.state == "faulty"]
    # sample ~5 per severity across phases/loads
    rng = np.random.RandomState(0)
    sample = pd.concat([g.sample(min(5, len(g)), random_state=0) for _, g in fa.groupby("severity_pct")])
    rows = []
    for _, c in sample.iterrows():
        cur = read_full_current(c.path_03)
        tc, fidx = fault_index(cur)
        onset, thr, refm = detect(tc, fidx)
        # pre-fault false alarm: any crossing in [SCAN_S, 7.1]s before true onset?
        pre = fidx[(tc >= SCAN_S) & (tc < TRUE_ONSET / FS)]
        fa_flag = bool((pre > thr).sum() >= CONSEC)
        delay_ms = (onset - TRUE_ONSET) / FS * 1000 if onset else None
        rows.append(dict(severity=c.severity_pct, load=c.load_label, phase=c.phase,
                         detected=onset is not None, onset_sample=onset,
                         delay_ms=delay_ms, prefault_false_alarm=fa_flag))
    R = pd.DataFrame(rows)
    det = R[R.detected]
    res = dict(
        n_cases=len(R),
        detection_rate=float(R.detected.mean()),
        prefault_false_alarm_rate=float(R.prefault_false_alarm.mean()),
        median_delay_ms=float(det.delay_ms.median()) if len(det) else None,
        mean_delay_ms=float(det.delay_ms.mean()) if len(det) else None,
        delay_ms_by_severity={str(s): float(g.delay_ms.median())
                              for s, g in det.groupby("severity") if g.delay_ms.notna().any()},
        detection_rate_by_severity={str(s): float(g.detected.mean())
                                    for s, g in R.groupby("severity")},
    )
    print(json.dumps(res, indent=2))
    R.to_csv(DS / "stage0_onset_cases.csv", index=False)
    with open(DS / "stage0_onset.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {DS/'stage0_onset.json'}, {DS/'stage0_onset_cases.csv'}")


if __name__ == "__main__":
    main()
