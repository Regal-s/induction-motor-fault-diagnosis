r"""Decisive test of the SCST hypothesis: under a HELD-OUT LOAD (cross-load), does the
load-invariant sequence-component representation generalize better than the raw-phase Stockwell
control? Detection + severity (SCST's claimed strengths). -> dataset/scst/scst_lolo.json"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scst_eval import prep, train_eval  # reuse

DS = Path(r"D:\Naveen") / "dataset" / "scst"
SEV = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]; RANK = {s: i for i, s in enumerate(SEV)}
HOLDOUT = "20"   # the hard light load


def main():
    scst = np.load(DS / "scst_X.npy"); raw = np.load(DS / "rawst_X.npy")
    lab = pd.read_csv(DS / "scst_labels.csv", low_memory=False)
    Xs, Xr = prep(scst), prep(raw)
    load = lab["load_label"].astype(str).to_numpy()
    res = {}

    # detection: train on other loads, test on held-out load
    yd = lab["label"].to_numpy(int)
    te = np.where(load == HOLDOUT)[0]; tr = np.where(load != HOLDOUT)[0]
    res["detection_LOLO20"] = {"SCST": train_eval(Xs, yd, tr, te, "detect"),
                               "raw-phase": train_eval(Xr, yd, tr, te, "detect")}
    # severity (faulty only)
    fmask = (lab["label"] == 1).to_numpy()
    yr = lab["severity_pct"].map(RANK).fillna(-1).to_numpy(int)
    te2 = np.where(fmask & (load == HOLDOUT))[0]; tr2 = np.where(fmask & (load != HOLDOUT))[0]
    res["severity_LOLO20"] = {"SCST": train_eval(Xs, yr, tr2, te2, "severity"),
                              "raw-phase": train_eval(Xr, yr, tr2, te2, "severity")}
    print(json.dumps(res, indent=2))
    json.dump(res, open(DS / "scst_lolo.json", "w"), indent=2)
    print("saved", DS / "scst_lolo.json")


if __name__ == "__main__":
    main()
