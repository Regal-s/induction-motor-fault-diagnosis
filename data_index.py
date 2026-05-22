r"""
Phase 1 / M1 — Build the case manifest for the TT + Healthy dataset.

Walks `Healthy case data` and `TT fault Data`, enumerates every simulation case
(one per *_03.out trio), parses its labels and event/region boundaries, and writes
`manifest.csv`. Anti-leakage CV grouping is encoded in `group_id` (operating point
WITHOUT inception, so regular + all inception variants of the same point stay together).

Run:  python data_index.py
Out:  D:\Naveen\manifest.csv  (+ printed summary & sanity checks)

Event timeline (verified, see DATASET_KNOWLEDGE.md), sample indices into .out rows:
  motor start = 2.0 s, load change = 4.0 s (none for NL), fault = inception (7.2 s regular),
  fault clears ~0.8 s later. Healthy files end at 5 s, fault files at 10 s.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\Naveen")
HEALTHY_DIR = ROOT / "Healthy case data"
TT_DIR = ROOT / "TT fault Data"
EXTRA_TAG = "Extra TT Fault Cases"

# event times (seconds)
T_START = 2.0
T_LOAD = 4.0
T_FAULT_REGULAR = 7.2
FAULT_DURATION = 0.8
MARGIN_CYCLES = 1  # trim this many cycles around region edges

SEV_COMP_RE = re.compile(r"^([\d.]+)%$")                 # "0.3%", "5%"
PHASE_COMP_RE = re.compile(r"^Phase_([abcABC])_")        # "Phase_a_5%"
HEALTHY_LOAD_RE = re.compile(r"^(\d+)%\s*Loading$", re.I)  # "5% Loading"
REG_LEAF_RE = re.compile(r"^[\d.]+%_(NL|\d+)%$", re.I)    # "5%_20%", "0.3%_NL%"
# extra leaf: "0.3%_NL_fault at 1_7.202"  /  "1%_40% fault at 10_7.22"
EXTRA_LEAF_RE = re.compile(
    r"^[\d.]+%[\s_]+(?P<load>NL|\d+)%?[\s_]*fault\s*at[\s_]*(?P<k>\d+)[\s_](?P<inc>[\d.]+)$",
    re.I,
)
INFX_RATE_RE = re.compile(r'rate="([\d.]+)"')
INFX_END_RE = re.compile(r'end="(\d+)"')


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def read_infx(infx: Path):
    """Return (fs, n_intervals) from the small .infx XML, else (None, None)."""
    try:
        txt = infx.read_text(errors="ignore")
    except OSError:
        return None, None
    fs = INFX_RATE_RE.search(txt)
    end = INFX_END_RE.search(txt)
    return (float(fs.group(1)) if fs else None,
            int(end.group(1)) if end else None)


def find_component(parts, regex):
    for p in parts:
        m = regex.match(p)
        if m:
            return m
    return None


def regions(state, fs, n_rows, load_label, inception_s):
    """Compute [healthy_region] and [faulty_region] sample bounds (inclusive start,
    exclusive end) into the .out rows. Returns dict of bounds (None where N/A)."""
    margin = int(MARGIN_CYCLES * fs / 50)  # 1 cycle = fs/50 samples
    start = round(T_START * fs)
    is_nl = str(load_label).upper() == "NL" or str(load_label) == "0"
    load_change = None if is_nl else round(T_LOAD * fs)

    if state == "faulty":
        f_start = round(inception_s * fs)
        f_end = min(f_start + round(FAULT_DURATION * fs), n_rows)
        # pre-fault steady healthy region (matched load), well clear of load-change & fault
        h_start = (load_change + 10000) if load_change else (start + 17000)
        h_start = max(h_start, 110000) if not is_nl else max(h_start, 80000)
        h_end = f_start - margin
        return dict(motor_start=start, load_change=load_change,
                    fault_start=f_start, fault_end=f_end,
                    hreg_start=h_start, hreg_end=h_end,
                    freg_start=f_start + margin, freg_end=f_end - margin)
    else:  # healthy
        h_start = (load_change + 1000) if load_change else (start + 17000)
        return dict(motor_start=start, load_change=load_change,
                    fault_start=None, fault_end=None,
                    hreg_start=h_start, hreg_end=n_rows,
                    freg_start=None, freg_end=None)


def parse_case(out03: Path) -> dict | None:
    leaf = out03.parent
    base = out03.name[: -len("_03.out")]
    # .inf/.infx may have a different base name than the .out files -> glob the leaf
    infx = next(iter(sorted(leaf.glob("*.infx"))), None)
    inf = next(iter(sorted(leaf.glob("*.inf"))), None)
    fs, n_int = read_infx(infx) if infx else (None, None)
    if fs is None or n_int is None:
        fs, n_int = 25000.0, (250000 if TT_DIR.name in str(leaf) else 125000)
    n_rows = n_int + 1  # .out includes the t=0 row

    rel = out03.relative_to(ROOT)
    parts = rel.parts
    row = dict(
        case_id=slug(str(leaf.relative_to(ROOT))),
        rel_dir=str(leaf.relative_to(ROOT)),
        path_03=str(out03),
        path_02=str(leaf / f"{base}_02.out"),
        path_inf=str(inf) if inf else "",
        path_infx=str(infx) if infx else "",
        fs=fs, n_intervals=n_int, n_rows=n_rows,
        inception_s=None, variant=None,
    )

    if parts[0] == HEALTHY_DIR.name:
        m = HEALTHY_LOAD_RE.match(leaf.name)
        if not m:
            return {**row, "parse_error": f"healthy load: {leaf.name}"}
        pct = int(m.group(1))
        row.update(state="healthy", severity_pct=0.0, phase=None,
                   load_pct=pct, load_label=("NL" if pct == 0 else str(pct)),
                   is_extra=False)
    elif parts[0] == TT_DIR.name:
        is_extra = EXTRA_TAG in parts
        sev_m = find_component(parts, SEV_COMP_RE)
        ph_m = find_component(parts, PHASE_COMP_RE)
        if not sev_m or not ph_m:
            return {**row, "parse_error": f"TT sev/phase: {rel}"}
        sev = float(sev_m.group(1))
        phase = ph_m.group(1).upper()
        if is_extra:
            lm = EXTRA_LEAF_RE.match(leaf.name)
            if not lm:
                return {**row, "parse_error": f"extra leaf: {leaf.name}"}
            load = lm.group("load").upper()
            row["inception_s"] = float(lm.group("inc"))
            row["variant"] = int(lm.group("k"))
        else:
            lm = REG_LEAF_RE.match(leaf.name)
            if not lm:
                return {**row, "parse_error": f"reg leaf: {leaf.name}"}
            load = lm.group(1).upper()
            row["inception_s"] = T_FAULT_REGULAR
        load_label = "NL" if load == "NL" else load
        row.update(state="faulty", severity_pct=sev, phase=phase,
                   load_pct=(0 if load_label == "NL" else int(load_label)),
                   load_label=load_label, is_extra=is_extra)
    else:
        return {**row, "parse_error": f"unknown root: {parts[0]}"}

    # group_id = operating point WITHOUT inception (anti-leakage CV grouping)
    if row["state"] == "healthy":
        row["group_id"] = f"HLT_l{row['load_label']}"
    else:
        row["group_id"] = (f"TT_s{row['severity_pct']}_p{row['phase']}"
                           f"_l{row['load_label']}")
    row.update(regions(row["state"], fs, n_rows, row["load_label"], row["inception_s"]))
    row["parse_error"] = None
    return row


def build():
    out03s = sorted(list(HEALTHY_DIR.rglob("*_03.out")) + list(TT_DIR.rglob("*_03.out")))
    # exclude the Extra-cases path from accidental double counting (already under TT_DIR)
    rows = [parse_case(p) for p in out03s]
    df = pd.DataFrame(rows)

    errs = df[df["parse_error"].notna()]
    if len(errs):
        print(f"!! {len(errs)} parse errors:")
        for _, r in errs.iterrows():
            print("   ", r["parse_error"])
    df = df[df["parse_error"].isna()].drop(columns=["parse_error"]).reset_index(drop=True)

    assert df["case_id"].is_unique, "duplicate case_id!"
    out = ROOT / "manifest.csv"
    df.to_csv(out, index=False)

    # ---- summary ----
    print(f"\nWrote {out}  ({len(df)} cases)")
    print("\nby state:\n", df["state"].value_counts().to_string())
    print("\nfaulty by severity:\n",
          df[df.state == "faulty"]["severity_pct"].value_counts().sort_index().to_string())
    print("\nfaulty by phase:\n",
          df[df.state == "faulty"]["phase"].value_counts().to_string())
    print("\nfaulty by load:\n",
          df[df.state == "faulty"]["load_label"].value_counts().to_string())
    print("\nfaulty regular vs extra:\n",
          df[df.state == "faulty"]["is_extra"].value_counts().to_string())
    print(f"\nunique group_id: {df['group_id'].nunique()}  "
          f"(healthy {df[df.state=='healthy']['group_id'].nunique()}, "
          f"faulty {df[df.state=='faulty']['group_id'].nunique()})")
    print("group sizes (top):\n",
          df["group_id"].value_counts().head(6).to_string())
    print("\nn_rows unique values:\n", df["n_rows"].value_counts().to_string())
    print("\nsample rows:")
    cols = ["case_id", "state", "severity_pct", "phase", "load_label",
            "inception_s", "variant", "hreg_start", "hreg_end", "freg_start", "freg_end"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].groupby("state").head(3).to_string(index=False))
    return df


if __name__ == "__main__":
    build()
