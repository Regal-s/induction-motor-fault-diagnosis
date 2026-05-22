# Dataset Knowledge — Induction Motor Stator Turn-to-Turn Fault Data

> Reference notes for `D:\Naveen`. Compiled from inspecting the folder tree, `.inf`/`.infx`
> headers, the `.out` data, and the embedded "Details of the fault cases.txt" notes,
> plus envelope analysis of the current waveforms. Keep this up to date as more is learned.

---

## 1. What this dataset is
- **Machine:** three-phase **induction motor**.
- **Fault studied:** **stator winding turn-to-turn (inter-turn) short circuit**.
- **Purpose:** fault detection / classification / severity estimation (healthy vs. inter-turn,
  and against high-resistance and line-to-line faults).
- **Source:** **PSCAD / EMTDC** time-domain simulation (output device `EMTDC`, version 2010).
- Each simulation run is exported as a trio of files: `.inf` + `.infx` + `.out`.

## 2. Electrical & sampling parameters
| Parameter | Value |
|---|---|
| Fundamental frequency | **50 Hz** |
| Sampling frequency | **25 kHz** |
| Sample period | 40 µs (1/25000 s) |
| Samples per cycle | **500** (25000 / 50) |
| Cycle period | 20 ms |
| Run length (fault cases) | **10 s = 250,000 samples** (`.out` has 250,001 data rows) |
| Run length (healthy cases) | **5 s = 125,000 samples** (125,001 rows) — *half length, see §7* |

> "25,000 samples = 1 second" is the conversion used throughout (and in the original notes).

## 3. Top-level folder structure
```
D:\Naveen\
├─ Healthy case data\   21 loadings × (1 .inf + 1 .infx + 3 .out) = 105 files
├─ TT fault Data\       336 cases, 1,680 files   <- turn-to-turn faults
├─ HRC fault data\      663 cases, 3,316 files    <- high-resistance faults
└─ LL\                  1 case, 5 files           <- line-to-line fault
```

### Healthy case data
- One folder per **motor loading**: `0% (NL), 5, 10, 15 … 100% Loading` (5% steps, 21 folders).
- File naming: `Normal<load>Plaoding.*` (note the misspelling "laoding"); 0% folder uses `NormalNLlaoding`.

### TT fault Data (the main fault set)
Two groups:
1. **Regular cases** — `<severity>\Phase_<x>_<severity>\<severity>_<load>\`
   - severity (= fraction of shorted turns): `0.3%, 0.5%, 1%, 2%, 3%, 4%, 5%`
   - faulted phase: `a, b, c`
   - load: `20%, 40%, 60%, 80%, 100%, NL`
   - → 7 × 3 × 6 = **126 cases**.
   - File naming: `TTfault_25k_phase_<x>_<sev>_loading<load>_0N.out`.
2. **Extra TT Fault Cases** — same `severity\phase` tree, each leaf split into **10 fault-inception-time
   variants** (`..._fault at 1_7.202`, `_at 2_7.204` … `_at 10_7.22`); trailing number = fault start
   time in seconds. → 7 × 3 × 10 = **210 cases**.
   - Total TT = 126 + 210 = **336**.

### HRC fault data
- Severity = **fault resistance**: `0.5, 0.8, 1, 2, 3, 4, 5, 10 ohm`, plus "Extra cases of HEC fault".
- Contains `Details of the fault cases.txt` (see §6).

### LL
- Single line-to-line fault example: `LLfault25kphaseB5Ploading100.*`.

## 4. File trio format (EMTDC)
- **`.inf`** — channel legend, one line per signal:
  `PGB(n) Output Desc="<name>" Group="Main" Max=2.0 Min=-2.0 Units="..."`
- **`.infx`** — XML metadata: sampling `rate`, `end` (sample count), channel list with `dim`/`unit`.
- **`.out`** — the data. Whitespace-delimited, scientific notation. **First line is blank** (header);
  data follows. **Column 1 = time (s)**, remaining columns = signals.
  Signals are split across the three `.out` files, 10 traces per file:
  - `_01.out` → time + PGB(1..10)
  - `_02.out` → time + PGB(11..20)
  - `_03.out` → time + PGB(21..24)

  Load in Python with `np.loadtxt(path)` (it skips the blank line automatically).

## 5. Channel / column map (24 signals = PGB 1–24)
| PGB | Desc | File | Col | Meaning |
|----:|------|------|----:|---------|
| 1 | Ea | _01 | 2 | Phase-A terminal voltage |
| 2 | Eb | _01 | 3 | Phase-B terminal voltage |
| 3 | Insa_s | _01 | 4 | Insulation / inter-turn signal, phase a |
| 4 | Ec | _01 | 5 | Phase-C terminal voltage |
| 5 | Insb_s | _01 | 6 | Insulation signal, phase b |
| 6 | Insc_s | _01 | 7 | Insulation signal, phase c |
| 7–9 | Ea_ABCIM:1/2/3 | _01 | 8–10 | Machine 3-phase voltages |
| 10 | Eab_s | _01 | 11 | Line voltage ab (sensed) |
| **11** | **Ia_ABCIM_s:1** | **_02** | **2** | **Phase-A stator current** |
| **12** | **Ia_ABCIM_s:2** | **_02** | **3** | **Phase-B stator current** |
| **13** | **Ia_ABCIM_s:3** | **_02** | **4** | **Phase-C stator current** |
| 14 | Eca_s | _02 | 5 | Line voltage ca (sensed) |
| 15 | Ebc_s | _02 | 6 | Line voltage bc (sensed) |
| 16 | Eca | _02 | 7 | Line voltage ca |
| 17–19 | InabcIM:1/2/3 | _02 | 8–10 | Neutral / in-winding currents |
| 20 | Ebc | _02 | 11 | Line voltage bc |
| 21–23 | Ia_ABCIM:1/2/3 | _03 | 2–4 | **3-phase currents in kA** (units="kA") |
| 24 | Eab | _03 | 5 | Line voltage ab |

> **Three-phase current** is available two ways:
> `_02.out` cols 2,3,4 (`Ia_ABCIM_s`, per-unit-style, ±2 range) and
> `_03.out` cols 2,3,4 (`Ia_ABCIM`, in **kA**). Voltages/currents in the unitless channels
> have Max/Min ±2 (normalised).

## 6. Event timeline (sample indices) — KEY
The simulations follow a fixed schedule. Sample indices (0-based on the data rows after the blank line):

| Event | Sample index | Time | Notes |
|---|---:|---:|---|
| Pre-energisation (current = 0) | 0 – 50,000 | 0 – 2.0 s | motor not yet connected |
| **Motor start (energise)** | **50,000** | **2.0 s** | large starting inrush; decays over ~0.5–0.7 s |
| Inrush settled (steady run) | ~67,000 | ~2.7 s | runs near no-load |
| **Load change (load applied)** | **100,000** | **4.0 s** | RMS steps up; loaded level set by the case's load % |
| **Fault inception** | **180,000** | **7.2 s** | **fixed across all fault cases** (user-confirmed) |
| **Fault cleared** | **~200,000** | **~8.0 s** | fault duration ≈ 0.8 s |
| End of record (fault cases) | 250,000 | 10.0 s | |
| End of record (healthy cases) | 125,000 | 5.0 s | no fault, run ends early |

Verified by per-cycle RMS envelope on the 3-phase current:
- Motor start = sample **50,000** in every case checked (healthy and fault).
- Load change ≈ sample **100,000** (gradual ramp, settled by ~4.04 s). **No load change for `NL` (no-load) cases.**
- Fault = sample **180,000 (7.2 s)** in all fault cases (matches the "Extra" folder names starting at 7.202 s
  and the user's statement).
- For the example `TT 5%, phase A, 20% load`: phase-A RMS ≈ 0.047 (pre-load) → 0.076 (loaded) → 0.375
  during fault (≈ 5×) → back to 0.076 after clearing.

> HRC notes (`HRC fault data\Details of the fault cases.txt`) describe the same scheme but with
> **load change at 3 s and HRC fault at 4 s, 0.8 s duration (clears 4.8 s)** — i.e. HRC uses a
> *different* timeline than TT. Use the per-folder timing; do not assume TT timing for HRC.

## 7. Important caveats
- **Healthy files are only 5 s (125,001 samples)**, fault files are 10 s (250,001 samples).
  When comparing, align on time/sample, not on row index, and remember healthy ends before the
  fault window. Healthy still contains motor start (50,000) and load change (100,000).
- **Loads don't match across sets:** Healthy uses 5% steps (0–100%); TT uses NL/20/40/60/80/100%.
  There is no 5% TT case to pair with "Healthy 5%".
- The current in `_02` is normalised (±2), `_03` is in kA — pick one consistently.
- Folder/file names contain typos: "laoding"/"loading", "HEC" vs "HRC".
- `.out` first line is blank — must skip it (np.loadtxt handles this).

## 8. Helper scripts in this folder
- `plot_currents.py` — plots `_02` cols 2,3,4 (3-phase current); healthy-vs-faulty compare,
  or `python plot_currents.py <path_to_02.out>` for a single case.
- `plot_events.py` — annotated phase-A current + RMS envelope marking start/load/fault/clear.
- Outputs: `currents_healthy_vs_faulty.png`, `events_annotated.png`.

## 9. Suggested labels for ML (per case)
`fault_type` (healthy / TT / HRC / LL), `severity` (% turns or ohms), `load` (%), `faulted_phase`
(a/b/c), `fault_inception` (s), plus the window split: pre-fault (50k–180k) vs faulted (180k–200k).
