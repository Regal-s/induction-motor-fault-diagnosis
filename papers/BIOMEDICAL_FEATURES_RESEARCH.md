# Research notes — biomedical anomaly features for stator current (deep-research agent, 2026-05-28)

Used to design `features_bio.py`. Novelty: import biomedical signal-complexity / irregularity /
variability features (ECG/EEG/HRV) into stator inter-turn fault diagnosis — the motor-fault literature
uses physics/spectral features only, so a complexity/irregularity view is a genuine gap.

**Key design rule:** compute complexity on the RESIDUAL (current − fitted fundamental) and the
Park-vector modulus / negative-sequence stream, NOT raw current (a clean fundamental is trivially
predictable → flat entropy). Prefer scale-free members (FD, entropies, SD1/SD2 ratio) for load-invariance.

## Prioritized shortlist (cheap, physical, vectorizable; zero new installs for #1–7,9,10)
1. **Cycle-to-cycle waveform distance** (mean+std) — HRV waveform analog; each fundamental cycle = a
   "heartbeat"; fault = cycle-to-cycle irregularity. Most physically apt import. [numpy]
2. **Poincaré SD1, SD2, SD1/SD2** on per-cycle RMS/peak series (HRV time-domain). [numpy]
3. **Hjorth mobility + complexity** (EEG). [antropy.hjorth_params]
4. **Permutation entropy** m=4 (EEG; Bandt & Pompe 2002). [antropy.perm_entropy]
5. **Spectral entropy** (Welch, normalized; EEG anesthesia). [antropy.spectral_entropy]
6. **Higuchi fractal dimension** kmax=10 sim/6 real. [antropy.higuchi_fd]
7. **Dispersion entropy** m=3,c=6 (Rostaghi & Azami 2016; cheap SampEn alternative). [EntropyHub.DispEn]
8. **DTW distance to healthy template cycle** (elastic, slip-tolerant). [needs dtaidistance] — 2nd wave
9. **Katz + Petrosian FD**. [antropy]
10. **Sample entropy** m=2, r=0.15·std (Richman & Moorman 2000). [antropy.sample_entropy] (O(N²), numba ok)

Streams: per-phase residual (aggregated), Park-vector modulus.
**Too heavy / 2nd wave:** RQA (O(N²) memory — decimate first), EEMD/VMD (hours), full MSE (use multiscale
dispersion entropy), DFA (statistically marginal on 500-sample real windows).

Refs: Richman & Moorman (Am J Physiol 2000); Bandt & Pompe (PRL 2002); Costa/Goldberger/Peng (PRL 2002);
Rostaghi & Azami (IEEE SPL 2016); Hjorth (1970); Higuchi (Physica D 1988); Katz (1988); Peng (Chaos 1995);
Task Force HRV (Circulation 1996); Brennan SD1/SD2 (IEEE TBME 2001); Marwan RQA (Phys Rep 2007).
