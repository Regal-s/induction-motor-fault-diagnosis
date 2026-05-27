# Research notes — augmentation & leakage-safe tuning (gathered 2026-05-27 via deep-research agents)

Used to design `tune_augment.py` and `augment.py`. Two agents surveyed the 2018–2026 literature.

## A. Hyperparameter tuning & k-fold on grouped fault-diagnosis data
- **Tune for cross-condition generalization, not in-distribution.** Gulrajani & Lopez-Paz (ICLR 2021,
  DomainBed): the model-selection criterion is part of the algorithm; use a **leave-one-load-out inner
  objective** so selected HPs transfer to unseen loads (our headline). Highest-impact single change.
- **Honest two-level protocol.** LOLO outer = headline (HPs tuned only on training loads within each
  fold); 10-fold StratifiedGroupKFold with **frozen** HPs = in-distribution estimate. State both roles.
- **Nesting bias is real but small** (Cawley & Talbot, JMLR 2010 ≈1–4 pts; Wainer & Cawley, ESWA 2021:
  flat selection usually picks the same model). Full nested-on-LOLO not needed for the headline.
- **`n_folds ≤ n_groups`.** We have 147 operating-point groups, so 10-fold grouped CV is valid.
- **Method:** Optuna TPE (Akiba 2019; Bergstra 2011), 25–100 trials fast models; XGBoost early-stopping
  on an inner held-out **load** (never a random split — window leakage); Hyperband for xLSTM; **no tuning
  for TabPFN-2.5** (pretrained). Pipeline-wrap preprocessing so scalers refit per fold. Report macro-F1 /
  within-1, mean±std.
- **Search spaces (XGBoost):** n_estimators 100–2000 (+ES), max_depth 2–8, lr 0.01–0.3 log, subsample
  0.5–1, colsample 0.4–1, min_child_weight 1–10 log, reg_lambda/alpha 1e-3–10 log, gamma 0–5.

## B. Data augmentation for 3-phase current (cross-load + sim→real)
Governing rule: **never fabricate/destroy inter-phase imbalance** (= the inter-turn signature) and
**never break Kirchhoff** (i_a+i_b+i_c=0). ⇒ all amplitude/scale ops applied **jointly** to the 3 phases.

**Prioritized (train-fold only):**
1. **Symmetrical-component load/slip resynthesis (highest value, zero risk):** keep |I2|/|I1| and ∠I2,
   scale |I1| to span 0.6–1.4× training-load range, slip Δf ±1%, I0=0 (Kirchhoff-exact), reconstruct.
   Targets unseen-load LOLO directly.
2. **Joint global amplitude scaling** α~U[0.8,1.2], same α all phases.
3. **Per-channel SNR/Gaussian jitter** 20–40 dB (sensor noise uncorrelated across phases) + small DC
   offset; pair with a 25→1 kHz decimated copy for sim→real.
4. **Cross-load mixup** (same severity, different load, ≤±1 severity step; shared λ~Beta(0.2,0.2)) —
   ordinal-safe (C-Mixup, Yao NeurIPS 2022).
5. **Joint amplitude-spectrum perturbation, signature bins frozen** (f0/harmonics/(1±2s)f0) — Chen ICCV
   2021; Ali arXiv 2506.08412 (2025) signature-guided.
6. **MixStyle** feature-statistic randomization (deep nets only; Zhou ICLR 2021).
- **Faulted-phase relabeling via phase rotation** (channel a→b→c + ∠I2 ∓120°): exact symmetry, augments
  & balances the phase classifier.
- **EXCLUDE:** per-channel scaling, permutation, rotation/flip, CutMix-1D, phase-spectrum randomization,
  aggressive time-warp. Add a **Kirchhoff residual reject filter** post-augmentation.

Refs: Iwana & Uchida (PLOS ONE 2021); Um et al. (ICMI 2017); Zhang mixup (ICLR 2018); Yao C-Mixup
(NeurIPS 2022); Chen amplitude-phase (ICCV 2021); Zhou MixStyle (ICLR 2021); Ali (arXiv 2506.08412 2025).
