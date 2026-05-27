# Biomedical-feature ablation (XGBoost, leakage-safe)

Physics (existing) vs physics + biomedical anomaly features, identical splits.

## Detection (macro-F1)

| Protocol | Physics | Physics+Bio | Δ |
|---|---|---|---|
| groupkfold | 0.977 | 0.977 | -0.000 |
| lolo | 0.948 | 0.978 | +0.030 |

Top biomedical features by gain: pmod_hjorth_mob, pmod_hjorth_comp, pmod_katz, res_petrosian, res_hjorth_comp, pmod_sampen
Biomedical features in overall top-20: pmod_hjorth_mob, pmod_hjorth_comp, pmod_katz, res_petrosian

## Severity (within1)

| Protocol | Physics | Physics+Bio | Δ |
|---|---|---|---|
| groupkfold | 0.997 | 0.997 | +0.000 |
| lolo | 0.942 | 0.979 | +0.037 |

Top biomedical features by gain: res_higuchi, res_perm_ent, res_petrosian, pmod_hjorth_mob, pmod_dispen, res_hjorth_comp
Biomedical features in overall top-20: res_higuchi, res_perm_ent, res_petrosian, pmod_hjorth_mob, pmod_dispen, res_hjorth_comp, pmod_hjorth_comp

## Phase (macro-F1)

| Protocol | Physics | Physics+Bio | Δ |
|---|---|---|---|
| groupkfold | 0.964 | 0.964 | +0.000 |
| lolo | 0.997 | 0.997 | +0.000 |

Top biomedical features by gain: pmod_katz, pmod_hjorth_mob, pmod_dispen, pmod_hrv_sd1, res_petrosian, pmod_sampen
Biomedical features in overall top-20: pmod_katz
