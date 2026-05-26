# Significance across models (Demšar protocol)

Friedman omnibus + Holm-corrected pairwise Wilcoxon signed-rank on per-fold (GroupKFold, 5 blocks) and per-load (LOLO, 6 blocks) primary scores (detection/phase = macro-F1, severity = within-1). Lower average rank = better.

## Detection

**groupkfold** (blocks=5): Friedman χ²=7.66, p=0.2644 (n.s.)

| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |
|---|---|---|---|---|
| LogReg | 0.979 | 4.20 | 1.000 | 1.000 |
| SVM-RBF | 0.979 | 4.70 | 1.000 | 0.750 |
| KNN | 0.978 | 5.50 | 1.000 | 0.750 |
| MLP | 0.979 | 3.70 | 1.000 | 1.000 |
| RandomForest | 0.978 | 3.50 | 1.000 | 1.000 |
| XGBoost | 0.977 | 3.70 | — | 1.000 |
| TabPFN-2.5 | 0.980 | 2.70 | 1.000 | — |

**lolo** (blocks=6): Friedman χ²=12.61, p=0.0497 (significant)

| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |
|---|---|---|---|---|
| LogReg | 0.980 | 4.33 | 1.000 | 1.000 |
| SVM-RBF | 0.980 | 5.08 | 1.000 | 1.000 |
| KNN | 0.980 | 5.08 | 1.000 | 1.000 |
| MLP | 0.980 | 3.92 | 1.000 | 1.000 |
| RandomForest | 0.981 | 2.92 | 1.000 | 1.000 |
| XGBoost | 0.939 | 3.75 | — | 1.000 |
| TabPFN-2.5 | 0.981 | 2.92 | 1.000 | — |

## Severity

**groupkfold** (blocks=5): Friedman χ²=21.84, p=0.0013 (significant)

| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |
|---|---|---|---|---|
| LogReg | 0.997 | 2.90 | 0.750 | 1.000 |
| SVM-RBF | 0.991 | 5.00 | 1.000 | 0.375 |
| KNN | 0.936 | 6.80 | 0.375 | 0.375 |
| MLP | 0.997 | 2.20 | 0.625 | 1.000 |
| RandomForest | 0.985 | 4.60 | 1.000 | 0.500 |
| XGBoost | 0.990 | 4.60 | — | 0.500 |
| TabPFN-2.5 | 0.997 | 1.90 | 0.625 | — |

**lolo** (blocks=6): Friedman χ²=22.85, p=0.0008 (significant)

| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |
|---|---|---|---|---|
| LogReg | 0.970 | 2.58 | 0.625 | 0.938 |
| SVM-RBF | 0.975 | 3.33 | 0.625 | 0.938 |
| KNN | 0.943 | 5.00 | 1.000 | 0.875 |
| MLP | 0.982 | 1.92 | 0.625 | 0.625 |
| RandomForest | 0.701 | 6.75 | 0.375 | 0.188 |
| XGBoost | 0.938 | 4.75 | — | 0.938 |
| TabPFN-2.5 | 0.967 | 3.67 | 0.625 | — |

## Phase

**groupkfold** (blocks=5): Friedman χ²=3.35, p=0.7641 (n.s.)

| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |
|---|---|---|---|---|
| LogReg | 0.969 | 4.70 | 1.000 | 1.000 |
| SVM-RBF | 0.984 | 4.60 | 1.000 | 1.000 |
| KNN | 0.982 | 3.70 | 1.000 | 1.000 |
| MLP | 0.956 | 4.50 | 1.000 | 1.000 |
| RandomForest | 0.963 | 3.70 | 1.000 | 1.000 |
| XGBoost | 0.975 | 3.30 | — | 1.000 |
| TabPFN-2.5 | 0.966 | 3.50 | 1.000 | — |

**lolo** (blocks=6): Friedman χ²=6.00, p=0.4232 (n.s.)

| Model | mean | avg rank | p_Holm vs XGBoost | p_Holm vs TabPFN-2.5 |
|---|---|---|---|---|
| LogReg | 0.902 | 4.50 | 1.000 | 1.000 |
| SVM-RBF | 0.966 | 4.33 | 1.000 | 1.000 |
| KNN | 0.980 | 4.00 | 1.000 | 1.000 |
| MLP | 0.980 | 4.17 | 1.000 | 1.000 |
| RandomForest | 0.998 | 3.58 | 1.000 | 1.000 |
| XGBoost | 0.998 | 3.58 | — | 1.000 |
| TabPFN-2.5 | 0.991 | 3.83 | 1.000 | — |
