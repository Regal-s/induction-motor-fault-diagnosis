# All-model metrics (StratifiedGroupKFold OOF): Precision / Recall / F1 / Accuracy

## Detection

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| LogReg | 0.973 | 0.984 | 0.978 | 0.980 |
| SVM-RBF | 0.973 | 0.984 | 0.978 | 0.980 |
| KNN | 0.972 | 0.983 | 0.977 | 0.979 |
| MLP | 0.973 | 0.985 | 0.978 | 0.980 |
| RandomForest | 0.973 | 0.984 | 0.978 | 0.980 |
| XGBoost | 0.972 | 0.982 | 0.976 | 0.978 |
| TabPFN-2.5 | 0.974 | 0.985 | 0.979 | 0.981 |
| xLSTM | — | — | 0.957 | 0.960 |

## Severity

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| LogReg | 0.881 | 0.879 | 0.880 | 0.879 |
| SVM-RBF | 0.878 | 0.874 | 0.875 | 0.874 |
| KNN | 0.205 | 0.178 | 0.190 | 0.178 |
| MLP | 0.912 | 0.911 | 0.911 | 0.911 |
| RandomForest | 0.937 | 0.929 | 0.927 | 0.929 |
| XGBoost | 0.833 | 0.830 | 0.829 | 0.830 |
| TabPFN-2.5 | 0.982 | 0.979 | 0.979 | 0.979 |
| xLSTM | — | — | 0.453 | 0.455 |

## Phase

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| LogReg | 0.971 | 0.970 | 0.970 | 0.970 |
| SVM-RBF | 0.984 | 0.984 | 0.984 | 0.984 |
| KNN | 0.983 | 0.982 | 0.982 | 0.982 |
| MLP | 0.957 | 0.956 | 0.956 | 0.956 |
| RandomForest | 0.967 | 0.965 | 0.965 | 0.965 |
| XGBoost | 0.977 | 0.976 | 0.976 | 0.976 |
| TabPFN-2.5 | 0.970 | 0.968 | 0.968 | 0.968 |
| xLSTM | — | — | 0.921 | 0.920 |

*xLSTM from stored benchmark (macro-F1 / accuracy; per-class P/R not re-computed — deep re-run is hours). Severity scored as 7-level classification here for P/R/F1; the deployed model reports within-1/MAE.*