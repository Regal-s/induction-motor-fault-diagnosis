# Model benchmark — TabPFN-2.5 & xLSTM vs prior models

Identical leakage-safe splits (StratifiedGroupKFold + Leave-One-Load-Out) on the 72 physics features (feature models) / decimated 3-phase windows (xLSTM). Severity = 7-level ordinal classification. New models marked *.

## Detection

| Model | Protocol | acc | macro_f1 |
|---|---|---|---|
| LogReg | groupkfold | 0.980 | 0.978 |
| LogReg | lolo | 0.979 | 0.978 |
| SVM-RBF | groupkfold | 0.980 | 0.978 |
| SVM-RBF | lolo | 0.979 | 0.977 |
| KNN | groupkfold | 0.979 | 0.977 |
| KNN | lolo | 0.979 | 0.977 |
| MLP | groupkfold | 0.980 | 0.978 |
| MLP | lolo | 0.980 | 0.978 |
| RandomForest | groupkfold | 0.980 | 0.978 |
| RandomForest | lolo | 0.980 | 0.979 |
| XGBoost | groupkfold | 0.978 | 0.976 |
| XGBoost | lolo | 0.953 | 0.948 |
| **TabPFN-2.5*** | groupkfold | 0.981 | 0.979 |
| **TabPFN-2.5*** | lolo | 0.980 | 0.979 |
| **xLSTM*** | groupkfold | 0.960 | 0.957 |
| **xLSTM*** | lolo | 0.900 | 0.890 |

## Severity

| Model | Protocol | acc | macro_f1 | within1 | mae | qwk |
|---|---|---|---|---|---|---|
| LogReg | groupkfold | 0.879 | 0.880 | 0.997 | 0.127 | 0.982 |
| LogReg | lolo | 0.781 | 0.756 | 0.971 | 0.252 | 0.961 |
| SVM-RBF | groupkfold | 0.874 | 0.875 | 0.991 | 0.142 | 0.976 |
| SVM-RBF | lolo | 0.910 | 0.910 | 0.975 | 0.124 | 0.973 |
| KNN | groupkfold | 0.178 | 0.190 | 0.937 | 0.893 | 0.861 |
| KNN | lolo | 0.143 | 0.170 | 0.942 | 0.925 | 0.855 |
| MLP | groupkfold | 0.911 | 0.911 | 0.997 | 0.095 | 0.986 |
| MLP | lolo | 0.915 | 0.915 | 0.982 | 0.112 | 0.976 |
| RandomForest | groupkfold | 0.929 | 0.927 | 0.985 | 0.095 | 0.978 |
| RandomForest | lolo | 0.072 | 0.069 | 0.688 | 1.397 | 0.652 |
| XGBoost | groupkfold | 0.830 | 0.829 | 0.990 | 0.189 | 0.968 |
| XGBoost | lolo | 0.099 | 0.131 | 0.943 | 0.982 | 0.821 |
| **TabPFN-2.5*** | groupkfold | 0.979 | 0.979 | 0.997 | 0.033 | 0.989 |
| **TabPFN-2.5*** | lolo | 0.529 | 0.509 | 0.969 | 0.532 | 0.905 |
| **xLSTM*** | groupkfold | 0.455 | 0.453 | 0.982 | 0.575 | 0.917 |
| **xLSTM*** | lolo | 0.295 | 0.317 | 0.908 | 0.839 | 0.825 |

## Phase

| Model | Protocol | acc | macro_f1 |
|---|---|---|---|
| LogReg | groupkfold | 0.970 | 0.970 |
| LogReg | lolo | 0.919 | 0.920 |
| SVM-RBF | groupkfold | 0.984 | 0.984 |
| SVM-RBF | lolo | 0.967 | 0.967 |
| KNN | groupkfold | 0.982 | 0.982 |
| KNN | lolo | 0.981 | 0.981 |
| MLP | groupkfold | 0.956 | 0.956 |
| MLP | lolo | 0.981 | 0.981 |
| RandomForest | groupkfold | 0.965 | 0.965 |
| RandomForest | lolo | 0.997 | 0.997 |
| XGBoost | groupkfold | 0.976 | 0.976 |
| XGBoost | lolo | 0.997 | 0.997 |
| **TabPFN-2.5*** | groupkfold | 0.968 | 0.968 |
| **TabPFN-2.5*** | lolo | 0.991 | 0.991 |
| **xLSTM*** | groupkfold | 0.920 | 0.921 |
| **xLSTM*** | lolo | 0.961 | 0.961 |

## Previously-reported deep baselines (partial / single-split protocols — NOT identical)

| Model | Task/Protocol | Metric |
|---|---|---|
| ResNet-1D | detection/GK | macro-F1 0.898 |
| ResNet-1D | severity/GK | within-1 1.000 |
| ResNet-1D | phase/GK | macro-F1 0.708 |
| ResNet-1D | detection/LOLO(load20 only) | macro-F1 0.401 |
| PCM-Net | detection/GK | macro-F1 0.898 |
| PCM-Net | phase/GK | macro-F1 0.850 |
| PCM-Net | severity/LOLO | within-1 0.982 |
