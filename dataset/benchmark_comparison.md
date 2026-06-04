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
| **ESN*** | groupkfold | 0.923 | 0.919 |
| **ESN*** | lolo | 0.890 | 0.885 |
| **NG-RC*** | groupkfold | 0.827 | 0.824 |
| **NG-RC*** | lolo | 0.856 | 0.852 |
| **ROCKET*** | groupkfold | 0.950 | 0.947 |
| **ROCKET*** | lolo | 0.839 | 0.817 |
| **PatchTST*** | groupkfold | 0.901 | 0.896 |
| **PatchTST*** | lolo | 0.839 | 0.824 |
| **iTransformer*** | groupkfold | 0.876 | 0.872 |
| **iTransformer*** | lolo | 0.846 | 0.831 |
| **TimesNet*** | groupkfold | 0.946 | 0.941 |
| **TimesNet*** | lolo | 0.883 | 0.868 |
| **TTM*** | groupkfold | 0.806 | 0.803 |
| **TTM*** | lolo | 0.662 | 0.618 |
| **Chronos-Bolt*** | groupkfold | 0.754 | 0.753 |
| **Chronos-Bolt*** | lolo | 0.583 | 0.545 |

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
| **ESN*** | groupkfold | 0.179 | 0.177 | 0.896 | 0.941 | 0.839 |
| **ESN*** | lolo | 0.232 | 0.240 | 0.854 | 0.934 | 0.814 |
| **NG-RC*** | groupkfold | 0.143 | 0.098 | 0.682 | 1.253 | 0.759 |
| **NG-RC*** | lolo | 0.220 | 0.217 | 0.652 | 1.228 | 0.729 |
| **ROCKET*** | groupkfold | 0.239 | 0.258 | 0.917 | 0.905 | 0.811 |
| **ROCKET*** | lolo | 0.200 | 0.212 | 0.856 | 1.058 | 0.728 |
| **PatchTST*** | groupkfold | 0.179 | 0.110 | 0.461 | 1.750 | 0.518 |
| **PatchTST*** | lolo | 0.226 | 0.199 | 0.560 | 1.497 | 0.578 |
| **iTransformer*** | groupkfold | 0.140 | 0.131 | 0.628 | 1.438 | 0.632 |
| **iTransformer*** | lolo | 0.179 | 0.176 | 0.470 | 1.860 | 0.281 |
| **TimesNet*** | groupkfold | 0.163 | 0.144 | 0.663 | 1.242 | 0.752 |
| **TimesNet*** | lolo | 0.193 | 0.185 | 0.541 | 1.483 | 0.545 |
| **TTM*** | groupkfold | 0.170 | 0.161 | 0.540 | 1.700 | 0.397 |
| **TTM*** | lolo | 0.104 | 0.095 | 0.510 | 1.927 | 0.224 |
| **Chronos-Bolt*** | groupkfold | 0.124 | 0.122 | 0.452 | 2.009 | 0.216 |
| **Chronos-Bolt*** | lolo | 0.104 | 0.100 | 0.372 | 2.395 | -0.061 |

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
| **ESN*** | groupkfold | 0.905 | 0.905 |
| **ESN*** | lolo | 0.931 | 0.932 |
| **NG-RC*** | groupkfold | 0.934 | 0.934 |
| **NG-RC*** | lolo | 0.926 | 0.926 |
| **PatchTST*** | groupkfold | 0.949 | 0.949 |
| **PatchTST*** | lolo | 0.967 | 0.967 |
| **iTransformer*** | groupkfold | 0.973 | 0.973 |
| **iTransformer*** | lolo | 0.979 | 0.979 |
| **TimesNet*** | groupkfold | 0.919 | 0.920 |
| **TimesNet*** | lolo | 0.932 | 0.932 |
| **TTM*** | groupkfold | 0.624 | 0.624 |
| **TTM*** | lolo | 0.637 | 0.637 |
| **Chronos-Bolt*** | groupkfold | 0.610 | 0.611 |
| **Chronos-Bolt*** | lolo | 0.607 | 0.607 |

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
