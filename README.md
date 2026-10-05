# 🍷 Wine Quality Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project to train and compare classification models for predicting wine quality based on physicochemical properties such as acidity, density, and alcohol content using the UCI Wine Quality dataset.

---

## 📌 Project Overview & Objectives

* **Objective:** Predict sensory wine quality scores (scale 3–8) from physicochemical laboratory test results.
* **Tech Stack:** Python, pandas, numpy, scikit-learn (Random Forest, SGD, SVC), seaborn, matplotlib, Jupyter Notebook.
* **Dataset:** [UCI Machine Learning Repository: Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009).

---

## 📂 Project Structure

```plaintext
Wine Quality Prediction/
├── Data/
│   ├── winequality-red.csv          # Red wine physicochemical & sensory dataset (1599 x 12)
│   ├── winequality-white.csv        # White wine physicochemical & sensory dataset (4898 x 12)
│   └── winequality.names            # UCI dataset citation & documentation
├── notebooks/
│   ├── Day1_Data_Inspection_and_Distribution.ipynb  # Executed data inspection & class distribution notebook
│   ├── generate_day1_notebook.py                    # Day 1 reproducible notebook generator
│   ├── Day2_Exploratory_Data_Analysis.ipynb         # Executed EDA, statistical profiling & correlation notebook
│   └── generate_day2_notebook.py                    # Day 2 reproducible notebook generator
├── reports/
│   └── figures/
│       ├── day1_class_distribution.png              # Quality class distribution plot
│       ├── day1_quality_distribution.png            # Annotated frequency bar plot
│       ├── day2_univariate_distributions.png        # 11 physicochemical feature distributions (Hist + KDE)
│       ├── day2_feature_boxplots.png                # Boxplots with Tukey IQR outlier detection
│       ├── day2_features_vs_quality.png             # Bivariate boxplots across quality scores (3–8)
│       ├── day2_correlation_heatmap.png             # Full annotated correlation matrix heatmap
│       └── day2_top_correlations.png                # Ranked feature correlations with wine quality
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── data_loader.py               # Reusable data loading, inspection, and plotting routines
│   └── eda_profiling.py             # Univariate profiling, outlier detection, and correlation visualization
├── download_data.py                 # UCI dataset automated download script
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 📅 Progress Tracker

- [x] **Day 1: Data Ingestion, Structural Inspection & Class Imbalance Analysis**
- [x] **Day 2: Exploratory Data Analysis, Outlier Profiling & Multicollinearity Analysis**
- [ ] **Day 3: Target Engineering, Preprocessing Pipelines & Baseline Classification Models**
- [ ] **Day 4: Hyperparameter Optimization, Cross-Validation & Model Evaluation**
- [ ] **Day 5: Model Interpretation, Feature Importance & Final Documentation**

---

## 📊 Day 1 Summary: Dataset Structure & Integrity

From `Data/winequality-red.csv`:
* **Dimensions:** 1,599 samples × 12 attributes (11 physicochemical inputs + 1 sensory quality rating).
* **Missing Values:** **0** missing values across all features.
* **Data Types:** All physicochemical attributes are `float64`; `quality` is discrete `int64`.

### Target Quality Distribution:

| Quality Score | Sample Count | Percentage (%) | Cumulative (%) | Classification Context |
| :---: | :---: | :---: | :---: | :--- |
| **3** | 10 | 0.63% | 0.63% | Severe outlier / Very poor |
| **4** | 53 | 3.31% | 3.94% | Underrepresented / Low quality |
| **5** | 681 | 42.59% | 46.53% | **Majority class (Average)** |
| **6** | 638 | 39.90% | 86.43% | **Majority class (Average)** |
| **7** | 199 | 12.45% | 98.88% | Above average / Good |
| **8** | 18 | 1.13% | 100.00% | Severe outlier / Premium |

> **Key Observation:** Scores 5 and 6 account for **82.49%** of the dataset, highlighting significant class imbalance.

---

## 🔬 Day 2: Exploratory Data Analysis & Statistical Profiling

### 1. Univariate Statistical Profiling

Descriptive statistics across all 11 physicochemical attributes:

| Feature | Mean | Std | Min | 25% (Q1) | Median | 75% (Q3) | Max | IQR | Skewness | Kurtosis | Distribution Shape |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `fixed acidity` | 8.320 | 1.741 | 4.600 | 7.100 | 7.900 | 9.200 | 15.900 | 2.100 | +0.983 | 0.987 | Moderately right-skewed |
| `volatile acidity` | 0.528 | 0.179 | 0.120 | 0.390 | 0.520 | 0.640 | 1.580 | 0.250 | +0.672 | 1.225 | Moderately right-skewed |
| `citric acid` | 0.271 | 0.195 | 0.000 | 0.090 | 0.260 | 0.420 | 1.000 | 0.330 | +0.318 | -0.788 | Broad / Multi-modal |
| `residual sugar` | 2.539 | 1.410 | 0.900 | 1.900 | 2.200 | 2.600 | 15.500 | 0.700 | **+4.541** | **28.618** | **Heavily right-skewed** |
| `chlorides` | 0.087 | 0.047 | 0.012 | 0.070 | 0.079 | 0.090 | 0.611 | 0.020 | **+5.680** | **41.714** | **Heavily right-skewed** |
| `free sulfur dioxide` | 15.875 | 10.460 | 1.000 | 7.000 | 14.000 | 21.000 | 72.000 | 14.000 | +1.251 | 2.024 | Moderately right-skewed |
| `total sulfur dioxide` | 46.468 | 32.895 | 6.000 | 22.000 | 38.000 | 62.000 | 289.000 | 40.000 | +1.516 | 3.810 | Positively right-skewed |
| `density` | 0.997 | 0.002 | 0.990 | 0.996 | 0.997 | 0.998 | 1.004 | 0.002 | **+0.071** | 0.934 | **Near-normal (Symmetric)** |
| `pH` | 3.311 | 0.154 | 2.740 | 3.210 | 3.310 | 3.400 | 4.010 | 0.190 | **+0.194** | 0.807 | **Near-normal (Gaussian)** |
| `sulphates` | 0.658 | 0.170 | 0.330 | 0.550 | 0.620 | 0.730 | 2.000 | 0.180 | **+2.429** | **11.720** | **Heavily right-skewed** |
| `alcohol` | 10.423 | 1.066 | 8.400 | 9.500 | 10.200 | 11.100 | 14.900 | 1.600 | +0.861 | 0.200 | Moderately right-skewed |

---

### 2. Quantitative Outlier Detection (Tukey's IQR Rule)

Outliers quantified using Tukey's fences ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$):

| Feature | Q1 (25%) | Q3 (75%) | IQR | Lower Fence | Upper Fence | Outlier Count | Outlier (%) | Enological Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `residual sugar` | 1.900 | 2.600 | 0.700 | 0.850 | 3.650 | **155** | **9.69%** | Sweet / off-dry dessert wine styles |
| `chlorides` | 0.070 | 0.090 | 0.020 | 0.041 | 0.119 | **112** | **7.00%** | Higher soil salinity & mineral salt content |
| `sulphates` | 0.550 | 0.730 | 0.180 | 0.280 | 1.000 | **59** | **3.69%** | Heavy potassium sulphate preservation |
| `total sulfur dioxide` | 22.000 | 62.000 | 40.000 | -38.000 | 122.000 | **55** | **3.44%** | Elevated antioxidant dosing |
| `fixed acidity` | 7.100 | 9.200 | 2.100 | 3.950 | 12.350 | **49** | **3.06%** | High natural tartaric acid concentrations |
| `density` | 0.996 | 0.998 | 0.002 | 0.992 | 1.002 | **45** | **2.81%** | Density shifts driven by sugar/alcohol variations |
| `pH` | 3.210 | 3.400 | 0.190 | 2.925 | 3.685 | **35** | **2.19%** | Minor natural biological acid variations |
| `free sulfur dioxide` | 7.000 | 21.000 | 14.000 | -14.000 | 42.000 | **30** | **1.88%** | High active antimicrobial SO2 fraction |
| `volatile acidity` | 0.390 | 0.640 | 0.250 | 0.015 | 1.015 | **19** | **1.19%** | Spoilage batches with excessive acetic acid |
| `alcohol` | 9.500 | 11.100 | 1.600 | 7.100 | 13.500 | **13** | **0.81%** | High-alcohol robust vintages |
| `citric acid` | 0.090 | 0.420 | 0.330 | -0.405 | 0.915 | **1** | **0.06%** | Near zero outliers; naturally bounded |

> **Outlier Treatment Strategy:** These outliers reflect authentic physical and chemical variations across wine production styles, not measurement or collection errors. Tree-based ensembles (Random Forest) handle these natively, while distance-based models (SVC, linear classifiers) will require `RobustScaler` or log transforms.

---

### 3. Correlation & Multicollinearity Analysis

#### Target Correlation Ranking (Relationship with Quality):

| Rank | Feature | Pearson Correlation ($r$) | Spearman Correlation ($\rho$) | Correlation Direction | Key Domain Impact |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `alcohol` | **+0.4762** | **+0.4785** | Strong Positive | Enhances wine body, warmth, mouthfeel, and perceived sweetness |
| 2 | `volatile acidity` | **-0.3906** | **-0.3806** | Strong Negative | Acetic acid creates a pungent vinegar flaw; penalizes sensory score |
| 3 | `sulphates` | **+0.2514** | **+0.3771** | Moderate Positive | Prevents wine oxidation and bacterial spoilage during aging |
| 4 | `citric acid` | **+0.2264** | **+0.2135** | Moderate Positive | Imparts crispness, freshness, and balanced acidity |
| 5 | `total sulfur dioxide` | **-0.1851** | **-0.1967** | Moderate Negative | High total SO2 can cause harsh odors and mask fruit character |
| 6 | `density` | **-0.1749** | **-0.1771** | Moderate Negative | Higher density correlates with lower alcohol content ($r = -0.50$) |
| 7 | `chlorides` | **-0.1289** | **-0.1897** | Weak Negative | Excessive saltiness detracts from wine harmony |
| 8 | `fixed acidity` | **+0.1241** | **+0.1141** | Weak Positive | Provides foundational tartaric acid structure |
| 9 | `pH` | **-0.0577** | **-0.0437** | Negligible Negative | Chemical balance indicator; indirect effect |
| 10 | `free sulfur dioxide` | **-0.0507** | **-0.0569** | Negligible Negative | Hygiene baseline; does not directly distinguish superior vintages |
| 11 | `residual sugar` | **+0.0137** | **+0.0320** | Negligible Positive | Dry red wines show no linear preference for sugar content |

#### Top Multicollinear Feature Pairs:

1. **`fixed acidity` vs. `pH` ($r = -0.683$):** Direct chemical consequence; acid concentration directly depresses pH ($-\\log[H^+]$).
2. **`fixed acidity` vs. `citric acid` ($r = +0.672$):** Citric acid directly contributes to total fixed organic acids.
3. **`fixed acidity` vs. `density` ($r = +0.668$):** Organic acid solids increase specific gravity.
4. **`free sulfur dioxide` vs. `total sulfur dioxide` ($r = +0.668$):** Free SO2 represents the active fraction of total sulfur dioxide.
5. **`citric acid` vs. `volatile acidity` ($r = -0.552$):** Well-regulated, clean fermentations with healthy citric levels resist acetobacter spoilage.
6. **`density` vs. `alcohol` ($r = -0.496$):** Ethanol has lower specific gravity ($\approx 0.789$ g/cm$^3$) than water, decreasing mixture density.

---

### 4. Generated EDA Figures

All high-resolution figures are persisted in `reports/figures/`:
* [day2_univariate_distributions.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day2_univariate_distributions.png) — Histograms with KDE and central tendency indicators.
* [day2_feature_boxplots.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day2_feature_boxplots.png) — Boxplots illustrating quartile spread and IQR outliers.
* [day2_features_vs_quality.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day2_features_vs_quality.png) — Bivariate boxplots across wine quality scores (3–8).
* [day2_correlation_heatmap.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day2_correlation_heatmap.png) — Annotated lower-triangular correlation matrix heatmap.
* [day2_top_correlations.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day2_top_correlations.png) — Ranked horizontal bar chart of feature correlation with quality.

---

## ⚙️ Setup and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data inspection and EDA routines:**
   ```python
   from src.data_loader import load_wine_data
   from src.eda_profiling import (
       compute_univariate_stats,
       detect_outliers_iqr,
       compute_target_correlations
   )

   df = load_wine_data("red")
   print(compute_univariate_stats(df))
   print(detect_outliers_iqr(df))
   print(compute_target_correlations(df))
   ```

3. **Launch Jupyter Notebooks:**
   ```bash
   # Day 1: Data Ingestion & Structure Inspection
   jupyter notebook notebooks/Day1_Data_Inspection_and_Distribution.ipynb

   # Day 2: Exploratory Data Analysis & Physicochemical Profiling
   jupyter notebook notebooks/Day2_Exploratory_Data_Analysis.ipynb
   ```
