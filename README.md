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
├── models/
│   ├── champion_rf_pipeline.joblib  # Serialized production Random Forest pipeline
│   └── champion_rf_pipeline.json    # Pipeline architecture, parameters & test benchmarks metadata
├── notebooks/
│   ├── Day1_Data_Inspection_and_Distribution.ipynb  # Executed data inspection & class distribution notebook
│   ├── generate_day1_notebook.py                    # Day 1 reproducible notebook generator
│   ├── Day2_Exploratory_Data_Analysis.ipynb         # Executed EDA, statistical profiling & correlation notebook
│   ├── generate_day2_notebook.py                    # Day 2 reproducible notebook generator
│   ├── Day3_Target_Engineering_and_Baseline_Models.ipynb # Executed target engineering & baseline model notebook
│   ├── generate_day3_notebook.py                    # Day 3 reproducible notebook generator
│   ├── Day4_Hyperparameter_Optimization_and_Evaluation.ipynb # Executed tuning & cross-validation notebook
│   ├── generate_day4_notebook.py                    # Day 4 reproducible notebook generator
│   ├── Day5_Model_Interpretation_and_Final_Documentation.ipynb # Executed interpretability & champion model notebook
│   └── generate_day5_notebook.py                    # Day 5 reproducible notebook generator
├── reports/
│   └── figures/
│       ├── day1_class_distribution.png              # Quality class distribution plot
│       ├── day1_quality_distribution.png            # Annotated frequency bar plot
│       ├── day2_univariate_distributions.png        # 11 physicochemical feature distributions (Hist + KDE)
│       ├── day2_feature_boxplots.png                # Boxplots with Tukey IQR outlier detection
│       ├── day2_features_vs_quality.png             # Bivariate boxplots across quality scores (3–8)
│       ├── day2_correlation_heatmap.png             # Full annotated correlation matrix heatmap
│       ├── day2_top_correlations.png                # Ranked feature correlations with wine quality
│       ├── day3_target_distribution_comparison.png  # Raw vs 3-class vs binary target distributions
│       ├── day3_confusion_matrices.png              # Heatmaps of RF, SGD, and SVC test confusion matrices
│       ├── day3_baseline_model_comparison.png       # Grouped bar chart comparing baseline classifier metrics
│       ├── day3_roc_curves.png                      # Overlaid ROC curves with test AUC benchmarks
│       ├── day4_cv_performance_comparison.png       # 5-Fold Stratified CV metric distributions before vs after tuning
│       ├── day4_hyperparameter_tuning_results.png   # Parameter search response surface heatmaps
│       ├── day4_precision_recall_curves.png         # Precision-Recall curves with PR-AUC benchmarks
│       ├── day4_tuned_confusion_matrices.png        # Heatmaps of tuned classifier test confusion matrices
│       ├── day4_tuned_vs_baseline_comparison.png    # Multi-metric grouped benchmark comparing baseline vs tuned
│       ├── day5_rf_feature_importance.png           # RF MDI (Gini) vs Test Permutation Importance with error bars
│       ├── day5_permutation_importance_comparison.png # Cross-model permutation importance (RF vs SVC vs SGD)
│       ├── day5_linear_coefficients.png             # Directional standardized weights & odds ratios (SGD)
│       ├── day5_partial_dependence_plots.png        # 1D PDP response curves & rug plots for top chemical drivers
│       ├── day5_pdp_2d_interactions.png             # Bivariate 2D contour maps for key enological trade-offs
│       └── day5_champion_model_summary.png          # Executive champion model scorecard & chemical contrast
├── src/
│   ├── __init__.py                  # Package initializer exposing all module routines
│   ├── data_loader.py               # Reusable data loading, inspection, and plotting routines
│   ├── eda_profiling.py             # Univariate profiling, outlier detection, and correlation visualization
│   ├── feature_engineering.py       # Target binning, domain chemical feature synthesis, stratified splitting
│   ├── baseline_models.py           # Classifier pipelines, evaluation diagnostics, and benchmark visualization
│   ├── model_tuning.py              # Hyperparameter grids, Stratified 5-Fold CV, and threshold tuning
│   └── model_interpretation.py      # Feature importance, PDP profiling, enological rules, and model serialization
├── download_data.py                 # UCI dataset automated download script
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 📅 Progress Tracker

- [x] **Day 1: Data Ingestion, Structural Inspection & Class Imbalance Analysis**
- [x] **Day 2: Exploratory Data Analysis, Outlier Profiling & Multicollinearity Analysis**
- [x] **Day 3: Target Engineering, Preprocessing Pipelines & Baseline Classification Models**
- [x] **Day 4: Hyperparameter Optimization, Cross-Validation & Model Evaluation**
- [x] **Day 5: Model Interpretation, Feature Importance & Final Documentation**

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

## 🧪 Day 3: Target Engineering, Preprocessing & Baseline Models

### 1. Target Formulation & Class Imbalance Analysis

Raw sensory scores span 3 to 8, with scores 5 and 6 accounting for **82.49%** of observations. Directly predicting 6 imbalanced ordinal classes causes severe failure on underrepresented classes (score 3 has only 10 samples; score 8 has 18 samples).

| Formulation Scheme | Categories | Class Mapping | Sample Count | Class Ratio (%) | Practical Application & Modeling Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Raw Discrete** | 6 Classes | Scores 3, 4, 5, 6, 7, 8 | 10, 53, 681, 638, 199, 18 | 0.6%, 3.3%, 42.6%, 39.9%, 12.5%, 1.1% | High label noise between 5 and 6; extreme sample sparsity at tails. |
| **3-Class Grouping** | Low / Med / High | Low (≤4), Med (5–6), High (≥7) | 63, 1,319, 217 | 3.94%, 82.49%, 13.57% | 'Low' class remains severely underrepresented (<4%), biasing classifiers. |
| **Binary (Selected)** | Normal vs. Good | **Normal / Low (<7)** (0)<br>**Good / Premium (≥7)** (1) | **1,382** (0)<br>**217** (1) | **86.43%** (0)<br>**13.57%** (1) | **Cortez et al. (2009) Benchmark Standard:** Reflects the industrial bottling decision for premium single-vineyard vs standard commercial wine. |

---

### 2. Domain-Specific Feature Engineering

Based on chemical relationships identified during Day 2 EDA, 5 new interaction features were synthesized:

1. **`total_acidity`** ($g/dm^3$): `fixed acidity` + `volatile acidity` — Quantifies total acidic strength and palate freshness.
2. **`bound_sulfur_dioxide`** ($mg/dm^3$): `total sulfur dioxide` - `free sulfur dioxide` — Measures the preservative fraction chemically bound to wine compounds.
3. **`free_to_total_so2_ratio`**: `free SO2` / `total SO2` — Indicator of molecular antioxidant efficacy against aerobic microbial spoilage.
4. **`alcohol_to_density_ratio`**: `alcohol` / `density` — Captures wine body and mouthfeel balance between light ethanol and dense residual solids.
5. **`sulphate_to_chloride_ratio`**: `sulphates` / `chlorides` — Ratios desirable preservation salts against undesirable mineral salinity.

---

### 3. Stratified Partitioning & Pipeline Architecture

To guarantee the 13.57% minority premium class is equally represented across splits, an 80/20 train/test split was performed using **stratified sampling** (`random_state=42`):

| Partition | Total Samples | Normal Wine (< 7) | Good Wine (≥ 7) | Premium Class Ratio (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 1,279 | 1,105 | 174 | **13.60%** |
| **Test Set** | 320 | 277 | 43 | **13.44%** |
| **Full Dataset** | 1,599 | 1,382 | 217 | **13.57%** |

> **Data Leakage Prevention:** Feature scalers (`StandardScaler`, `RobustScaler`) and classification estimators are strictly encapsulated inside scikit-learn `Pipeline` objects, ensuring test and validation partitions remain completely unexposed during feature normalization.

---

### 4. Baseline Classification Performance Benchmark

All three required classifiers were evaluated on the held-out test set ($N=320$):

| Model Architecture | Preprocessor | Accuracy | Balanced Acc | Weighted F1 | Good Wine Precision | Good Wine Recall | Good Wine F1 | ROC-AUC | Primary Trait |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | `StandardScaler` | **93.44%** | **0.7951** | **0.9293** | **86.67%** | **60.47%** | **0.7123** | **0.9318** | **Top Performer:** High non-linear separation; robust to outliers |
| **Support Vector Classifier (SVC)** | `StandardScaler` | **90.31%** | 0.6690 | 0.8854 | 83.33% | 34.88% | 0.4918 | **0.8698** | High precision (83.3%); conservative decision boundary |
| **SGD Classifier (log_loss)** | `StandardScaler` | 85.00% | 0.5204 | 0.8097 | 27.27% | 6.98% | 0.1111 | 0.7501 | Baseline linear margin classifier; requires loss/weight tuning |

---

### 5. Confusion Matrix & Diagnostic Insights

| Classifier | True Negatives (Normal) | False Positives (Normal as Good) | False Negatives (Good as Normal) | True Positives (Good) | Good Wine Recall (%) | Good Wine Precision (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **273** / 277 (98.6%) | 4 / 277 (1.4%) | 17 / 43 (39.5%) | **26** / 43 | **60.5%** | **86.7%** |
| **SVC** | **274** / 277 (98.9%) | 3 / 277 (1.1%) | 28 / 43 (65.1%) | 15 / 43 | 34.9% | 83.3% |
| **SGD Classifier** | 269 / 277 (97.1%) | 8 / 277 (2.9%) | 40 / 43 (93.0%) | 3 / 43 | 7.0% | 27.3% |

#### Key Diagnostic Findings:
1. **Ensemble Non-Linearity Wins:** Random Forest achieved near-perfect negative class specificity (98.6%) while capturing over 60% of premium wines with **86.7% precision** out-of-the-box.
2. **The Imbalance Challenge in Linear/Margin Models:** While SVC and SGD achieved high raw accuracy (>85%), their default decision boundaries are pulled heavily toward the majority class, leading to high false negatives on premium wines.
3. **Foundation for Day 4:** To unlock higher minority recall without sacrificing precision, Day 4 will focus on `class_weight='balanced'`, cross-validated threshold optimization, and hyperparameter tuning.

---

### 6. Generated Day 3 Figures

All diagnostic figures are saved in `reports/figures/`:
* [day3_target_distribution_comparison.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day3_target_distribution_comparison.png) — Visual comparison of Raw (3–8), 3-Class (Low, Medium, High), and Binary target distributions.
* [day3_confusion_matrices.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day3_confusion_matrices.png) — Annotated test confusion matrix heatmaps for Random Forest, SGD, and SVC.
* [day3_baseline_model_comparison.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day3_baseline_model_comparison.png) — Grouped performance bar chart comparing all 3 classifiers across key metrics.
* [day3_roc_curves.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day3_roc_curves.png) — Overlaid Receiver Operating Characteristic (ROC) curves with computed AUC scores.

---

## 🎯 Day 4: Hyperparameter Optimization, Cross-Validation & Model Evaluation

### 1. Stratified 5-Fold Cross-Validation Framework

To move beyond single train/test split bias and rigorously evaluate generalization variance across folds, all models were evaluated using **Stratified 5-Fold Cross-Validation** (`random_state=42`) on the training partition ($N=1,279$).

#### Cross-Validation Benchmark (Mean ± Std over 5 Stratified Folds):

| Model Architecture | Evaluation Stage | F1 Score (Class 1) | Balanced Accuracy | ROC-AUC | Minority Precision | Minority Recall | Primary Analytical Insight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Baseline (Default) | 0.4846 ± 0.097 | 0.6784 ± 0.048 | 0.8888 ± 0.025 | 64.27% ± 10.9% | 39.03% ± 8.8% | High precision; default trees fail to capture ~61% of minority wines across folds |
| **Random Forest** | **Tuned (`GridSearchCV`)** | **0.6044 ± 0.090** | **0.7833 ± 0.055** | **0.9014 ± 0.017** | 57.14% ± 8.7% | **64.27% ± 9.8%** | **+24.7% F1 lift & +64.7% recall lift**; ensemble stability maintained |
| **SVC** | Baseline (Default) | 0.4053 ± 0.065 | 0.6345 ± 0.028 | 0.8065 ± 0.033 | 68.60% ± 16.2% | 29.26% ± 5.4% | Default margin hyperplanes severely suppress minority positive predictions |
| **SVC** | **Tuned (`GridSearchCV`)** | **0.5845 ± 0.055** | **0.7767 ± 0.043** | **0.8494 ± 0.024** | 54.30% ± 3.5% | **63.76% ± 9.1%** | **+44.2% F1 lift & +117.9% recall lift**; strong non-linear decision boundary |
| **SGD Classifier** | Baseline (Default) | 0.3790 ± 0.240 | 0.6626 ± 0.122 | 0.8255 ± 0.055 | 38.40% ± 21.5% | 38.94% ± 26.6% | High variance across folds; unregularized linear hyperplane |
| **SGD Classifier** | **Tuned (`GridSearchCV`)** | **0.5013 ± 0.031** | **0.7955 ± 0.019** | **0.8746 ± 0.013** | 36.15% ± 3.1% | **82.18% ± 2.8%** | **+32.3% F1 lift & +111.0% recall lift**; stabilized fold-to-fold variance ($\sigma$ down from 0.24 to 0.03) |

---

### 2. Systematic Hyperparameter Optimization (`GridSearchCV`)

Exhaustive grid searches were conducted using 5-fold stratified cross-validation targeting minority-class $F_1$-score maximization:

| Estimator Pipeline | Tuned Hyperparameters | Best Value | Grid Exploration Space | Enological & Algorithmic Rationale |
| :--- | :--- | :---: | :--- | :--- |
| **Random Forest** | `clf__class_weight`<br>`clf__max_depth`<br>`clf__min_samples_split`<br>`clf__n_estimators` | `'balanced'`<br>`18`<br>`2`<br>`200` | `[None, 'balanced', 'balanced_subsample']`<br>`[None, 12, 18]`<br>`[2, 5]`<br>`[100, 200]` | Inverting tree sample weights directly penalizes misclassification of rare premium batches; depth limit of 18 prevents leaf overfitting. |
| **SVC** | `clf__C`<br>`clf__gamma`<br>`clf__class_weight` | `10.0`<br>`0.1`<br>`'balanced'` | `[0.5, 1.0, 3.0, 5.0, 10.0]`<br>`['scale', 0.05, 0.1, 0.2]`<br>`[None, 'balanced']` | Higher $C=10.0$ imposes tighter margin constraints on false negatives; $\gamma=0.1$ fits non-linear chemical interactions accurately. |
| **SGD Classifier** | `clf__loss`<br>`clf__penalty`<br>`clf__alpha`<br>`clf__class_weight` | `'log_loss'`<br>`'l2'`<br>`0.05`<br>`'balanced'` | `['log_loss', 'modified_huber']`<br>`['l2', 'elasticnet']`<br>`[1e-4, 1e-3, 1e-2, 5e-2]`<br>`[None, 'balanced']` | Log-loss logistic margins combined with stronger $L_2$ regularization ($\alpha=0.05$) prevent runaway gradient divergence and stabilize predictions. |

---

### 3. Held-Out Test Set Performance & Metric Lift Analysis ($N=320$)

All tuned pipelines were evaluated on the held-out test partition ($N=320$, 277 Normal vs. 43 Good Wine):

| Model Stage | Model Architecture | Accuracy | Balanced Acc | Good Wine Prec | Good Wine Rec | Good Wine F1 | Weighted F1 | ROC-AUC | Key Performance Trait |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Day 3)** | Random Forest | **93.44%** | 0.7951 | **86.67%** | 60.47% | **0.7123** | **0.9293** | 0.9318 | High precision; misses 39.5% of premium wines |
| **Baseline (Day 3)** | Support Vector Classifier | 90.31% | 0.6690 | 83.33% | 34.88% | 0.4918 | 0.8854 | 0.8698 | Severe conservative bias (misses 65.1% of premium wines) |
| **Baseline (Day 3)** | SGD Classifier | 85.00% | 0.5204 | 27.27% | 6.98% | 0.1111 | 0.8097 | 0.7501 | Near-total failure on minority class (detects only 3/43) |
| **Tuned (Day 4)** | **Random Forest** | **91.87%** | **0.8057** | 71.79% | **65.12%** | **0.6829** | 0.9171 | **0.9368** | **Overall Best Model:** Peak ROC-AUC (0.9368) and PR-AUC (0.8095) |
| **Tuned (Day 4)** | **SVC** | 89.06% | **0.7895** | 58.33% | **65.12%** | **0.6154** | 0.8931 | **0.8979** | **+86.7% Recall Boost:** Balanced hyperplane captures 28 of 43 |
| **Tuned (Day 4)** | **SGD Classifier** | 81.56% | **0.8149** | 40.70% | **81.40%** | **0.5426** | 0.8386 | **0.9039** | **+1066.7% Recall Surge:** Catches 35 of 43 good wines |
| **Tuned + Thresh Opt** | **Random Forest ($\tau=0.35$)** | 89.69% | **0.8520** | 58.62% | **79.07%** | **0.6733** | 0.9031 | **0.9368** | **Industrial Screening Standard:** Catches 34/43 premium wines |

#### Quantitative Optimization Lift ($\Delta$ and % Lift vs. Day 3 Baseline):

* **SGD Classifier:**
  - Good Wine Recall: **6.98% $\to$ 81.40%** (**+74.42% delta**, **+1,066.67% lift**)
  - Good Wine F1: **0.1111 $\to$ 0.5426** (**+0.4315 delta**, **+388.37% lift**)
  - Balanced Accuracy: **0.5204 $\to$ 0.8149** (**+0.2945 delta**, **+56.58% lift**)
  - ROC-AUC: **0.7501 $\to$ 0.9039** (**+0.1538 delta**, **+20.51% lift**)
* **Support Vector Classifier (SVC):**
  - Good Wine Recall: **34.88% $\to$ 65.12%** (**+30.23% delta**, **+86.67% lift**)
  - Good Wine F1: **0.4918 $\to$ 0.6154** (**+0.1236 delta**, **+25.13% lift**)
  - Balanced Accuracy: **0.6690 $\to$ 0.7895** (**+0.1205 delta**, **+18.01% lift**)
  - ROC-AUC: **0.8698 $\to$ 0.8979** (**+0.0281 delta**, **+3.23% lift**)
* **Random Forest:**
  - 5-Fold CV F1: **0.4846 $\to$ 0.6044** (**+0.1198 delta**, **+24.72% lift**)
  - Test Minority Recall: **60.47% $\to$ 65.12%** (Default threshold) $\to$ **79.07%** (Threshold $\tau = 0.35$)
  - Test ROC-AUC: **0.9318 $\to$ 0.9368** (**+0.0050 delta**)

---

### 4. Decision Threshold Calibration & Precision-Recall Dynamics

In heavily imbalanced domains (13.6% premium prevalence), evaluating Precision-Recall curves and Average Precision (PR-AUC) provides a significantly more stringent test than standard ROC curves:

* **Random Forest (Tuned):** PR-AUC = **0.8095** (Baseline random guessing prevalence is 13.4%)
* **SVC (Tuned):** PR-AUC = **0.7105**
* **SGD (Tuned):** PR-AUC = **0.5823**

#### Operating Threshold Trade-Offs (Tuned Random Forest):
1. **Maximum Precision Operating Point ($\tau = 0.59$):**
   - Yields **87.10% Precision**, 62.79% Recall, and peak point $F_1 = 0.7297$.
   - Recommended when commercial blending costs for false positives are severe.
2. **Industrial Screening Operating Point ($\tau = 0.35$):**
   - Yields **79.07% Recall (34/43 caught)**, **0.8520 Balanced Accuracy**, and **58.62% Precision**.
   - Recommended for winery reserve candidate screening to minimize discarding premium batches into bulk wine.

---

### 5. Confusion Matrix Diagnostics (Tuned Test Set)

| Classifier Architecture | True Negatives (Normal) | False Positives (Normal as Good) | False Negatives (Good as Normal) | True Positives (Good) | Good Wine Recall (%) | Good Wine Precision (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **266** / 277 (96.0%) | 11 / 277 (4.0%) | 15 / 43 (34.9%) | **28** / 43 | **65.1%** | **71.8%** |
| **SVC (Tuned)** | **257** / 277 (92.8%) | 20 / 277 (7.2%) | 15 / 43 (34.9%) | **28** / 43 | **65.1%** | **58.3%** |
| **SGD Classifier (Tuned)** | 226 / 277 (81.6%) | 51 / 277 (18.4%) | **8** / 43 (18.6%) | **35** / 43 | **81.4%** | 40.7% |
| **RF ($\tau = 0.35$, Screen)** | 253 / 277 (91.3%) | 24 / 277 (8.7%) | **9** / 43 (20.9%) | **34** / 43 | **79.1%** | 58.6% |

---

### 6. Generated Day 4 Figures

All generated diagnostic figures are saved in `reports/figures/`:
* [day4_cv_performance_comparison.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day4_cv_performance_comparison.png) — 5-Fold Stratified Cross-Validation metric distributions ($F_1$, Balanced Accuracy, ROC-AUC) comparing baseline vs tuned models.
* [day4_hyperparameter_tuning_results.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day4_hyperparameter_tuning_results.png) — Parameter search response surface heatmaps for Random Forest, SVC, and SGDClassifier.
* [day4_precision_recall_curves.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day4_precision_recall_curves.png) — Precision-Recall curves with calculated Average Precision (PR-AUC) scores and operating trade-off points.
* [day4_tuned_confusion_matrices.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day4_tuned_confusion_matrices.png) — Annotated test confusion matrix heatmaps showing the massive decrease in false negatives.
* [day4_tuned_vs_baseline_comparison.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day4_tuned_vs_baseline_comparison.png) — Multi-metric side-by-side grouped benchmark comparing Baseline vs Tuned model performance.

---

## 🔬 Day 5: Model Interpretation, Feature Importance & Final Documentation

### 1. Tree-Based Feature Importance (MDI / Gini Impurity Reduction)

Feature importances were computed across the 200 estimators of the tuned champion Random Forest ensemble. The table below presents the ranked Mean Decrease in Impurity (MDI), standard deviation across trees, and relative percentage contribution across all 16 physicochemical features:

| Rank | Feature | Feature Type | Importance (Mean) | Tree Std ($\sigma$) | Relative Contribution (%) | Enological Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | `alcohol_to_density_ratio` | Engineered | **0.1718** | 0.1165 | **17.18%** | Mouthfeel balance; ethanol vs residual solid extract |
| 2 | `alcohol` | Physicochemical | **0.1375** | 0.1166 | **13.75%** | Primary sensory warmth, aromatic volatility, body |
| 3 | `sulphates` | Physicochemical | **0.0917** | 0.0609 | **9.17%** | Potassium sulphate antioxidant and preservative defense |
| 4 | `volatile acidity` | Physicochemical | **0.0760** | 0.0473 | **7.60%** | Acetic acid spoilage marker; vinegar taint |
| 5 | `sulphate_to_chloride_ratio` | Engineered | **0.0637** | 0.0435 | **6.37%** | Preservative salt balance vs undesirable mineral salinity |
| 6 | `citric acid` | Physicochemical | **0.0614** | 0.0422 | **6.14%** | Palate freshness, crispness, natural acid buffer |
| 7 | `bound_sulfur_dioxide` | Engineered | **0.0529** | 0.0330 | **5.29%** | Chemically bound antioxidant pool |
| 8 | `free_to_total_so2_ratio` | Engineered | **0.0454** | 0.0225 | **4.54%** | Active molecular antimicrobial fraction |
| 9 | `total sulfur dioxide` | Physicochemical | 0.0449 | 0.0305 | 4.49% | Total preservative load; indicator of must stress |
| 10 | `density` | Physicochemical | 0.0436 | 0.0298 | 4.36% | Specific gravity influenced by sugar and alcohol |
| 11 | `total_acidity` | Engineered | 0.0435 | 0.0310 | 4.35% | Sum of fixed tartaric and volatile acids |
| 12 | `fixed acidity` | Physicochemical | 0.0416 | 0.0294 | 4.16% | Foundational tartaric acid structure |
| 13 | `chlorides` | Physicochemical | 0.0402 | 0.0290 | 4.02% | Sodium chloride mineral saltiness |
| 14 | `pH` | Physicochemical | 0.0396 | 0.0270 | 3.96% | Acid-base equilibrium and microbial susceptibility |
| 15 | `residual sugar` | Physicochemical | 0.0395 | 0.0267 | 3.95% | Residual sweetness; dry wine style baseline |
| 16 | `free sulfur dioxide` | Physicochemical | 0.0387 | 0.0245 | 3.87% | Active antimicrobial free preservative gas |

> **Key Observation:** The 5 domain-engineered features account for **39.73%** of total ensemble decision weight, demonstrating substantial value over raw baseline features alone.

---

### 2. Unbiased Test Set Permutation Feature Importance ($N=320$)

Because MDI can exhibit bias towards continuous attributes with many split candidates, **Permutation Feature Importance** was evaluated on the held-out test partition ($N=320$) using 10 shuffle repeats scored on ROC-AUC:

| Rank | Feature | Mean ROC-AUC Drop ($\Delta$) | Shuffle Std ($\sigma$) | Evaluation Context |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `sulphates` | **+0.0344** | 0.0067 | Most critical generalization anchor on unseen wine test samples |
| 2 | `alcohol_to_density_ratio` | **+0.0172** | 0.0115 | Primary generalizable body and dryness discriminator |
| 3 | `sulphate_to_chloride_ratio` | **+0.0109** | 0.0018 | Robust salt harmony discriminator |
| 4 | `volatile acidity` | **+0.0107** | 0.0037 | Critical penalty factor for flawed batches |
| 5 | `total_acidity` | **+0.0072** | 0.0013 | Structural acidic backbone |
| 6 | `fixed acidity` | **+0.0066** | 0.0013 | Natural tartaric acid concentration |
| 7 | `total sulfur dioxide` | **+0.0057** | 0.0043 | General preservative antioxidant ceiling |
| 8 | `free sulfur dioxide` | **+0.0036** | 0.0021 | Active microbial barrier |

---

### 3. Directional Decision Weights: SGD Classifier Linear Coefficients

Standardized linear regression coefficients ($\beta_i$) and odds multipliers ($e^{\beta_i}$) reveal the explicit positive vs. negative push exerted by each chemical component on wine quality:

| Rank | Feature | Standardized Weight ($\beta$) | Odds Ratio ($e^\beta$) | Direction | Enological Rationale |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `alcohol` | **+0.4410** | **1.5543** | **Positive (+)** | +1 std alcohol increases odds of premium rating by **55.4%** |
| 2 | `alcohol_to_density_ratio` | **+0.4379** | **1.5495** | **Positive (+)** | Reflects complete fermentation and low residual heaviness |
| 3 | `sulphate_to_chloride_ratio` | **+0.3050** | **1.3566** | **Positive (+)** | Favorable mineral balance without salty flaw |
| 4 | `volatile acidity` | **-0.3042** | **0.7377** | **Negative (-)** | +1 std acetic acid cuts odds of premium rating by **26.2%** |
| 5 | `sulphates` | **+0.2979** | **1.3470** | **Positive (+)** | Antioxidant stability and phenolic preservation |
| 6 | `free_to_total_so2_ratio` | **+0.2353** | **1.2653** | **Positive (+)** | Ensures active preservative fraction is available |
| 7 | `bound_sulfur_dioxide` | **-0.2022** | **0.8169** | **Negative (-)** | High bound SO2 indicates prior oxidative stress in must |
| 8 | `total sulfur dioxide` | **-0.1934** | **0.8242** | **Negative (-)** | Masks fruit aromatics and creates pungent chemical odor |
| 9 | `residual sugar` | **+0.1604** | 1.1739 | Positive (+) | Trace sugar rounds harsh tannins in robust red wines |
| 10 | `citric acid` | **+0.1319** | 1.1410 | Positive (+) | Imparts crispness and lively sensory brightness |
| 11 | `fixed acidity` | **+0.1307** | 1.1396 | Positive (+) | Foundation for fresh tartaric palate structure |
| 12 | `total_acidity` | **+0.1016** | 1.1070 | Positive (+) | Structural palate firmness |
| 13 | `density` | **-0.0962** | 0.9083 | Negative (-) | Heavy density reflects lower ethanol concentration |
| 14 | `pH` | **-0.0886** | 0.9152 | Negative (-) | Higher pH increases bacterial spoilage risk |
| 15 | `free sulfur dioxide` | **-0.0868** | 0.9169 | Negative (-) | Slight negative when unbound in excess without context |
| 16 | `chlorides` | **-0.0511** | 0.9502 | Negative (-) | Saline mineral salts detract from phenolic harmony |

---

### 4. Non-Linear Partial Dependence Profiles & Tipping Points (1D PDP)

Partial dependence analysis traces non-linear marginal predictions of the champion Random Forest:

1. **`alcohol` (Threshold: $\ge 11.5\%$ vol):**
   - Predicted probability of premium quality remains flat ($\sim 0.10$) below $10.0\%$ vol, surges steeply between $10.5\%$ and $12.0\%$, and reaches maximum plateau above $12.5\%$ vol ($\sim 0.45+$).
2. **`volatile acidity` (Tipping Point: $\le 0.40$ g/dm$^3$):**
   - Shows a steep monotonic drop. Probability falls precipitously once acetic acid exceeds $0.40$ g/dm$^3$, dropping to near zero above $0.65$ g/dm$^3$.
3. **`sulphates` (Optimal Window: $0.70 - 0.90$ g/dm$^3$):**
   - Sharp rise from $0.50$ to $0.75$ g/dm$^3$ where antioxidant efficacy peaks; saturates above $0.95$ g/dm$^3$.
4. **`free_to_total_so2_ratio` (Target: $\ge 0.25$):**
   - Wines with active free SO2 ratio above $25\%$ show a consistent positive shift in predicted premium likelihood.

---

### 5. Actionable Viticultural & Enological Decision Guidelines

By benchmarking the chemical distributions of Normal (<7) vs. Good (≥7) wines, quantitative targets were synthesized for winemakers:

| Physicochemical Attribute | Normal Median | Good Median | Mean Shift (%) | Viticultural & Cellar Guideline |
| :--- | :---: | :---: | :---: | :--- |
| **`citric acid`** | 0.24 g/dm$^3$ | **0.40 g/dm$^3$** | **+48.0%** | Target $\ge 0.40$ g/dm$^3$ to maintain palate freshness and acid balance |
| **`sulphate_to_chloride_ratio`** | 7.56 | **10.20** | **+34.9%** | Target $\ge 10.2$ to optimize preservation salts over mineral salinity |
| **`bound_sulfur_dioxide`** | 23.0 mg/dm$^3$ | **14.0 mg/dm$^3$** | **-34.9%** | Suppress $\le 18.0$ mg/dm$^3$ by preventing early-stage must oxidation |
| **`total sulfur dioxide`** | 39.5 mg/dm$^3$ | **27.0 mg/dm$^3$** | **-27.7%** | Keep $\le 30.0$ mg/dm$^3$ to avoid masking natural fruit aromas |
| **`volatile acidity`** | 0.54 g/dm$^3$ | **0.37 g/dm$^3$** | **-25.9%** | Restrict $\le 0.40$ g/dm$^3$ via strict temperature and acetobacter control |
| **`free_to_total_so2_ratio`** | 0.36 | **0.44** | **+15.9%** | Maintain $\ge 0.25$ active free preservative fraction |
| **`sulphates`** | 0.60 g/dm$^3$ | **0.74 g/dm$^3$** | **+15.3%** | Target $0.70 - 0.85$ g/dm$^3$ for long-term bottle oxidation defense |
| **`alcohol`** | 10.0% vol | **11.5% vol** | **+11.8%** | Harvest at full physiological maturity targeting $\ge 11.5\%$ vol |
| **`chlorides`** | 0.080 g/dm$^3$ | **0.073 g/dm$^3$** | **-15.0%** | Restrict $\le 0.075$ g/dm$^3$ by managing vineyard soil salinity |

---

### 6. Champion Model Packaging & Production Inference API

The finalized champion model is persisted as a production artifact in `models/`:
* **Artifact Path:** [`models/champion_rf_pipeline.joblib`](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/models/champion_rf_pipeline.joblib)
* **Metadata Record:** [`models/champion_rf_pipeline.json`](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/models/champion_rf_pipeline.json)
* **Architecture:** Scikit-learn Pipeline with `StandardScaler` + `RandomForestClassifier` (200 trees, `max_depth=18`, `class_weight='balanced'`)
* **Test Performance:** ROC-AUC = **0.9368**, PR-AUC = **0.8095**, Balanced Accuracy ($\tau=0.35$) = **0.8520**, Minority Recall = **79.07%**

#### Live Inference Demonstration:
```python
from src.model_interpretation import load_champion_model_pipeline, predict_wine_quality_sample

# Load serialized champion pipeline
pipeline = load_champion_model_pipeline("models/champion_rf_pipeline.joblib")

# Example: Premium reserve candidate batch
sample_lab_data = {
    "fixed acidity": 8.5,
    "volatile acidity": 0.32,
    "citric acid": 0.45,
    "residual sugar": 2.1,
    "chlorides": 0.065,
    "free sulfur dioxide": 18.0,
    "total sulfur dioxide": 36.0,
    "density": 0.9955,
    "pH": 3.30,
    "sulphates": 0.82,
    "alcohol": 12.4
}

prediction = predict_wine_quality_sample(pipeline, sample_lab_data, threshold=0.35)
print(prediction["predicted_label"])
# -> "Good / Premium Wine (Score >= 7)"
print(f"Confidence: {prediction['probability_good']:.1%}")
# -> "Confidence: 81.3%"
```

---

### 7. Generated Day 5 Figures

All diagnostic and interpretability figures are saved in `reports/figures/`:
* [day5_rf_feature_importance.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day5_rf_feature_importance.png) — Random Forest MDI (Gini) vs. Test Permutation Importance with tree-level standard error bars.
* [day5_permutation_importance_comparison.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day5_permutation_importance_comparison.png) — Cross-model test set permutation importance comparison across Random Forest, SVC, and SGDClassifier.
* [day5_linear_coefficients.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day5_linear_coefficients.png) — Standardized directional linear weights ($\beta$) and odds ratios indicating positive vs. negative drivers.
* [day5_partial_dependence_plots.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day5_partial_dependence_plots.png) — 6-panel 1D Partial Dependence Profiles with empirical sample rug distributions.
* [day5_pdp_2d_interactions.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day5_pdp_2d_interactions.png) — Bivariate 2D contour interaction maps (Alcohol vs. Volatile Acidity and Sulphates vs. Chlorides).
* [day5_champion_model_summary.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day5_champion_model_summary.png) — Executive dashboard comparing chemical profiles of Normal vs. Good wines alongside champion model specification.

---

## ⚙️ Setup and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run end-to-end training, interpretation, and inference routines:**
   ```python
   from src.data_loader import load_wine_data
   from src.feature_engineering import (
       create_binned_targets,
       create_engineered_features,
       get_stratified_train_test_split
   )
   from src.baseline_models import (
       get_baseline_classifiers,
       train_and_evaluate_baselines
   )
   from src.model_tuning import (
       get_tuning_pipelines,
       get_hyperparameter_grids,
       run_hyperparameter_search,
       evaluate_tuned_models_on_test,
       compile_tuned_comparison_dataframe
   )
   from src.model_interpretation import (
       load_champion_model_pipeline,
       predict_wine_quality_sample,
       compute_rf_feature_importances,
       compute_permutation_importances
   )

   # Load and prepare features
   df = load_wine_data("red")
   df_binned = create_binned_targets(df)
   df_feat = create_engineered_features(df_binned)

   X = df_feat.drop(columns=["quality", "binary_quality", "three_class_quality"])
   y = df_feat["binary_quality"]
   X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y)

   # 1. Run Baseline Benchmarking (Day 3)
   baseline_models = get_baseline_classifiers()
   base_results = train_and_evaluate_baselines(baseline_models, X_train, y_train, X_test, y_test)

   # 2. Run Hyperparameter Optimization (Day 4)
   tuning_pipelines = get_tuning_pipelines()
   param_grids = get_hyperparameter_grids()
   search_results = run_hyperparameter_search(tuning_pipelines, param_grids, X_train, y_train, n_splits=5)

   # 3. Load Champion Serialized Model & Run Production Inference (Day 5)
   champion_pipeline = load_champion_model_pipeline("models/champion_rf_pipeline.joblib")
   sample = {
       "fixed acidity": 8.5, "volatile acidity": 0.32, "citric acid": 0.45,
       "residual sugar": 2.1, "chlorides": 0.065, "free sulfur dioxide": 18.0,
       "total sulfur dioxide": 36.0, "density": 0.9955, "pH": 3.30,
       "sulphates": 0.82, "alcohol": 12.4
   }
   result = predict_wine_quality_sample(champion_pipeline, sample, threshold=0.35)
   print("Predicted:", result["predicted_label"], "| Probability:", result["probability_good"])
   ```

3. **Launch Jupyter Notebooks:**
   ```bash
   # Day 1: Data Ingestion & Structure Inspection
   jupyter notebook notebooks/Day1_Data_Inspection_and_Distribution.ipynb

   # Day 2: Exploratory Data Analysis & Physicochemical Profiling
   jupyter notebook notebooks/Day2_Exploratory_Data_Analysis.ipynb

   # Day 3: Target Engineering & Baseline Classification Models
   jupyter notebook notebooks/Day3_Target_Engineering_and_Baseline_Models.ipynb

   # Day 4: Hyperparameter Optimization, Cross-Validation & Model Evaluation
   jupyter notebook notebooks/Day4_Hyperparameter_Optimization_and_Evaluation.ipynb

   # Day 5: Model Interpretation, Feature Importance & Final Documentation
   jupyter notebook notebooks/Day5_Model_Interpretation_and_Final_Documentation.ipynb
   ```
