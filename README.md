# 🍷 Wine Quality Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project to train and compare classification models for predicting wine quality based on physicochemical properties such as acidity, density, and alcohol content using the UCI Wine Quality dataset.

---

## 📌 Project Overview & Objectives

* **Objective:** Predict sensory wine quality scores (scale 3–8) from physicochemical test results.
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
│   ├── Day1_Data_Inspection_and_Distribution.ipynb  # Executed data inspection & distribution notebook
│   └── generate_day1_notebook.py                    # Reproducible notebook generator
├── reports/
│   └── figures/
│       ├── day1_class_distribution.png              # Quality class distribution plot
│       └── day1_quality_distribution.png            # Annotated frequency bar plot
├── src/
│   ├── __init__.py                  # Package initializer
│   └── data_loader.py               # Reusable data loading, inspection, and plotting routines
├── download_data.py                 # UCI dataset automated download script
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 📊 Dataset Structure & Inspection

From `Data/winequality-red.csv`:
* **Dimensions:** 1,599 samples × 12 attributes (11 physicochemical inputs + 1 sensory quality rating).
* **Missing Values:** **0** missing values across all features.
* **Data Types:** All physicochemical attributes are `float64`; `quality` is discrete `int64`.

### Chemical Features:
1. `fixed acidity` (tartaric acid - $g/dm^3$)
2. `volatile acidity` (acetic acid - $g/dm^3$)
3. `citric acid` ($g/dm^3$)
4. `residual sugar` ($g/dm^3$)
5. `chlorides` (sodium chloride - $g/dm^3$)
6. `free sulfur dioxide` ($mg/dm^3$)
7. `total sulfur dioxide` ($mg/dm^3$)
8. `density` ($g/cm^3$)
9. `pH`
10. `sulphates` (potassium sulphate - $g/dm^3$)
11. `alcohol` (% by volume)
12. `quality` (score between 0 and 10, observed 3–8)

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

## ⚙️ Setup and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data inspection:**
   ```python
   from src.data_loader import load_wine_data, inspect_data, get_class_distribution

   df = load_wine_data("red")
   print(get_class_distribution(df))
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/Day1_Data_Inspection_and_Distribution.ipynb
   ```
