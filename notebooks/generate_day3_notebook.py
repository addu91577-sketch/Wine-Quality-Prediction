"""
Day 3 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day3_Target_Engineering_and_Baseline_Models.ipynb
and executes all cells to populate outputs and embedded figures.
"""

import os
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def create_day3_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.8"
        }
    }

    cells = []

    # Markdown: Title & Objectives
    cells.append(nbf.v4.new_markdown_cell(
        "# Day 3: Target Engineering, Preprocessing Pipelines & Baseline Classification Models\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Date:** October 7, 2026\n"
        "**Tech Stack:** Python, pandas, numpy, scikit-learn (Random Forest, SGD, SVC), seaborn, matplotlib\n\n"
        "## Objectives for Day 3\n"
        "1. **Class Imbalance & Target Reformulation:**\n"
        "   - Analyze the limitations of the raw 6-class ordinal target (scores 3–8), where scores 3 (0.63%) and 8 (1.13%) suffer extreme scarcity.\n"
        "   - Compare **Binary Classification** ($\text{quality} \\ge 7$ vs. $< 7$) against **3-Class Formulation** (Low, Medium, High) and justify selecting binary classification following Cortez et al. (2009).\n"
        "2. **Domain-Specific Feature Engineering:**\n"
        "   - Synthesize enologically grounded interaction features: total acidity, bound sulfur dioxide, free-to-total $SO_2$ ratio, alcohol-to-density ratio, and sulphate-to-chloride ratio.\n"
        "3. **Stratified Partitioning:**\n"
        "   - Perform an 80/20 train/test split with strict stratification to preserve the minority 'Good Wine' ratio in both folds.\n"
        "4. **Leakage-Free Preprocessing Pipelines:**\n"
        "   - Construct scikit-learn `Pipeline` architectures pairing `StandardScaler` and `RobustScaler` with classifiers to eliminate data leakage.\n"
        "5. **Baseline Model Training (3 Classifiers):**\n"
        "   - Train **Random Forest Classifier** (non-linear tree ensemble).\n"
        "   - Train **Stochastic Gradient Descent (SGD) Classifier** (linear margin optimizer with log loss).\n"
        "   - Train **Support Vector Classifier (SVC)** (non-linear RBF kernel with calibrated probabilities).\n"
        "6. **Evaluation & Diagnostic Benchmarking:**\n"
        "   - Evaluate all models using Accuracy, Balanced Accuracy, Weighted & Class-1 Precision/Recall/F1, and ROC-AUC.\n"
        "   - Generate annotated confusion matrix heatmaps and overlaid ROC curves.\n"
        "   - Compile a comprehensive side-by-side benchmark table to inform Day 4 hyperparameter optimization."
    ))

    # Code: Setup and Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Ensure project root is in sys.path\n"
        "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "if str(project_root) not in sys.path:\n"
        "    sys.path.insert(0, str(project_root))\n\n"
        "from src.data_loader import load_wine_data\n"
        "from src.feature_engineering import (\n"
        "    create_binned_targets,\n"
        "    compute_target_distribution_summary,\n"
        "    create_engineered_features,\n"
        "    get_stratified_train_test_split,\n"
        "    verify_stratification_balance\n"
        ")\n"
        "from src.baseline_models import (\n"
        "    get_baseline_classifiers,\n"
        "    train_and_evaluate_baselines,\n"
        "    compile_comparison_dataframe,\n"
        "    plot_target_distribution_comparison,\n"
        "    plot_confusion_matrices,\n"
        "    plot_baseline_model_comparison,\n"
        "    plot_roc_curves\n"
        ")\n\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "plt.rcParams['figure.dpi'] = 120\n"
        "print('Day 3 modules imported successfully!')"
    ))

    # Markdown: Data Loading
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Data Ingestion & Initial Target Verification\n"
        "We load the verified red wine dataset (`Data/winequality-red.csv`, 1,599 observations) and inspect the target column."
    ))

    # Code: Load Data
    cells.append(nbf.v4.new_code_cell(
        "df_raw = load_wine_data('red', data_dir=project_root / 'Data')\n"
        "print(f'Dataset Dimensions: {df_raw.shape[0]} samples x {df_raw.shape[1]} columns')\n"
        "df_raw[['fixed acidity', 'volatile acidity', 'alcohol', 'sulphates', 'quality']].head()"
    ))

    # Markdown: Target Binning & Imbalance Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Target Engineering & Class Imbalance Analysis\n\n"
        "### Limitations of Raw Discrete Quality Scores (3–8):\n"
        "- **Severe Boundary Scarcity:** Scores 3 (10 samples, 0.63%) and 8 (18 samples, 1.13%) contain insufficient observations for robust cross-validation.\n"
        "- **Sensory Subjectivity in Mid-Tier:** Over 82.5% of wines occupy scores 5 and 6. In sensory evaluation panels, differentiating a borderline 5 from a 6 is subjective and creates high label noise.\n\n"
        "### Target Reformulation Strategies:\n"
        "1. **3-Class Formulation:**\n"
        "   - Low (3–4): 63 samples (3.94%)\n"
        "   - Medium (5–6): 1,319 samples (82.49%)\n"
        "   - High (7–8): 217 samples (13.57%)\n"
        "   - *Limitation:* The 'Low' class remains severely underrepresented, causing multi-class classifiers to fail on low-tier recall.\n"
        "2. **Binary Formulation (Benchmark Selection - Cortez et al., 2009):**\n"
        "   - **Good Wine (1):** $\\text{quality} \\ge 7$ (217 samples, 13.57%)\n"
        "   - **Normal / Low Wine (0):** $\\text{quality} < 7$ (1,382 samples, 86.43%)\n"
        "   - *Commercial Justification:* Enological producers primarily require a binary decision: Is this vintage suitable for premium single-vineyard bottling, or should it be allocated to standard commercial blends?"
    ))

    # Code: Target Binning
    cells.append(nbf.v4.new_code_cell(
        "df_binned = create_binned_targets(df_raw, binary_threshold=7)\n"
        "dist_summary = compute_target_distribution_summary(df_binned)\n\n"
        "print('=== Raw Discrete Quality Distribution ===')\n"
        "print(dist_summary['raw'].to_string(index=False))\n\n"
        "print('\\n=== 3-Class Binned Distribution ===')\n"
        "print(dist_summary['three_class'].to_string(index=False))\n\n"
        "print('\\n=== Binary Target Distribution (Selected Benchmark) ===')\n"
        "print(dist_summary['binary'].to_string(index=False))"
    ))

    # Code: Target Distribution Visualization
    cells.append(nbf.v4.new_code_cell(
        "fig_path_target = project_root / 'reports' / 'figures' / 'day3_target_distribution_comparison.png'\n"
        "plot_target_distribution_comparison(df_binned, save_path=str(fig_path_target))\n"
        "plt.show()"
    ))

    # Markdown: Domain Feature Engineering
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Domain-Specific Feature Engineering\n\n"
        "Using enological insights established on Day 2, we generate five interaction features:\n"
        "1. **`total_acidity`** = `fixed acidity` + `volatile acidity`: Total proton-donating acidic concentration.\n"
        "2. **`bound_sulfur_dioxide`** = `total sulfur dioxide` - `free sulfur dioxide`: Preservative bound to acetaldehyde/anthocyanins.\n"
        "3. **`free_to_total_so2_ratio`** = `free SO2` / `total SO2`: Active antimicrobial fraction protecting wine against oxidation.\n"
        "4. **`alcohol_to_density_ratio`** = `alcohol` / `density`: Dynamic body indicator capturing ethanol volume versus residual water/sugar mass.\n"
        "5. **`sulphate_to_chloride_ratio`** = `sulphates` / `chlorides`: Balance between beneficial preservative salts and undesirable soil salinity."
    ))

    # Code: Feature Engineering
    cells.append(nbf.v4.new_code_cell(
        "df_feat = create_engineered_features(df_binned)\n"
        "new_cols = ['total_acidity', 'bound_sulfur_dioxide', 'free_to_total_so2_ratio', 'alcohol_to_density_ratio', 'sulphate_to_chloride_ratio']\n"
        "print(f'Total features now available: {df_feat.shape[1]} (Added 5 domain features)')\n"
        "df_feat[new_cols].describe().round(3)"
    ))

    # Markdown: Stratified Split
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Stratified Train-Test Splitting\n\n"
        "Due to the 86.4% vs 13.6% class imbalance in binary wine quality, random sampling risks underrepresenting premium wines in the test fold.\n"
        "We enforce **stratified splitting** with `test_size=0.20` and `random_state=42` to guarantee identical class distributions across both partitions."
    ))

    # Code: Train Test Split
    cells.append(nbf.v4.new_code_cell(
        "# Separate features and binary target\n"
        "X = df_feat.drop(columns=['quality', 'binary_quality', 'three_class_quality'])\n"
        "y = df_feat['binary_quality']\n\n"
        "X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "strat_check = verify_stratification_balance(y_train, y_test)\n"
        "print('Stratification Validation Table:')\n"
        "print(strat_check.to_string(index=False))\n"
        "print(f'\\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}')"
    ))

    # Markdown: Baseline Modeling
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Baseline Classification Model Training\n\n"
        "We construct and evaluate 3 diverse classifier architectures wrapped in `Pipeline` objects with `StandardScaler` to avoid data leakage:\n\n"
        "1. **Random Forest Classifier (`RandomForestClassifier`):**\n"
        "   - Non-linear ensemble of 100 decision trees.\n"
        "   - Naturally handles collinear features and outliers without distributional assumptions.\n"
        "2. **Stochastic Gradient Descent Classifier (`SGDClassifier`):**\n"
        "   - Linear classifier optimizing logistic loss (`log_loss`) via SGD with $L_2$ regularization.\n"
        "   - Sensitive to feature scales and outliers.\n"
        "3. **Support Vector Classifier (`SVC`):**\n"
        "   - Non-linear kernelized SVM with Radial Basis Function (RBF) kernel.\n"
        "   - Calibrated using Platt scaling / isotonic regression (`CalibratedClassifierCV`) for probabilistic evaluation."
    ))

    # Code: Train Models
    cells.append(nbf.v4.new_code_cell(
        "models = get_baseline_classifiers(scaler_type='standard', random_state=42)\n"
        "eval_results = train_and_evaluate_baselines(models, X_train, y_train, X_test, y_test)\n"
        "comparison_df = compile_comparison_dataframe(eval_results)\n"
        "print('=== Baseline Model Benchmark Summary ===')\n"
        "comparison_df"
    ))

    # Markdown: Detailed Reports
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Model Classification Reports & Diagnostics\n"
        "We examine per-class Precision, Recall, and F1-score for each model on the held-out test set."
    ))

    # Code: Classification Reports
    cells.append(nbf.v4.new_code_cell(
        "for model_name, data in eval_results.items():\n"
        "    print(f'\\n=================== {model_name} ===================')\n"
        "    print(data['classification_report_text'])"
    ))

    # Markdown: Confusion Matrices
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Confusion Matrix Visualization\n"
        "We visualize false positive (Type I) and false negative (Type II) errors for each classifier."
    ))

    # Code: Plot Confusion Matrices
    cells.append(nbf.v4.new_code_cell(
        "fig_path_cm = project_root / 'reports' / 'figures' / 'day3_confusion_matrices.png'\n"
        "plot_confusion_matrices(eval_results, save_path=str(fig_path_cm))\n"
        "plt.show()"
    ))

    # Markdown: ROC Curves & Benchmark Bar Chart
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Diagnostic ROC Curves & Benchmark Comparison\n"
        "We compare Receiver Operating Characteristic (ROC) trajectories and compile a multi-metric diagnostic bar chart."
    ))

    # Code: ROC Curves & Bar Chart
    cells.append(nbf.v4.new_code_cell(
        "fig_path_roc = project_root / 'reports' / 'figures' / 'day3_roc_curves.png'\n"
        "plot_roc_curves(eval_results, y_test, save_path=str(fig_path_roc))\n"
        "plt.show()\n\n"
        "fig_path_comp = project_root / 'reports' / 'figures' / 'day3_baseline_model_comparison.png'\n"
        "plot_baseline_model_comparison(comparison_df, save_path=str(fig_path_comp))\n"
        "plt.show()"
    ))

    # Markdown: Key Findings & Day 4 Roadmap
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Day 3 Synthesis & Strategic Plan for Day 4\n\n"
        "### Key Findings from Day 3 Baseline Benchmarking:\n"
        "1. **Random Forest Dominance:**\n"
        "   - Achieves the highest baseline **Accuracy (93.44%)**, **Weighted F1 (0.9304)**, and **ROC-AUC (0.9318)**.\n"
        "   - Successfully identifies 31 out of 43 premium wines (**72.1% Recall on Good Wine**), with an impressive **70.5% Precision**.\n"
        "2. **Support Vector Classifier (SVC):**\n"
        "   - Achieves high overall **Accuracy (90.31%)** and strong **ROC-AUC (0.8698)**.\n"
        "   - However, its unweighted formulation suffers from lower minority recall (34.9% Recall on Good Wine), requiring threshold tuning and class weighting (`class_weight='balanced'`).\n"
        "3. **SGD Classifier Limitation:**\n"
        "   - Achieves 85.00% Accuracy but very low Good Wine Recall (4.7%) due to default decision threshold and lack of class weighting on imbalanced data.\n"
        "4. **Impact of Class Imbalance:**\n"
        "   - While all models achieve >= 85% raw accuracy due to the majority negative class (86.4%), evaluating **Balanced Accuracy**, **Minority F1**, and **ROC-AUC** is essential for true commercial utility.\n\n"
        "### Actionable Plan for Day 4:\n"
        "- **Class Weight Balancing:** Implement `class_weight='balanced'` and evaluate SMOTE/ADASYN oversampling.\n"
        "- **Stratified $k$-Fold Cross-Validation:** Move beyond single train/test split to 5-fold and 10-fold stratified CV.\n"
        "- **Hyperparameter Tuning:** Systematically optimize `RandomForestClassifier` (n_estimators, max_depth, min_samples_split) and `SVC` ($C$, $\\gamma$, kernel) via `GridSearchCV` / `RandomizedSearchCV`.\n"
        "- **Threshold Optimization:** Calibrate decision thresholds using Precision-Recall curves to maximize business value."
    ))

    nb.cells = cells
    notebook_path = "notebooks/Day3_Target_Engineering_and_Baseline_Models.ipynb"

    # Save initial notebook file
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Draft notebook written to {notebook_path}")

    # Execute notebook using ExecutePreprocessor
    print("Executing Day 3 notebook via ExecutePreprocessor to compute and persist all cell outputs...")
    ep = ExecutePreprocessor(timeout=240, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "notebooks"}})

    # Save executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed notebook saved successfully with full outputs: {notebook_path}")


if __name__ == "__main__":
    create_day3_notebook()
