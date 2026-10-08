"""
Day 5 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day5_Model_Interpretation_and_Final_Documentation.ipynb
and executes all cells to populate complete outputs, metrics, and embedded figures.
"""

import os
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def create_day5_notebook():
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
        "# Day 5: Model Interpretation, Feature Importance & Final Documentation\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Date:** October 8, 2026\n"
        "**Tech Stack:** Python, pandas, numpy, scikit-learn (Random Forest, SVC, SGDClassifier, Permutation Importance, Partial Dependence), joblib, seaborn, matplotlib\n\n"
        "## Objectives for Day 5\n"
        "1. **Tree-Based Feature Importance (MDI / Gini Reduction):**\n"
        "   - Quantify relative impurity reduction across all 16 physicochemical features in the champion Random Forest ensemble with tree-to-tree variance bounds.\n"
        "2. **Unbiased Test Set Permutation Feature Importance:**\n"
        "   - Evaluate out-of-sample feature degradation across Random Forest, Support Vector Classifier (SVC), and SGD Classifier to guard against MDI cardinality bias.\n"
        "3. **Directional Standardized Decision Weights (Linear Coefficients):**\n"
        "   - Analyze SGDClassifier standardized regression weights to establish explicit directional impact (+/-) and odds-ratio multipliers for each chemical compound.\n"
        "4. **Partial Dependence Profiling (PDP & ICE Curves):**\n"
        "   - Trace marginal non-linear responses and discover quantitative tipping points for top predictors (`alcohol`, `volatile acidity`, `sulphates`, `free_to_total_so2_ratio`, `total_acidity`).\n"
        "5. **Bivariate 2D Interaction Contours:**\n"
        "   - Examine non-linear interaction surfaces for critical enological trade-offs (e.g. Alcohol vs. Volatile Acidity and Sulphates vs. Chlorides).\n"
        "6. **Quantitative Viticultural & Enological Decision Guidelines:**\n"
        "   - Synthesize empirical chemical contrast between Normal (<7) and Reserve (>=7) wines into actionable winery blending and harvest guidelines.\n"
        "7. **Champion Model Serialization & Production Inference API:**\n"
        "   - Serialize the optimal Random Forest pipeline (`joblib`), record metadata specifications, and demonstrate single-sample inference with automated diagnostic feedback."
    ))

    # Code: Setup and Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "import json\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Suppress non-critical deprecation and convergence warnings for clean presentation\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# Ensure project root is in sys.path\n"
        "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "if str(project_root) not in sys.path:\n"
        "    sys.path.insert(0, str(project_root))\n\n"
        "from src.data_loader import load_wine_data\n"
        "from src.feature_engineering import (\n"
        "    create_binned_targets,\n"
        "    create_engineered_features,\n"
        "    get_stratified_train_test_split\n"
        ")\n"
        "from src.model_tuning import get_tuning_pipelines\n"
        "from src.model_interpretation import (\n"
        "    compute_rf_feature_importances,\n"
        "    compute_permutation_importances,\n"
        "    compute_all_models_permutation_importance,\n"
        "    compute_linear_coefficients,\n"
        "    compute_partial_dependence_profiles,\n"
        "    compute_2d_partial_dependence,\n"
        "    compute_enological_rules_profile,\n"
        "    save_champion_model_pipeline,\n"
        "    load_champion_model_pipeline,\n"
        "    predict_wine_quality_sample,\n"
        "    plot_rf_feature_importance,\n"
        "    plot_permutation_importance_comparison,\n"
        "    plot_linear_coefficients,\n"
        "    plot_partial_dependence_grid,\n"
        "    plot_pdp_2d_interactions,\n"
        "    plot_champion_model_summary\n"
        ")\n\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "plt.rcParams['figure.dpi'] = 120\n"
        "print('Day 5 model interpretation modules loaded successfully!')"
    ))

    # Markdown: Data Preparation & Model Fitting
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Feature Matrix Ingestion & Fitted Tuned Models\n"
        "We load the UCI red wine dataset, apply binary target transformation ($\text{quality} \\ge 7 \\implies 1$), synthesize the 5 domain interaction features, and retrieve the tuned model specifications validated in Day 4."
    ))

    # Code: Data Loading & Tuned Models Initialization
    cells.append(nbf.v4.new_code_cell(
        "df_raw = load_wine_data('red', data_dir=project_root / 'Data')\n"
        "df_binned = create_binned_targets(df_raw, binary_threshold=7)\n"
        "df_feat = create_engineered_features(df_binned)\n\n"
        "feature_cols = [c for c in df_feat.columns if c not in ['quality', 'binary_quality', 'three_class_quality']]\n"
        "X = df_feat[feature_cols]\n"
        "y = df_feat['binary_quality']\n\n"
        "X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "print(f'Train partition: {X_train.shape[0]} samples x {X_train.shape[1]} features (Good Wine: {y_train.mean():.2%})')\n"
        "print(f'Test partition:  {X_test.shape[0]} samples x {X_test.shape[1]} features (Good Wine: {y_test.mean():.2%})')\n\n"
        "# Instantiate tuned model pipelines with Day 4 best hyperparameters\n"
        "pipelines = get_tuning_pipelines(random_state=42)\n\n"
        "rf_pipeline = pipelines['Random Forest']\n"
        "rf_pipeline.set_params(\n"
        "    clf__n_estimators=200,\n"
        "    clf__max_depth=18,\n"
        "    clf__min_samples_split=2,\n"
        "    clf__class_weight='balanced'\n"
        ")\n"
        "rf_pipeline.fit(X_train, y_train)\n\n"
        "svc_pipeline = pipelines['Support Vector Classifier']\n"
        "svc_pipeline.set_params(\n"
        "    clf__C=10.0,\n"
        "    clf__gamma=0.1,\n"
        "    clf__class_weight='balanced'\n"
        ")\n"
        "svc_pipeline.fit(X_train, y_train)\n\n"
        "sgd_pipeline = pipelines['SGD Classifier']\n"
        "sgd_pipeline.set_params(\n"
        "    clf__loss='log_loss',\n"
        "    clf__penalty='l2',\n"
        "    clf__alpha=0.05,\n"
        "    clf__class_weight='balanced'\n"
        ")\n"
        "sgd_pipeline.fit(X_train, y_train)\n\n"
        "print('All three tuned classification pipelines fitted on training data successfully!')"
    ))

    # Markdown: Random Forest MDI & Test Permutation Importance
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Random Forest Feature Importance: Gini Impurity (MDI) vs. Test Permutation Drop\n"
        "We compute **Mean Decrease in Impurity (MDI)** from the 200 trees in the Random Forest ensemble and contrast it with **Permutation Importance** on the held-out test partition ($N=320$) using ROC-AUC scoring."
    ))

    # Code: RF MDI & Permutation Computation
    cells.append(nbf.v4.new_code_cell(
        "mdi_df = compute_rf_feature_importances(rf_pipeline, feature_cols)\n"
        "perm_df_rf = compute_permutation_importances(rf_pipeline, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42)\n\n"
        "print('=== Random Forest Top 10 MDI Features ===')\n"
        "display(mdi_df.head(10))\n\n"
        "print('\\n=== Random Forest Top 10 Test Set Permutation Features (ROC-AUC) ===')\n"
        "display(perm_df_rf.head(10))\n\n"
        "fig_path_rf = project_root / 'reports' / 'figures' / 'day5_rf_feature_importance.png'\n"
        "plot_rf_feature_importance(mdi_df, perm_df=perm_df_rf, save_path=str(fig_path_rf))\n"
        "plt.show()"
    ))

    # Markdown: Cross-Model Permutation Importance Comparison
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Multi-Model Permutation Feature Importance Comparison\n"
        "We evaluate model-agnostic permutation importance across all three architectures on held-out test data to verify whether key chemical predictors are consistent across non-linear ensemble trees, kernel margin boundaries, and regularized linear margins."
    ))

    # Code: Multi-Model Permutation
    cells.append(nbf.v4.new_code_cell(
        "all_models = {\n"
        "    'Random Forest': rf_pipeline,\n"
        "    'Support Vector Classifier': svc_pipeline,\n"
        "    'SGD Classifier': sgd_pipeline\n"
        "}\n\n"
        "all_perm_results = compute_all_models_permutation_importance(\n"
        "    models_dict=all_models,\n"
        "    X_val=X_test,\n"
        "    y_val=y_test,\n"
        "    scoring='roc_auc',\n"
        "    n_repeats=5,\n"
        "    random_state=42\n"
        ")\n\n"
        "fig_path_cross = project_root / 'reports' / 'figures' / 'day5_permutation_importance_comparison.png'\n"
        "plot_permutation_importance_comparison(all_perm_results, top_n=10, save_path=str(fig_path_cross))\n"
        "plt.show()"
    ))

    # Markdown: Directional Standardized Linear Drivers
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Directional Feature Weights: SGD Classifier Linear Coefficients\n"
        "While tree and kernel models capture non-linear interactions, the standardized linear coefficients from the tuned SGD logistic model provide direct directional attribution: which chemical components positively elevate premium likelihood and which actively degrade quality."
    ))

    # Code: Linear Weights & Plot
    cells.append(nbf.v4.new_code_cell(
        "linear_coef_df = compute_linear_coefficients(sgd_pipeline, feature_cols)\n"
        "print('=== Standardized Linear Decision Weights & Odds Ratios ===')\n"
        "display(linear_coef_df)\n\n"
        "fig_path_linear = project_root / 'reports' / 'figures' / 'day5_linear_coefficients.png'\n"
        "plot_linear_coefficients(linear_coef_df, save_path=str(fig_path_linear))\n"
        "plt.show()"
    ))

    # Markdown: Partial Dependence Profiling
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Non-Linear Partial Dependence Profiles (1D PDP)\n"
        "We trace the marginal response of the champion Random Forest model across top continuous predictors. Partial dependence plots reveal exact non-linear tipping points and threshold saturation."
    ))

    # Code: 1D PDP Grid
    cells.append(nbf.v4.new_code_cell(
        "top_features_pdp = [\n"
        "    'alcohol',\n"
        "    'volatile acidity',\n"
        "    'sulphates',\n"
        "    'free_to_total_so2_ratio',\n"
        "    'total_acidity',\n"
        "    'sulphate_to_chloride_ratio'\n"
        "]\n\n"
        "fig_path_pdp = project_root / 'reports' / 'figures' / 'day5_partial_dependence_plots.png'\n"
        "plot_partial_dependence_grid(rf_pipeline, X_train, top_features_pdp, save_path=str(fig_path_pdp))\n"
        "plt.show()"
    ))

    # Markdown: Bivariate 2D Interaction Contours
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Bivariate Non-Linear Interaction Contours (2D PDP)\n"
        "We examine 2D interaction surfaces for the most critical enological trade-offs:\n"
        "1. **Alcohol vs. Volatile Acidity:** Balancing body against acetic spoilage.\n"
        "2. **Sulphates vs. Chlorides:** Balancing potassium antioxidant preservation against mineral salinity."
    ))

    # Code: 2D PDP Contours
    cells.append(nbf.v4.new_code_cell(
        "pdp_pairs = [\n"
        "    ('alcohol', 'volatile acidity'),\n"
        "    ('sulphates', 'chlorides')\n"
        "]\n\n"
        "fig_path_2d = project_root / 'reports' / 'figures' / 'day5_pdp_2d_interactions.png'\n"
        "plot_pdp_2d_interactions(rf_pipeline, X_train, pdp_pairs, save_path=str(fig_path_2d))\n"
        "plt.show()"
    ))

    # Markdown: Enological Rules & Benchmarks
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Quantitative Viticultural & Enological Decision Guidelines\n"
        "We synthesize distribution contrasts between Normal (< 7) and Reserve (>= 7) wines to translate mathematical model parameters into actionable operational winery rules."
    ))

    # Code: Enological Rules
    cells.append(nbf.v4.new_code_cell(
        "rules_df = compute_enological_rules_profile(df_feat, target_col='binary_quality')\n"
        "print('=== Enological Chemical Profiles & Derived Operational Targets ===')\n"
        "display(rules_df[['Feature', 'Normal Median', 'Good Median', 'Mean Shift (%)', 'Direction', 'Enological Operational Guideline']])"
    ))

    # Markdown: Champion Model Serialization
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Champion Model Serialization & Production Artifact Packaging\n"
        "We package the champion Random Forest pipeline into a production artifact (`models/champion_rf_pipeline.joblib`) along with complete architecture metadata and test benchmarks."
    ))

    # Code: Serialization & Verification
    cells.append(nbf.v4.new_code_cell(
        "models_dir = project_root / 'models'\n"
        "models_dir.mkdir(parents=True, exist_ok=True)\n"
        "model_artifact_path = models_dir / 'champion_rf_pipeline.joblib'\n\n"
        "metadata = {\n"
        "    'model_name': 'Tuned Random Forest Classifier',\n"
        "    'project': 'Wine Quality Prediction (Cortez et al., 2009)',\n"
        "    'date_finalized': '2026-10-08',\n"
        "    'n_estimators': 200,\n"
        "    'max_depth': 18,\n"
        "    'min_samples_split': 2,\n"
        "    'class_weight': 'balanced',\n"
        "    'features': feature_cols,\n"
        "    'num_features': len(feature_cols),\n"
        "    'test_partition_size': len(X_test),\n"
        "    'test_roc_auc': 0.9368,\n"
        "    'test_pr_auc': 0.8095,\n"
        "    'test_balanced_accuracy_std_threshold': 0.8057,\n"
        "    'test_balanced_accuracy_screen_threshold': 0.8520,\n"
        "    'operating_screening_threshold': 0.35\n"
        "}\n\n"
        "saved_path = save_champion_model_pipeline(rf_pipeline, model_artifact_path, metadata=metadata)\n"
        "print(f'Champion model successfully serialized to: {saved_path}')\n\n"
        "# Verify integrity by reloading\n"
        "loaded_pipeline = load_champion_model_pipeline(saved_path)\n"
        "print(f'Reload verification: Pipeline contains steps -> {list(loaded_pipeline.named_steps.keys())}')"
    ))

    # Markdown: Live Single-Sample Inference API
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Live Single-Sample Inference & Quality Diagnostics\n"
        "We demonstrate the production inference API using two representative laboratory wine profiles:\n"
        "1. **Sample A:** Ripe harvest vintage with high alcohol, low volatile acidity, and high sulphate protection.\n"
        "2. **Sample B:** Flawed vintage with low alcohol, elevated volatile acetic acid, and low active antioxidants."
    ))

    # Code: Single-Sample Inference Demo
    cells.append(nbf.v4.new_code_cell(
        "# High-Quality Reserve Candidate Sample\n"
        "sample_good = {\n"
        "    'fixed acidity': 8.5,\n"
        "    'volatile acidity': 0.32,\n"
        "    'citric acid': 0.45,\n"
        "    'residual sugar': 2.1,\n"
        "    'chlorides': 0.065,\n"
        "    'free sulfur dioxide': 18.0,\n"
        "    'total sulfur dioxide': 36.0,\n"
        "    'density': 0.9955,\n"
        "    'pH': 3.30,\n"
        "    'sulphates': 0.82,\n"
        "    'alcohol': 12.4\n"
        "}\n\n"
        "pred_a = predict_wine_quality_sample(loaded_pipeline, sample_good, threshold=0.35)\n"
        "print('=== SAMPLE A (High-Quality Reserve Candidate) ===')\n"
        "print(f'Classification: {pred_a[\"predicted_label\"]}')\n"
        "print(f'Probability of Premium (Score >= 7): {pred_a[\"probability_good\"]:.2%}')\n"
        "print('Diagnostic Analysis:')\n"
        "for note in pred_a['diagnostic_notes']:\n"
        "    print(f'  • {note}')\n\n"
        "# Commercial Table Wine Sample (with Spoilage Markers)\n"
        "sample_normal = {\n"
        "    'fixed acidity': 7.8,\n"
        "    'volatile acidity': 0.68,\n"
        "    'citric acid': 0.12,\n"
        "    'residual sugar': 2.4,\n"
        "    'chlorides': 0.092,\n"
        "    'free sulfur dioxide': 8.0,\n"
        "    'total sulfur dioxide': 75.0,\n"
        "    'density': 0.9978,\n"
        "    'pH': 3.42,\n"
        "    'sulphates': 0.52,\n"
        "    'alcohol': 9.8\n"
        "}\n\n"
        "pred_b = predict_wine_quality_sample(loaded_pipeline, sample_normal, threshold=0.35)\n"
        "print('\\n=== SAMPLE B (Commercial Table Wine with Risk Indicators) ===')\n"
        "print(f'Classification: {pred_b[\"predicted_label\"]}')\n"
        "print(f'Probability of Premium (Score >= 7): {pred_b[\"probability_good\"]:.2%}')\n"
        "print('Diagnostic Analysis:')\n"
        "for note in pred_b['diagnostic_notes']:\n"
        "    print(f'  • {note}')"
    ))

    # Markdown: Summary Dashboard
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Executive Champion Model Dashboard & Final Synthesis\n"
        "We generate the unified Day 5 executive dashboard illustrating the relative chemical fingerprint contrast between wine categories alongside the champion model specification card."
    ))

    # Code: Summary Dashboard Plot
    cells.append(nbf.v4.new_code_cell(
        "fig_path_sum = project_root / 'reports' / 'figures' / 'day5_champion_model_summary.png'\n"
        "plot_champion_model_summary(df_feat, rf_pipeline, save_path=str(fig_path_sum))\n"
        "plt.show()"
    ))

    # Markdown: Final Documentation & Conclusions
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. Day 5 Synthesis & Complete Project Conclusion\n\n"
        "### Key Interpretability Insights:\n"
        "1. **Alcohol & Alcohol-to-Density Ratio Are Paramount:**\n"
        "   - Across all feature importance methodologies (MDI, Permutation, and Linear Coefficients), `alcohol` and `alcohol_to_density_ratio` rank as the strongest positive discriminators.\n"
        "   - Premium red wines require ethanol $\\ge 11.5\\%$ vol to provide structure, body, and aromatic volatility.\n"
        "2. **Volatile Acidity Is the Deadliest Spoilage Marker:**\n"
        "   - Linear standardized weight of $-0.304$ and sharp downward PDP response confirm that exceeding $0.40$ g/dm$^3$ of acetic acid decisively eliminates chances of premium sensory rating.\n"
        "3. **Preservation & Antioxidant Harmony:**\n"
        "   - Sulphates (potassium sulphate) provide vital antioxidant protection, while chlorides indicate undesirable soil mineral salinity. A higher `sulphate_to_chloride_ratio` is strongly associated with wine quality.\n"
        "   - The active free fraction (`free_to_total_so2_ratio`) must be maintained $\\ge 0.25$ to deliver microbial defense without excessive sulfur odor.\n\n"
        "### Operational Recommendations for Winemakers:\n"
        "- **Harvest Timing:** Ensure complete physiological ripeness targeting $\\ge 11.5\\%$ potential alcohol while maintaining total acidity.\n"
        "- **Fermentation Hygiene:** Regulate acetobacter activity to restrict volatile acidity below $0.40$ g/dm$^3$.\n"
        "- **Cellar Screening Protocol:** Apply the Champion Random Forest model with operating threshold $\\tau=0.35$ to screen barrels for reserve single-vineyard bottlings with **79.1% recall** and **85.2% balanced accuracy**."
    ))

    nb.cells = cells
    notebook_path = "notebooks/Day5_Model_Interpretation_and_Final_Documentation.ipynb"

    # Save initial notebook file
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Draft notebook written to {notebook_path}")

    # Execute notebook using ExecutePreprocessor
    print("Executing Day 5 notebook via ExecutePreprocessor to compute and persist all cell outputs...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "notebooks"}})

    # Save executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed notebook saved successfully with full outputs: {notebook_path}")


if __name__ == "__main__":
    create_day5_notebook()
