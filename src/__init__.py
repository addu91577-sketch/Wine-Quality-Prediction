"""
Wine Quality Prediction package.

Modules:
- data_loader: UCI data ingestion and structural validation.
- eda_profiling: Statistical distributions, outlier profiling, and correlation analysis.
- feature_engineering: Target binning, domain chemical feature synthesis, and stratified splits.
- baseline_models: Scikit-learn classification pipelines (RF, SGD, SVC) and diagnostic evaluation.
"""

from src.data_loader import load_wine_data, inspect_data, get_class_distribution
from src.eda_profiling import (
    get_feature_columns,
    compute_univariate_stats,
    detect_outliers_iqr,
    compute_target_correlations
)
from src.feature_engineering import (
    create_binned_targets,
    compute_target_distribution_summary,
    create_engineered_features,
    get_stratified_train_test_split,
    verify_stratification_balance
)
from src.baseline_models import (
    get_baseline_classifiers,
    train_and_evaluate_baselines,
    compile_comparison_dataframe,
    plot_target_distribution_comparison,
    plot_confusion_matrices,
    plot_baseline_model_comparison,
    plot_roc_curves
)

__all__ = [
    "load_wine_data",
    "inspect_data",
    "get_class_distribution",
    "get_feature_columns",
    "compute_univariate_stats",
    "detect_outliers_iqr",
    "compute_target_correlations",
    "create_binned_targets",
    "compute_target_distribution_summary",
    "create_engineered_features",
    "get_stratified_train_test_split",
    "verify_stratification_balance",
    "get_baseline_classifiers",
    "train_and_evaluate_baselines",
    "compile_comparison_dataframe",
    "plot_target_distribution_comparison",
    "plot_confusion_matrices",
    "plot_baseline_model_comparison",
    "plot_roc_curves"
]
