"""
Wine Quality Prediction package.

Modules:
- data_loader: UCI data ingestion and structural validation.
- eda_profiling: Statistical distributions, outlier profiling, and correlation analysis.
- feature_engineering: Target binning, domain chemical feature synthesis, and stratified splits.
- baseline_models: Scikit-learn classification pipelines (RF, SGD, SVC) and diagnostic evaluation.
- model_tuning: Hyperparameter optimization, Stratified 5-Fold CV, threshold calibration, and evaluation.
- model_interpretation: Feature importance (MDI & Permutation), directional weights, PDP profiles, enological rules, and serialization.
- deployment_synthesis: Hardware latency benchmarking, multi-criteria trade-off scoring, and handover synthesis.
- production_service: Production FastAPI REST microservice, Pydantic v2 schemas, and health endpoints.
- drift_monitoring: Covariate shift analysis (PSI, K-S), anomaly detection, sommelier concept drift tracking, and automated champion-challenger retraining.
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
from src.model_tuning import (
    get_tuning_pipelines,
    get_hyperparameter_grids,
    evaluate_kfold_cv,
    run_hyperparameter_search,
    evaluate_tuned_models_on_test,
    find_optimal_thresholds,
    compile_tuned_comparison_dataframe,
    compile_tuning_lift_dataframe,
    plot_cv_performance_comparison,
    plot_hyperparameter_tuning_results,
    plot_precision_recall_curves,
    plot_tuned_confusion_matrices,
    plot_tuned_vs_baseline_comparison
)
from src.model_interpretation import (
    compute_rf_feature_importances,
    compute_permutation_importances,
    compute_all_models_permutation_importance,
    compute_linear_coefficients,
    compute_partial_dependence_profiles,
    compute_2d_partial_dependence,
    compute_enological_rules_profile,
    save_champion_model_pipeline,
    load_champion_model_pipeline,
    predict_wine_quality_sample,
    plot_rf_feature_importance,
    plot_permutation_importance_comparison,
    plot_linear_coefficients,
    plot_partial_dependence_grid,
    plot_pdp_2d_interactions,
    plot_champion_model_summary
)
from src.deployment_synthesis import (
    benchmark_model_latencies,
    compile_master_comparison_table,
    compute_deployment_suitability_matrix,
    plot_comprehensive_model_comparison,
    plot_latency_vs_performance_tradeoff,
    plot_deployment_decision_radar,
    plot_executive_scorecard,
    generate_production_fastapi_snippet,
    export_final_comparison_markdown
)
from src.production_service import (
    WinePhysicochemicalInput,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    WineQualityInferenceService,
    derive_interaction_features,
    evaluate_enological_rules,
    create_app,
    get_test_client,
    RECOMMENDED_SCREENING_THRESHOLD,
    FeatureDriftItem,
    DriftAuditResponse,
    DriftMetricsResponse
)
from src.drift_monitoring import (
    calculate_psi,
    compute_kolmogorov_smirnov,
    compute_wasserstein_distance,
    compute_feature_drift,
    compute_dataset_drift_summary,
    simulate_vintage_drift,
    MultivariateAnomalyDetector,
    AuditPerformanceTracker,
    DriftRetrainingPipeline,
    RAW_PHYSICOCHEMICAL_FEATURES,
    ALL_MODEL_FEATURES,
    PSI_THRESHOLD_GREEN,
    PSI_THRESHOLD_AMBER
)
from src.cross_cultivar_and_recourse import (
    CultivarDomainAnalyzer,
    CultivarDetector,
    WhiteWineChampionTrainer,
    CellarRecourseOptimizer,
    UnifiedWinePredictor,
    compute_derived_features_dict,
    DEFAULT_WHITE_THRESHOLD,
    WHITE_CHAMPION_MODEL_PATH,
    CULTIVAR_DETECTOR_PATH
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
    "plot_roc_curves",
    "get_tuning_pipelines",
    "get_hyperparameter_grids",
    "evaluate_kfold_cv",
    "run_hyperparameter_search",
    "evaluate_tuned_models_on_test",
    "find_optimal_thresholds",
    "compile_tuned_comparison_dataframe",
    "compile_tuning_lift_dataframe",
    "plot_cv_performance_comparison",
    "plot_hyperparameter_tuning_results",
    "plot_precision_recall_curves",
    "plot_tuned_confusion_matrices",
    "plot_tuned_vs_baseline_comparison",
    "compute_rf_feature_importances",
    "compute_permutation_importances",
    "compute_all_models_permutation_importance",
    "compute_linear_coefficients",
    "compute_partial_dependence_profiles",
    "compute_2d_partial_dependence",
    "compute_enological_rules_profile",
    "save_champion_model_pipeline",
    "load_champion_model_pipeline",
    "predict_wine_quality_sample",
    "plot_rf_feature_importance",
    "plot_permutation_importance_comparison",
    "plot_linear_coefficients",
    "plot_partial_dependence_grid",
    "plot_pdp_2d_interactions",
    "plot_champion_model_summary",
    "benchmark_model_latencies",
    "compile_master_comparison_table",
    "compute_deployment_suitability_matrix",
    "plot_comprehensive_model_comparison",
    "plot_latency_vs_performance_tradeoff",
    "plot_deployment_decision_radar",
    "plot_executive_scorecard",
    "generate_production_fastapi_snippet",
    "export_final_comparison_markdown",
    "WinePhysicochemicalInput",
    "PredictionResult",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "MetadataResponse",
    "WineQualityInferenceService",
    "derive_interaction_features",
    "evaluate_enological_rules",
    "create_app",
    "get_test_client",
    "RECOMMENDED_SCREENING_THRESHOLD",
    "FeatureDriftItem",
    "DriftAuditResponse",
    "DriftMetricsResponse",
    "calculate_psi",
    "compute_kolmogorov_smirnov",
    "compute_wasserstein_distance",
    "compute_feature_drift",
    "compute_dataset_drift_summary",
    "simulate_vintage_drift",
    "MultivariateAnomalyDetector",
    "AuditPerformanceTracker",
    "DriftRetrainingPipeline",
    "RAW_PHYSICOCHEMICAL_FEATURES",
    "ALL_MODEL_FEATURES",
    "PSI_THRESHOLD_GREEN",
    "PSI_THRESHOLD_AMBER",
    "CultivarDomainAnalyzer",
    "CultivarDetector",
    "WhiteWineChampionTrainer",
    "CellarRecourseOptimizer",
    "UnifiedWinePredictor",
    "compute_derived_features_dict",
    "DEFAULT_WHITE_THRESHOLD",
    "WHITE_CHAMPION_MODEL_PATH",
    "CULTIVAR_DETECTOR_PATH"
]

