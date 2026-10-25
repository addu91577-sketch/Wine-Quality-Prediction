"""
Unit and integration tests for Continuous MLOps Monitoring, Statistical Drift Detection,
Multivariate Anomaly Screening, and Champion-Challenger Retraining.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_wine_data
from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
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
from src.production_service import get_test_client, WinePhysicochemicalInput, BatchPredictionRequest


@pytest.fixture(scope="module")
def red_wine_df():
    """Load baseline red wine dataset fixture."""
    return load_wine_data("red")


@pytest.fixture(scope="module")
def feature_engineered_data(red_wine_df):
    """Generate feature engineered train-test partitions."""
    df_binned = create_binned_targets(red_wine_df)
    df_feat = create_engineered_features(df_binned)
    X = df_feat.drop(columns=["quality", "binary_quality", "three_class_quality"])
    y = df_feat["binary_quality"]
    return get_stratified_train_test_split(X, y)


def test_calculate_psi_identical_distributions():
    """Identical or resampled distributions from the same process must yield PSI near 0.0."""
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=2.0, size=1000)
    tar = np.random.normal(loc=10.0, scale=2.0, size=1000)

    psi = calculate_psi(ref, tar, num_bins=10)
    assert isinstance(psi, float)
    assert psi < PSI_THRESHOLD_GREEN, f"Expected stable PSI < {PSI_THRESHOLD_GREEN}, got {psi}"


def test_calculate_psi_shifted_distributions():
    """A heavily shifted distribution must trigger a high PSI exceeding the severe alert threshold."""
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=1.5, size=1000)
    # Substantial distribution shift (+2.5 standard deviations)
    tar = np.random.normal(loc=14.0, scale=1.5, size=1000)

    psi = calculate_psi(ref, tar, num_bins=10)
    assert psi >= PSI_THRESHOLD_AMBER, f"Expected alert PSI >= {PSI_THRESHOLD_AMBER}, got {psi}"


def test_kolmogorov_smirnov_drift():
    """Two-sample KS test should reflect null hypothesis retention for identical and rejection for shifted."""
    np.random.seed(42)
    ref = np.random.normal(10.0, 1.0, 500)
    tar_identical = np.random.normal(10.0, 1.0, 500)
    tar_shifted = np.random.normal(12.5, 1.0, 500)

    res_null = compute_kolmogorov_smirnov(ref, tar_identical)
    assert res_null["p_value"] > 0.01

    res_alt = compute_kolmogorov_smirnov(ref, tar_shifted)
    assert res_alt["p_value"] < 1e-5
    assert res_alt["statistic"] > 0.5


def test_compute_dataset_drift_summary(red_wine_df):
    """Summary scorecard must evaluate all 11 physicochemical features with required schema fields."""
    shifted = simulate_vintage_drift(red_wine_df, scenario="climate_warming", severity=1.2)
    summary_df = compute_dataset_drift_summary(red_wine_df, shifted)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == len(RAW_PHYSICOCHEMICAL_FEATURES)
    expected_cols = ["Feature", "PSI", "KS_Statistic", "KS_P_Value", "Wasserstein_Dist", "Drift_Status", "Alert_Level"]
    for col in expected_cols:
        assert col in summary_df.columns

    # In climate warming scenario, alcohol, fixed acidity, and pH should experience drift
    alc_row = summary_df[summary_df["Feature"] == "alcohol"].iloc[0]
    assert alc_row["Alert_Level"] in ["AMBER", "RED"]


def test_simulate_vintage_drift_scenarios(red_wine_df):
    """Enological simulation scenarios must respect domain directionality and biological invariants."""
    # Scenario 1: Climate Warming
    warming = simulate_vintage_drift(red_wine_df, scenario="climate_warming", severity=1.0)
    assert warming["alcohol"].mean() > red_wine_df["alcohol"].mean()
    assert warming["fixed acidity"].mean() < red_wine_df["fixed acidity"].mean()
    assert (warming["total sulfur dioxide"] >= warming["free sulfur dioxide"]).all()

    # Scenario 2: Cool Rainy Harvest
    rainy = simulate_vintage_drift(red_wine_df, scenario="cool_rainy_harvest", severity=1.0)
    assert rainy["alcohol"].mean() < red_wine_df["alcohol"].mean()
    assert rainy["total sulfur dioxide"].mean() > red_wine_df["total sulfur dioxide"].mean()

    # Scenario 3: Stable Harvest
    stable = simulate_vintage_drift(red_wine_df, scenario="stable_harvest", severity=1.0)
    assert np.allclose(stable["alcohol"].values, red_wine_df["alcohol"].values)


def test_multivariate_anomaly_detector(red_wine_df):
    """Anomaly detector must identify standard harvest samples as inliers and aberrant specimens as anomalies."""
    detector = MultivariateAnomalyDetector(contamination=0.03, random_state=42)
    detector.fit(red_wine_df)

    clean_subset = red_wine_df.iloc[:20].copy()
    preds_clean = detector.predict(clean_subset)
    # The vast majority of standard samples should be inliers (+1)
    assert (preds_clean == 1).sum() >= 17

    # Create severe aberrant adulteration
    aberrant_samples = pd.DataFrame([
        {
            "fixed acidity": 28.0, "volatile acidity": 2.8, "citric acid": 2.5,
            "residual sugar": 45.0, "chlorides": 1.2, "free sulfur dioxide": 180.0,
            "total sulfur dioxide": 450.0, "density": 1.08, "pH": 1.9,
            "sulphates": 4.0, "alcohol": 22.0
        }
    ])
    preds_aberrant = detector.predict(aberrant_samples)
    assert preds_aberrant[0] == -1, "Severe physicochemical outlier should be flagged as anomaly (-1)"

    clean_df, anomaly_df = detector.filter_clean_samples(pd.concat([clean_subset, aberrant_samples]))
    assert len(anomaly_df) >= 1


def test_audit_performance_tracker():
    """Audit tracker must compute discrimination metrics and raise concept drift alarm when SLA is breached."""
    tracker = AuditPerformanceTracker()

    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0] * 10)
    
    # High-quality calibrated probabilities
    y_proba_good = np.where(y_true == 1, 0.85, 0.15)
    rec_good = tracker.evaluate_audit_batch(y_true, y_proba_good, threshold=0.35, batch_id="vintage_audit_2021")
    assert rec_good["concept_drift_detected"] is False
    assert rec_good["roc_auc"] > 0.90
    assert rec_good["alert_status"] == "GREEN"

    # Degraded probabilities under severe concept drift
    y_proba_degraded = np.random.uniform(0.3, 0.5, len(y_true))
    rec_bad = tracker.evaluate_audit_batch(y_true, y_proba_degraded, threshold=0.35, batch_id="vintage_audit_2023")
    assert rec_bad["concept_drift_detected"] is True
    assert rec_bad["alert_status"] == "RED"


def test_drift_retraining_trigger_rules(red_wine_df):
    """Retraining trigger must fire when >= 2 features or key enological levers exhibit RED PSI."""
    import joblib
    champ_pipeline = joblib.load("models/champion_rf_pipeline.joblib")

    df_binned = create_binned_targets(red_wine_df)
    df_feat = create_engineered_features(df_binned)
    X = df_feat.drop(columns=["quality", "binary_quality", "three_class_quality"])
    y = df_feat["binary_quality"]

    pipeline = DriftRetrainingPipeline(champ_pipeline, X, y)

    # 1. Stable summary -> Should not trigger
    stable_summary = pd.DataFrame([
        {"Feature": col, "PSI": 0.02, "Alert_Level": "GREEN"} for col in RAW_PHYSICOCHEMICAL_FEATURES
    ])
    should_retrain, reasons = pipeline.evaluate_retraining_trigger(stable_summary)
    assert should_retrain is False
    assert len(reasons) == 0

    # 2. Critical driver severe drift -> Should trigger
    drifted_summary = pd.DataFrame([
        {"Feature": "alcohol", "PSI": 0.38, "Alert_Level": "RED"},
        {"Feature": "volatile acidity", "PSI": 0.29, "Alert_Level": "RED"},
        {"Feature": "citric acid", "PSI": 0.04, "Alert_Level": "GREEN"}
    ])
    should_retrain, reasons = pipeline.evaluate_retraining_trigger(drifted_summary)
    assert should_retrain is True
    assert len(reasons) >= 1


def test_champion_challenger_tournament_execution(feature_engineered_data):
    """Challenger pipeline must be trained and evaluated against Champion on held-out audit set."""
    X_train, X_test, y_train, y_test = feature_engineered_data
    import joblib
    champ_pipeline = joblib.load("models/champion_rf_pipeline.joblib")

    retrainer = DriftRetrainingPipeline(champ_pipeline, X_train, y_train)

    # Small audit slices for fast test execution
    audit_new_X = X_test.iloc[:50]
    audit_new_y = y_test.iloc[:50]
    audit_held_out_X = X_test.iloc[50:100]
    audit_held_out_y = y_test.iloc[50:100]

    train_res = retrainer.train_challenger(audit_new_X, audit_new_y, cv_splits=3)
    assert "cv_roc_auc_mean" in train_res
    assert train_res["cv_roc_auc_mean"] > 0.70

    tourn_res = retrainer.execute_champion_challenger_tournament(audit_held_out_X, audit_held_out_y)
    assert tourn_res["decision"] in ["PROMOTED", "REJECTED"]
    assert "champion_metrics" in tourn_res
    assert "challenger_metrics" in tourn_res

    # Test audit export
    audit_json = Path("reports/test_retraining_governance.json")
    retrainer.export_governance_report(audit_json)
    assert audit_json.exists()
    audit_json.unlink()


def test_api_drift_monitoring_endpoints(red_wine_df):
    """FastAPI TestClient must verify /metrics/drift and /audit/evaluate endpoints."""
    client = get_test_client()

    # 1. GET /metrics/drift
    resp_metrics = client.get("/metrics/drift")
    assert resp_metrics.status_code == 200
    data = resp_metrics.json()
    assert data["status"] == "ACTIVE"
    assert data["baseline_dataset_size"] == 1599
    assert "psi_thresholds" in data

    # 2. POST /audit/evaluate with batch
    samples = []
    for _, row in red_wine_df.iloc[:20].iterrows():
        sample = {
            "fixed acidity": float(row["fixed acidity"]),
            "volatile acidity": float(row["volatile acidity"]),
            "citric acid": float(row["citric acid"]),
            "residual sugar": float(row["residual sugar"]),
            "chlorides": float(row["chlorides"]),
            "free sulfur dioxide": float(row["free sulfur dioxide"]),
            "total sulfur dioxide": float(row["total sulfur dioxide"]),
            "density": float(row["density"]),
            "pH": float(row["pH"]),
            "sulphates": float(row["sulphates"]),
            "alcohol": float(row["alcohol"])
        }
        samples.append(sample)

    resp_audit = client.post("/audit/evaluate", json={"samples": samples, "threshold": 0.35})
    assert resp_audit.status_code == 200
    audit_data = resp_audit.json()
    assert audit_data["total_features_evaluated"] == 11
    assert "overall_drift_status" in audit_data
    assert "feature_drift_scorecard" in audit_data
    assert len(audit_data["feature_drift_scorecard"]) == 11
