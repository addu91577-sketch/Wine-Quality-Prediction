"""
Unit and integration tests for Day 10 Enological Coupage Optimizer, Split Conformal
Uncertainty Quantification, Winery Economic Engine, and Stacking Meta-Learner.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from starlette.testclient import TestClient

from src.data_loader import load_wine_data
from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
from src.assemblage_and_conformal import (
    calculate_blended_chemistry,
    WineAssemblageOptimizer,
    EnologicalConformalPredictor,
    WineryEconomicEngine,
    WineStackingEnsemble,
    ALL_FEATURE_NAMES,
    DEFAULT_SCREENING_THRESHOLD
)
from src.production_service import create_app


@pytest.fixture(scope="module")
def sample_lots():
    """Two contrasting commercial harvest lots for blending experiments."""
    lot_a = {
        "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.02,
        "residual sugar": 2.0, "chlorides": 0.078, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 36.0, "density": 0.9976, "pH": 3.52,
        "sulphates": 0.54, "alcohol": 9.5
    }
    lot_b = {
        "fixed acidity": 10.2, "volatile acidity": 0.32, "citric acid": 0.46,
        "residual sugar": 2.5, "chlorides": 0.062, "free sulfur dioxide": 26.0,
        "total sulfur dioxide": 58.0, "density": 0.9959, "pH": 3.24,
        "sulphates": 0.80, "alcohol": 12.6
    }
    return [lot_a, lot_b]


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient fixture."""
    app = create_app()
    return TestClient(app)


# ============================================================================
# 1. Physicochemical Mixing Laws & Enological Mathematics
# ============================================================================

def test_blended_chemistry_mass_conservation_and_ph(sample_lots):
    """Test additive mass conservation and logarithmic [H+] acid equilibrium."""
    lot_a, lot_b = sample_lots
    weights = [0.4, 0.6]

    blend = calculate_blended_chemistry([lot_a, lot_b], weights)

    # 1. Linear additive checks
    expected_alc = 0.4 * 9.5 + 0.6 * 12.6  # 3.8 + 7.56 = 11.36
    assert blend["alcohol"] == pytest.approx(expected_alc, abs=1e-4)

    expected_va = 0.4 * 0.70 + 0.6 * 0.32  # 0.28 + 0.192 = 0.472
    assert blend["volatile acidity"] == pytest.approx(expected_va, abs=1e-4)

    # 2. Non-linear pH logarithmic equilibrium
    h_a = 10.0 ** (-3.52)
    h_b = 10.0 ** (-3.24)
    expected_h = 0.4 * h_a + 0.6 * h_b
    expected_ph = -np.log10(expected_h)
    assert blend["pH"] == pytest.approx(expected_ph, abs=1e-3)

    # 3. Recomputed domain interaction features
    assert blend["total_acidity"] == pytest.approx(blend["fixed acidity"] + blend["volatile acidity"], abs=1e-4)
    assert blend["bound_sulfur_dioxide"] == pytest.approx(blend["total sulfur dioxide"] - blend["free sulfur dioxide"], abs=1e-4)
    assert "free_to_total_so2_ratio" in blend
    assert "alcohol_to_density_ratio" in blend
    assert "sulphate_to_chloride_ratio" in blend


def test_blended_chemistry_single_lot_identity(sample_lots):
    """Single lot blended with weight 1.0 should return identical raw parameters."""
    lot_a = sample_lots[0]
    blend = calculate_blended_chemistry([lot_a], [1.0])
    for k, v in lot_a.items():
        assert blend[k] == pytest.approx(v, abs=1e-4)


# ============================================================================
# 2. Wine Assemblage (Coupage) Optimizer
# ============================================================================

def test_wine_assemblage_optimizer_rectification(sample_lots):
    """Coupage optimizer should combine lots to produce compliant, elevated wine."""
    optimizer = WineAssemblageOptimizer()
    res = optimizer.optimize_blend(sample_lots, lot_names=["Weak Lot A", "Reserve Lot B"])

    assert res["status"] == "success"
    assert len(res["optimal_weights"]) == 2
    assert sum(res["optimal_weights"]) == pytest.approx(1.0, abs=1e-3)
    assert res["predicted_premium_probability"] > res["individual_lot_evaluations"][0]["probability"]
    assert res["constraints_compliant"] is True
    assert res["blended_chemistry"]["volatile acidity"] <= 0.60
    assert res["blended_chemistry"]["alcohol"] >= 11.5


def test_simulate_blend_grid_2lots(sample_lots):
    """Verify 1D trajectory grid generates compliant steps."""
    optimizer = WineAssemblageOptimizer()
    df_grid = optimizer.simulate_blend_grid_2lots(sample_lots[0], sample_lots[1], steps=21)

    assert len(df_grid) == 21
    assert "predicted_probability" in df_grid.columns
    assert "is_enologically_compliant" in df_grid.columns
    assert df_grid["weight_lot_a"].iloc[0] == pytest.approx(0.0)
    assert df_grid["weight_lot_b"].iloc[0] == pytest.approx(1.0)


# ============================================================================
# 3. Split Conformal Prediction
# ============================================================================

def test_conformal_predictor_calibration():
    """Verify conformal non-conformity quantiles scale monotonically with confidence."""
    df = load_wine_data("red")
    df_tar = create_binned_targets(df)
    df_f = create_engineered_features(df_tar)

    X = df_f[ALL_FEATURE_NAMES]
    y = df_f["binary_quality"]
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_cal, y_tr, y_cal = get_stratified_train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    optimizer = WineAssemblageOptimizer()
    conformal = EnologicalConformalPredictor(model=optimizer.model)
    conformal.fit_calibration(X_cal, y_cal)

    q_90 = conformal.get_conformal_quantile(0.10)
    q_95 = conformal.get_conformal_quantile(0.05)
    # Stricter confidence (lower alpha) requires equal or larger non-conformity quantile
    assert q_95 >= q_90

    # Evaluate test coverage
    df_cov = conformal.evaluate_coverage(X_test, y_test, alpha_levels=[0.05, 0.10, 0.20])
    for _, row in df_cov.iterrows():
        # Coverage should remain within reasonable finite-sample tolerance of nominal
        assert row["empirical_coverage"] >= (row["nominal_confidence"] - 0.08)


def test_conformal_prediction_set_structure(sample_lots):
    """Verify single specimen conformal prediction set format."""
    optimizer = WineAssemblageOptimizer()
    conformal = EnologicalConformalPredictor(model=optimizer.model)
    conformal_path = Path("models/conformal_calibrator.json")
    if conformal_path.exists():
        conformal.load(conformal_path)
    else:
        df = load_wine_data("red")
        df_tar = create_binned_targets(df)
        df_f = create_engineered_features(df_tar)
        _, X_cal, _, y_cal = get_stratified_train_test_split(df_f[ALL_FEATURE_NAMES], df_f["binary_quality"], test_size=0.25, random_state=42)
        conformal.fit_calibration(X_cal, y_cal)

    res = conformal.predict_conformal_set(sample_lots[1], alpha=0.10)
    assert "prediction_set" in res
    assert "prediction_labels" in res
    assert res["set_size"] in [1, 2]
    assert "sommelier_action" in res


# ============================================================================
# 4. Winery Economic Value Engine
# ============================================================================

def test_winery_economic_engine():
    """Verify Expected Monetary Value (EMV) optimization finds a profitable operating threshold."""
    econ = WineryEconomicEngine(tp_margin=28.00, fp_penalty=-45.00, fn_penalty=-12.00, tn_margin=4.50)

    # Synthetic realistic ground truth & probabilities
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.15, size=500)
    y_prob = np.clip(y_true * 0.7 + rng.normal(0.2, 0.15, size=500), 0.0, 1.0)

    opt_res = econ.optimize_financial_threshold(y_true, y_prob, lot_size_bottles=25000)
    assert 0.05 <= opt_res["optimal_financial_threshold"] <= 0.95
    assert opt_res["max_harvest_profit"] >= opt_res["default_threshold_evaluation"]["harvest_lot_profit"]


# ============================================================================
# 5. Stacking Meta-Learner Ensemble
# ============================================================================

def test_stacking_ensemble_artifact_and_scoring():
    """Verify Stacking Ensemble achieves high discriminative power and valid serialization."""
    stacker_path = Path("models/stacking_ensemble.joblib")
    assert stacker_path.exists(), "Stacking ensemble artifact models/stacking_ensemble.joblib must exist."

    stacker = WineStackingEnsemble.load(stacker_path)
    df = load_wine_data("red")
    df_tar = create_binned_targets(df)
    df_f = create_engineered_features(df_tar)

    _, X_test, _, y_test = get_stratified_train_test_split(df_f[ALL_FEATURE_NAMES], df_f["binary_quality"], test_size=0.20, random_state=42)
    metrics = stacker.evaluate(X_test, y_test)

    assert metrics["stacking_roc_auc"] >= 0.90, f"Stacking ROC-AUC {metrics['stacking_roc_auc']} should be >= 0.90"
    assert metrics["stacking_brier_score"] <= 0.10, f"Brier score {metrics['stacking_brier_score']} should be <= 0.10"
    assert len(metrics["meta_learner_coefficients"]) == 4


# ============================================================================
# 6. REST API Endpoints Verification
# ============================================================================

def test_api_blend_optimize_endpoint(api_client, sample_lots):
    """POST /blend/optimize should return optimal blending weights and chemistry."""
    payload = {
        "lots": [
            dict(sample_lots[0], lot_name="Tank 1 (Sub-optimal)", volume_liters=5000),
            dict(sample_lots[1], lot_name="Tank 2 (Reserve)", volume_liters=8000)
        ],
        "constraints": {
            "max_volatile_acidity": 0.60,
            "min_alcohol": 11.5
        }
    }
    resp = api_client.post("/blend/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["optimal_weights"]) == 2
    assert data["is_premium"] is True
    assert data["constraints_compliant"] is True


def test_api_predict_conformal_endpoint(api_client, sample_lots):
    """POST /predict/conformal should return certified conformal prediction sets."""
    payload = dict(sample_lots[1], cultivar="red")
    resp = api_client.post("/predict/conformal?alpha=0.10", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_set" in data
    assert data["confidence_pct"] == "90.0%"
    assert data["set_size"] in [1, 2]


def test_api_economics_evaluate_endpoint(api_client):
    """POST /economics/evaluate should return commercial payoff optimization metrics."""
    payload = {
        "lot_size_bottles": 25000,
        "tp_margin": 28.00,
        "fp_penalty": -45.00
    }
    resp = api_client.post("/economics/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "optimal_financial_threshold" in data
    assert data["max_harvest_profit"] > 0.0
    assert "dollar_uplift_vs_default" in data


def test_api_metrics_stacking_endpoint(api_client):
    """GET /metrics/stacking should return benchmarking results for stacking meta-learner."""
    resp = api_client.get("/metrics/stacking")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stacking_roc_auc"] >= 0.90
    assert "meta_learner_coefficients" in data
    assert "random_forest" in data["base_models_benchmarks"]
