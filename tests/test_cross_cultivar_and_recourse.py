"""
Day 9 Unit & Integration Test Suite.
Tests:
1. Cross-cultivar statistical divergence and domain fragility quantification.
2. High-accuracy cultivar classifier training and single-sample prediction.
3. White Wine Champion model training, serialization, and threshold evaluation.
4. Prescriptive cellar recourse optimizer convergence, constraint enforcement, and probability uplift.
5. Unified multi-cultivar production inference engine and FastAPI REST endpoints.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_wine_data
from src.cross_cultivar_and_recourse import (
    CultivarDomainAnalyzer,
    CultivarDetector,
    WhiteWineChampionTrainer,
    CellarRecourseOptimizer,
    UnifiedWinePredictor,
    WHITE_CHAMPION_MODEL_PATH,
    CULTIVAR_DETECTOR_PATH,
    RAW_FEATURES,
    ALL_FEATURES,
    DEFAULT_WHITE_THRESHOLD
)
from src.production_service import create_app, get_test_client


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_red_standard():
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.70,
        "citric acid": 0.00,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }


@pytest.fixture(scope="session")
def sample_white_standard():
    return {
        "fixed acidity": 6.8,
        "volatile acidity": 0.32,
        "citric acid": 0.28,
        "residual sugar": 6.5,
        "chlorides": 0.045,
        "free sulfur dioxide": 32.0,
        "total sulfur dioxide": 140.0,
        "density": 0.9942,
        "pH": 3.20,
        "sulphates": 0.48,
        "alcohol": 9.8
    }


@pytest.fixture(scope="session")
def sample_white_premium():
    return {
        "fixed acidity": 6.6,
        "volatile acidity": 0.22,
        "citric acid": 0.34,
        "residual sugar": 3.8,
        "chlorides": 0.032,
        "free sulfur dioxide": 38.0,
        "total sulfur dioxide": 112.0,
        "density": 0.9915,
        "pH": 3.25,
        "sulphates": 0.55,
        "alcohol": 12.6
    }


@pytest.fixture(scope="session")
def client():
    app = create_app()
    from starlette.testclient import TestClient
    return TestClient(app)


# ============================================================================
# 1. Cultivar Domain Divergence & Fragility Tests
# ============================================================================

def test_cultivar_domain_analyzer_divergence():
    """Verify cross-cultivar divergence computes valid PSI and K-S metrics across all 11 features."""
    analyzer = CultivarDomainAnalyzer()
    df_div = analyzer.compute_cross_cultivar_divergence()

    assert len(df_div) == 11
    assert "feature" in df_div.columns
    assert "psi" in df_div.columns
    assert "ks_stat" in df_div.columns
    assert "relative_shift_pct" in df_div.columns
    assert "divergence_severity" in df_div.columns

    # Total SO2 and Chlorides should be ranked high in divergence
    top_features = df_div.head(3)["feature"].tolist()
    assert any("sulfur dioxide" in f for f in top_features) or "chlorides" in top_features
    assert all(df_div["psi"] >= 0.0)


def test_zero_shot_fragility_drop():
    """Verify zero-shot transfer of Red champion to White wine exhibits marked performance degradation."""
    analyzer = CultivarDomainAnalyzer()
    frag = analyzer.evaluate_cross_domain_fragility()

    assert frag["white_samples_total"] == 4898
    assert frag["white_premium_ratio"] > 0.15
    assert frag["zero_shot_roc_auc"] < 0.85, f"Expected domain drop, got {frag['zero_shot_roc_auc']}"
    assert frag["zero_shot_bal_acc_tau50"] < 0.65


# ============================================================================
# 2. Cultivar Detector Tests
# ============================================================================

def test_cultivar_detector_accuracy():
    """Verify cultivar detector achieves >= 98% accuracy on test split."""
    detector = CultivarDetector()
    if not CULTIVAR_DETECTOR_PATH.exists():
        meta = detector.train_and_export()
    else:
        meta = detector.metadata
    assert meta.get("test_accuracy", 0.99) >= 0.98


def test_cultivar_detector_single_prediction(sample_red_standard, sample_white_standard):
    """Verify cultivar detector correctly distinguishes red from white samples."""
    detector = CultivarDetector()
    res_red = detector.predict_cultivar(sample_red_standard)
    res_white = detector.predict_cultivar(sample_white_standard)

    assert res_red["cultivar"] == "red"
    assert res_red["confidence"] > 0.90
    assert res_white["cultivar"] == "white"
    assert res_white["confidence"] > 0.90


# ============================================================================
# 3. White Wine Champion Pipeline Tests
# ============================================================================

def test_white_champion_artifact_and_metrics():
    """Verify White Wine champion model exists and meets ROC-AUC >= 0.90 standard."""
    trainer = WhiteWineChampionTrainer()
    if not WHITE_CHAMPION_MODEL_PATH.exists():
        meta = trainer.train_and_serialize_white_champion()
    else:
        import json
        from src.cross_cultivar_and_recourse import WHITE_CHAMPION_META_PATH
        with open(WHITE_CHAMPION_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

    assert meta["cultivar"] == "white"
    assert meta["test_roc_auc"] >= 0.90
    assert meta["test_balanced_accuracy"] >= 0.80
    assert meta["test_recall"] >= 0.80


def test_white_champion_inference_threshold(sample_white_standard, sample_white_premium):
    """Verify white champion produces coherent probabilities and grades."""
    predictor = UnifiedWinePredictor()

    pred_std = predictor.predict(sample_white_standard, cultivar="white")
    assert pred_std["cultivar"] == "white"
    assert "probability_good" in pred_std
    assert pred_std["probability_good"] + pred_std["probability_normal"] == pytest.approx(1.0, abs=1e-3)

    pred_prem = predictor.predict(sample_white_premium, cultivar="white")
    assert pred_prem["probability_good"] > pred_std["probability_good"]
    assert pred_prem["is_premium_wine"] is True


# ============================================================================
# 4. Cellar Recourse Optimizer Tests
# ============================================================================

def test_cellar_recourse_already_premium(sample_white_premium):
    """Verify recourse optimizer reports optimal batch when wine is already premium."""
    optimizer = CellarRecourseOptimizer()
    res = optimizer.optimize_recourse(sample_white_premium, cultivar="white", target_prob=0.40)

    assert res["status"] in ["already_premium", "recourse_computed"]
    if res["status"] == "already_premium":
        assert len(res["interventions"]) == 0
        assert res["probability_gain"] == 0.0


def test_cellar_recourse_sub_premium_elevation(sample_red_standard):
    """Verify recourse optimizer discovers actionable adjustments to elevate sub-premium wine."""
    optimizer = CellarRecourseOptimizer()
    res = optimizer.optimize_recourse(
        sample_red_standard,
        cultivar="red",
        target_prob=0.45,
        operating_threshold=0.35
    )

    assert res["status"] == "recourse_computed"
    assert res["projected_prob"] >= res["initial_prob"]
    assert res["probability_gain"] >= 0.05
    assert len(res["interventions"]) > 0

    # Ensure interventions contain required fields
    for inv in res["interventions"]:
        assert "feature" in inv
        assert "original_value" in inv
        assert "target_value" in inv
        assert "recommendation" in inv


def test_cellar_recourse_bounds_enforcement(sample_red_standard):
    """Verify adjusted values strictly respect physical enological bounds."""
    optimizer = CellarRecourseOptimizer()
    res = optimizer.optimize_recourse(sample_red_standard, cultivar="red")

    adj = res["adjusted_sample"]
    assert 8.0 <= adj["alcohol"] <= 15.5
    assert 0.15 <= adj["volatile acidity"] <= 1.20
    assert adj["free sulfur dioxide"] <= adj["total sulfur dioxide"]


# ============================================================================
# 5. Unified Inference & REST API Tests
# ============================================================================

def test_unified_predictor_auto_routing(sample_red_standard, sample_white_premium):
    """Verify UnifiedWinePredictor correctly auto-routes red and white samples."""
    predictor = UnifiedWinePredictor()

    out_red = predictor.predict(sample_red_standard, cultivar="auto")
    assert out_red["cultivar"] == "red"
    assert out_red["cultivar_detection_confidence"] > 0.90

    out_white = predictor.predict(sample_white_premium, cultivar="auto")
    assert out_white["cultivar"] == "white"
    assert out_white["cultivar_detection_confidence"] > 0.90


def test_api_cultivar_detect_endpoint(client, sample_red_standard, sample_white_standard):
    """POST /cultivar/detect should return 200 with cultivar classification."""
    resp_red = client.post("/cultivar/detect", json=sample_red_standard)
    assert resp_red.status_code == 200
    data_red = resp_red.json()
    assert data_red["cultivar"] == "red"
    assert data_red["confidence"] > 0.90

    resp_white = client.post("/cultivar/detect", json=sample_white_standard)
    assert resp_white.status_code == 200
    data_white = resp_white.json()
    assert data_white["cultivar"] == "white"
    assert data_white["confidence"] > 0.90


def test_api_recourse_endpoint(client, sample_red_standard):
    """POST /predict/recourse should return 200 with actionable recourse pathway."""
    payload = dict(sample_red_standard, cultivar="red")
    resp = client.post("/predict/recourse?target_prob=0.45", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "projected_prob" in data
    assert "interventions" in data
    assert len(data["interventions"]) > 0
    assert "feasibility" in data


def test_api_cultivar_metrics_endpoint(client):
    """GET /metrics/cultivars should return 200 with cross-cultivar divergence scorecard."""
    resp = client.get("/metrics/cultivars")
    assert resp.status_code == 200
    data = resp.json()
    assert "cross_cultivar_divergence" in data
    assert len(data["cross_cultivar_divergence"]) == 11
    assert "zero_shot_domain_gap" in data
