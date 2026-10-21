"""
Automated Pytest Suite: Production REST API and Inference Engine
Validates model artifact integrity, Pydantic data contracts, feature derivation parity,
single and batch inference endpoints, threshold override mechanics, and latency budgets.
"""

import pytest
import time
from pathlib import Path
from pydantic import ValidationError

from src.production_service import (
    create_app,
    get_test_client,
    WinePhysicochemicalInput,
    derive_interaction_features,
    DEFAULT_MODEL_PATH,
    DEFAULT_METADATA_PATH,
    RECOMMENDED_SCREENING_THRESHOLD,
    FULL_FEATURE_ORDER
)


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI TestClient fixture."""
    return get_test_client()


@pytest.fixture
def valid_standard_wine():
    """Typical table wine chemical profile."""
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.58,
        "citric acid": 0.08,
        "residual sugar": 1.9,
        "chlorides": 0.080,
        "free sulfur dioxide": 12.0,
        "total sulfur dioxide": 45.0,
        "density": 0.9965,
        "pH": 3.40,
        "sulphates": 0.52,
        "alcohol": 9.5
    }


@pytest.fixture
def valid_premium_wine():
    """Typical high-quality reserve red wine chemical profile."""
    return {
        "fixed acidity": 9.2,
        "volatile acidity": 0.30,
        "citric acid": 0.48,
        "residual sugar": 2.2,
        "chlorides": 0.065,
        "free sulfur dioxide": 22.0,
        "total sulfur dioxide": 38.0,
        "density": 0.9950,
        "pH": 3.30,
        "sulphates": 0.82,
        "alcohol": 12.6
    }


# ============================================================================
# 1. Artifact Verification & Feature Parity Tests
# ============================================================================

def test_model_artifact_exists_and_loads():
    """Verify serialized champion pipeline artifact exists and has predict_proba."""
    import joblib
    assert DEFAULT_MODEL_PATH.exists(), f"Model file missing at {DEFAULT_MODEL_PATH}"
    pipeline = joblib.load(DEFAULT_MODEL_PATH)
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "predict")


def test_metadata_artifact_exists():
    """Verify JSON metadata exists and includes core metrics."""
    import json
    assert DEFAULT_METADATA_PATH.exists(), f"Metadata file missing at {DEFAULT_METADATA_PATH}"
    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "test_roc_auc" in meta
    assert meta["test_roc_auc"] >= 0.90
    assert "operating_screening_threshold" in meta


def test_feature_engineering_parity(valid_standard_wine):
    """
    Ensure production service feature synthesis strictly matches domain math:
    - total_acidity = fixed + volatile
    - bound_so2 = total - free
    - free/total SO2
    - alcohol / density
    - sulphates / chlorides
    """
    eng = derive_interaction_features(valid_standard_wine)
    assert "total_acidity" in eng
    assert eng["total_acidity"] == pytest.approx(7.4 + 0.58)
    assert eng["bound_sulfur_dioxide"] == pytest.approx(45.0 - 12.0)
    assert eng["free_to_total_so2_ratio"] == pytest.approx(12.0 / (45.0 + 1e-5))
    assert eng["alcohol_to_density_ratio"] == pytest.approx(9.5 / 0.9965)
    assert eng["sulphate_to_chloride_ratio"] == pytest.approx(0.52 / (0.080 + 1e-5))


# ============================================================================
# 2. Pydantic Schema Validation Tests
# ============================================================================

def test_pydantic_valid_input_aliases(valid_premium_wine):
    """Verify WinePhysicochemicalInput accepts space-separated aliases."""
    model = WinePhysicochemicalInput(**valid_premium_wine)
    assert model.fixed_acidity == 9.2
    assert model.volatile_acidity == 0.30
    assert model.alcohol == 12.6


def test_pydantic_invalid_out_of_bounds():
    """Verify out-of-range inputs raise Pydantic validation error."""
    # Negative pH
    with pytest.raises(ValidationError):
        WinePhysicochemicalInput(
            fixed_acidity=8.0, volatile_acidity=0.4, citric_acid=0.2,
            residual_sugar=2.0, chlorides=0.08, free_sulfur_dioxide=15.0,
            total_sulfur_dioxide=50.0, density=0.996, pH=1.2, # Out of range [2.6, 4.6]
            sulphates=0.6, alcohol=10.0
        )

    # Impossible alcohol percentage
    with pytest.raises(ValidationError):
        WinePhysicochemicalInput(
            fixed_acidity=8.0, volatile_acidity=0.4, citric_acid=0.2,
            residual_sugar=2.0, chlorides=0.08, free_sulfur_dioxide=15.0,
            total_sulfur_dioxide=50.0, density=0.996, pH=3.3,
            sulphates=0.6, alcohol=35.0 # Out of range [7.0, 17.5]
        )


def test_pydantic_so2_contradiction():
    """Verify total sulfur dioxide cannot be strictly lower than free sulfur dioxide."""
    with pytest.raises(ValidationError):
        WinePhysicochemicalInput(
            fixed_acidity=8.0, volatile_acidity=0.4, citric_acid=0.2,
            residual_sugar=2.0, chlorides=0.08, free_sulfur_dioxide=45.0,
            total_sulfur_dioxide=20.0, # Contradiction: total < free
            density=0.996, pH=3.3, sulphates=0.6, alcohol=10.0
        )


# ============================================================================
# 3. REST API Endpoint Tests
# ============================================================================

def test_health_endpoint(client):
    """GET /health should return 200 with status=healthy and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_metadata_endpoint(client):
    """GET /metadata should return 200 with model card specifications."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "Tuned Random Forest" in data["model_name"]
    assert data["num_features"] == 16
    assert data["operating_screening_threshold"] == RECOMMENDED_SCREENING_THRESHOLD


def test_rules_endpoint(client):
    """GET /rules should return 200 with enological thresholds."""
    response = client.get("/rules")
    assert response.status_code == 200
    data = response.json()
    assert "thresholds" in data
    assert "alcohol" in data["thresholds"]
    assert "volatile_acidity" in data["thresholds"]


def test_predict_standard_wine(client, valid_standard_wine):
    """POST /predict on everyday table wine should return standard verdict."""
    response = client.post("/predict", json=valid_standard_wine)
    assert response.status_code == 200
    data = response.json()
    assert "is_premium_wine" in data
    assert "probability_good" in data
    assert "sommelier_grade" in data
    assert data["probability_good"] + data["probability_normal"] == pytest.approx(1.0, abs=1e-3)
    assert "engineered_features" in data
    assert len(data["engineered_features"]) == 5


def test_predict_premium_wine(client, valid_premium_wine):
    """POST /predict on high-quality specimen should classify as premium."""
    response = client.post("/predict", json=valid_premium_wine)
    assert response.status_code == 200
    data = response.json()
    assert data["is_premium_wine"] is True
    assert data["probability_good"] >= RECOMMENDED_SCREENING_THRESHOLD
    assert data["sommelier_grade"] in ["Reserve Grade", "Grand Reserve"]


def test_threshold_override(client, valid_premium_wine):
    """Custom threshold query parameter should override default decision boundary."""
    # Test with permissive threshold 0.20
    resp_low = client.post("/predict?threshold=0.20", json=valid_premium_wine)
    assert resp_low.status_code == 200
    assert resp_low.json()["operating_threshold"] == 0.20

    # Test with strict threshold 0.85 (within valid range [0.05, 0.95])
    resp_high = client.post("/predict?threshold=0.85", json=valid_premium_wine)
    assert resp_high.status_code == 200
    assert resp_high.json()["operating_threshold"] == 0.85


def test_batch_prediction_endpoint(client, valid_standard_wine, valid_premium_wine):
    """POST /predict/batch should vectorize multiple samples and provide summary KPIs."""
    batch_payload = {
        "samples": [
            dict(valid_standard_wine, sample_id="BATCH-001"),
            dict(valid_premium_wine, sample_id="BATCH-002"),
            dict(valid_standard_wine, sample_id="BATCH-003")
        ],
        "threshold": 0.35
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 3
    assert data["premium_samples"] >= 1
    assert data["standard_samples"] >= 1
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["sample_id"] == "BATCH-001"
    assert data["predictions"][1]["sample_id"] == "BATCH-002"
    assert data["batch_latency_ms"] >= 0.0


# ============================================================================
# 4. Latency Budget Verification
# ============================================================================

def test_inference_latency_budget(client, valid_standard_wine):
    """Verify single-sample inference latency budget is within production bounds."""
    latencies = []
    # Warmup
    for _ in range(3):
        client.post("/predict", json=valid_standard_wine)

    # 10 iterations measurement
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=valid_standard_wine)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        latencies.append(t_elapsed)

    import numpy as np
    p95 = float(np.percentile(latencies, 95))
    median_lat = float(np.median(latencies))
    # Assert ASGI test client latency remains within production envelope (< 1200ms)
    assert p95 < 1200.0, f"P95 latency {p95:.2f}ms exceeded 1200ms budget!"
    assert median_lat < 800.0, f"Median latency {median_lat:.2f}ms exceeded 800ms budget!"


def test_direct_service_engine_latency(valid_standard_wine):
    """Verify direct Python service inference latency is within CPU budget."""
    from src.production_service import WineQualityInferenceService, WinePhysicochemicalInput
    service = WineQualityInferenceService()
    wine_in = WinePhysicochemicalInput(**valid_standard_wine)

    # Warmup
    for _ in range(3):
        service.predict_single(wine_in)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        service.predict_single(wine_in)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    import numpy as np
    median_lat = float(np.median(latencies))
    assert median_lat < 600.0, f"Core model latency {median_lat:.2f}ms exceeded 600ms budget!"
