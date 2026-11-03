"""
Automated Pytest Suite for Day 12:
- Fermentation Kinetics Simulator (Biomass proliferation, hexose consumption, ethanol kinetics)
- Stuck & Sluggish Fermentation Early-Warning Detector & Prescriptive Remediation
- Sensory Organoleptic 6-Axis Radar, Tasting Note Synthesis, and Food Pairings
- Causal Enological Engine (Doubly Robust IPW Average Treatment Effects)
- FastAPI Production REST Microservice Endpoints
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from starlette.testclient import TestClient

from src.data_loader import load_wine_data
from src.production_service import create_app
from src.fermentation_and_sensory import (
    FermentationKineticsSimulator,
    StuckFermentationDetector,
    SensoryOrganolepticEngine,
    CausalEnologicalEngine
)


@pytest.fixture(scope="module")
def api_client():
    """Provides test client connected to the production FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def baseline_sample():
    """Standard red wine chemical specimen."""
    return {
        "fixed acidity": 8.5,
        "volatile acidity": 0.38,
        "citric acid": 0.32,
        "residual sugar": 2.2,
        "chlorides": 0.068,
        "free sulfur dioxide": 22.0,
        "total sulfur dioxide": 52.0,
        "density": 0.9962,
        "pH": 3.38,
        "sulphates": 0.72,
        "alcohol": 12.1
    }


# ============================================================================
# 1. Fermentation Kinetics Simulator Tests
# ============================================================================

def test_fermentation_kinetics_rate_equations():
    """Verify forward numerical integration of alcoholic fermentation bioprocess."""
    sim = FermentationKineticsSimulator()
    res = sim.simulate_fermentation(
        initial_sugar_g_l=220.0,
        initial_yan_mg_l=200.0,
        jacket_temp_c=22.0,
        total_days=14
    )

    assert res["duration_days"] == 14
    assert res["final_residual_sugar_g_l"] < 220.0
    assert res["final_ethanol_pct"] > 8.0
    assert res["peak_fermentation_rate_g_l_day"] > 5.0
    assert len(res["trajectory"]) > 10

    # Test monotonic decay of sugar and accumulation of ethanol
    traj = res["trajectory"]
    sugars = [pt["sugar_g_l"] for pt in traj]
    ethanols = [pt["ethanol_pct"] for pt in traj]
    assert sugars[-1] <= sugars[0]
    assert ethanols[-1] >= ethanols[0]


def test_fermentation_thermal_stress_and_strains():
    """Verify temperature modulation and yeast strain profiles."""
    sim = FermentationKineticsSimulator()

    # Optimal controlled vs Warm thermal stress
    opt_run = sim.simulate_fermentation(initial_sugar_g_l=220.0, jacket_temp_c=20.0, total_days=10)
    warm_run = sim.simulate_fermentation(initial_sugar_g_l=220.0, jacket_temp_c=32.0, total_days=10)

    # Elevated temperature produces higher stress volatile acidity
    assert warm_run["final_volatile_acidity_g_l"] >= opt_run["final_volatile_acidity_g_l"]


# ============================================================================
# 2. Stuck Fermentation Early-Warning Detector Tests
# ============================================================================

def test_stuck_detector_healthy_active():
    """Verify diagnostic classification on an active normal fermentation."""
    detector = StuckFermentationDetector()
    readings = [
        {"day": 0, "sugar": 230.0, "temp": 20.0, "yan": 220.0},
        {"day": 2, "sugar": 190.0, "temp": 22.0, "yan": 170.0},
        {"day": 4, "sugar": 135.0, "temp": 23.0, "yan": 110.0}
    ]
    diag = detector.diagnose_trajectory(readings)
    assert diag["diagnosis"] == "HEALTHY_ACTIVE"
    assert diag["risk_score_pct"] <= 25.0
    assert not diag["is_stuck_or_sluggish"]
    assert diag["current_rate_g_l_day"] > 10.0


def test_stuck_detector_imminent_stuck_alert():
    """Verify alarm generation and prescriptive actions on arrested tank."""
    detector = StuckFermentationDetector()
    readings = [
        {"day": 0, "sugar": 230.0, "temp": 20.0},
        {"day": 4, "sugar": 65.0, "temp": 23.0},
        {"day": 6, "sugar": 64.0, "temp": 21.0}  # Nearly zero consumption
    ]
    diag = detector.diagnose_trajectory(readings)
    assert diag["diagnosis"] == "STUCK_FERMENTATION"
    assert diag["risk_score_pct"] >= 90.0
    assert diag["is_stuck_or_sluggish"]
    assert len(diag["prescriptive_actions"]) >= 2
    action_types = [a["action"] for a in diag["prescriptive_actions"]]
    assert "DETOXIFY_LEES" in action_types
    assert "REINOCULATE_ACCLIMATIZED_STARTER" in action_types


def test_stuck_detector_thermal_hazard():
    """Verify thermal hazard detection on runaway overheated must."""
    detector = StuckFermentationDetector()
    readings = [
        {"day": 0, "sugar": 230.0, "temp": 22.0},
        {"day": 2, "sugar": 170.0, "temp": 32.5}  # Above 30°C
    ]
    diag = detector.diagnose_trajectory(readings)
    assert diag["diagnosis"] == "THERMAL_OVERHEATING_RISK"
    assert diag["risk_score_pct"] >= 70.0
    assert any("COOLING" in a["action"] for a in diag["prescriptive_actions"])


# ============================================================================
# 3. Sensory Organoleptic Lexicon & Spider Radar Tests
# ============================================================================

def test_sensory_organoleptic_profile_bounds(baseline_sample):
    """Verify 6-axis organoleptic radar dimensions and bounded scaling (1.0 - 10.0)."""
    res = SensoryOrganolepticEngine.compute_sensory_profile(baseline_sample)

    assert len(res["radar_scores"]) == 6
    for attr, score in res["radar_scores"].items():
        assert 1.0 <= score <= 10.0, f"Score for {attr} out of range: {score}"

    assert 1.0 <= res["overall_sensory_intensity"] <= 10.0
    assert res["dry_extract_est_g_l"] > 10.0

    # Tasting note
    tn = res["tasting_note"]
    assert "appearance" in tn and len(tn["appearance"]) > 10
    assert "nose" in tn and len(tn["nose"]) > 10
    assert "palate" in tn and len(tn["palate"]) > 10
    assert "finish" in tn and len(tn["finish"]) > 10

    # Culinary food pairings
    pairings = res["food_pairings"]
    assert len(pairings) >= 4
    for p in pairings:
        assert 0.0 <= p["affinity"] <= 100.0
        assert "pairing_rationale" in p


# ============================================================================
# 4. Causal Enological Engine Tests
# ============================================================================

def test_causal_enological_engine_ate():
    """Verify Doubly Robust IPW causal Average Treatment Effect computation."""
    df_red = load_wine_data("red")
    res = CausalEnologicalEngine.estimate_causal_effect(
        df_red,
        intervention_key="low_volatile_acidity",
        n_bootstraps=30
    )

    assert "causal_ate" in res
    assert "ci_95_lower" in res
    assert "ci_95_upper" in res
    assert res["treated_sample_count"] > 0
    assert res["control_sample_count"] > 0
    # Low VA should have a positive causal effect on wine quality
    assert res["causal_ate"] > 0.0


# ============================================================================
# 5. Model Benchmark Artifacts Integrity
# ============================================================================

def test_day12_benchmark_artifacts_exist():
    """Verify Day 12 JSON model benchmark artifacts exist and contain valid payloads."""
    p_ferm = Path("models/fermentation_kinetics_benchmarks.json")
    p_sens = Path("models/sensory_lexicon_benchmarks.json")
    p_caus = Path("models/causal_intervention_benchmarks.json")

    assert p_ferm.exists(), "Missing models/fermentation_kinetics_benchmarks.json"
    assert p_sens.exists(), "Missing models/sensory_lexicon_benchmarks.json"
    assert p_caus.exists(), "Missing models/causal_intervention_benchmarks.json"

    with open(p_ferm) as f:
        d_ferm = json.load(f)
        assert "standard_red_fermentation" in d_ferm

    with open(p_sens) as f:
        d_sens = json.load(f)
        assert "reserve_grand_vin" in d_sens

    with open(p_caus) as f:
        d_caus = json.load(f)
        assert "low_volatile_acidity" in d_caus


# ============================================================================
# 6. Production REST API Endpoints Tests
# ============================================================================

def test_api_fermentation_simulate_endpoint(api_client):
    """Verify POST /fermentation/simulate endpoint."""
    payload = {
        "initial_sugar_g_l": 225.0,
        "initial_yan_mg_l": 220.0,
        "initial_temp_c": 20.0,
        "jacket_temp_c": 20.0,
        "total_days": 10,
        "yeast_strain": "EC1118_champagne"
    }
    resp = api_client.post("/fermentation/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "fermentation_status" in data
    assert "final_residual_sugar_g_l" in data
    assert "final_ethanol_pct" in data
    assert len(data["trajectory"]) > 0


def test_api_fermentation_diagnose_stuck_endpoint(api_client):
    """Verify POST /fermentation/diagnose-stuck endpoint."""
    payload = {
        "readings": [
            {"day": 0.0, "sugar": 220.0, "temp": 20.0},
            {"day": 3.0, "sugar": 180.0, "temp": 22.0},
            {"day": 6.0, "sugar": 178.0, "temp": 22.0}
        ]
    }
    resp = api_client.post("/fermentation/diagnose-stuck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["diagnosis"] == "STUCK_FERMENTATION"
    assert data["is_stuck_or_sluggish"] is True
    assert data["risk_score_pct"] >= 90.0


def test_api_sensory_profile_endpoint(api_client, baseline_sample):
    """Verify POST /sensory/profile endpoint."""
    payload = {"sample": baseline_sample}
    resp = api_client.post("/sensory/profile", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["radar_scores"]) == 6
    assert "tasting_note" in data
    assert "food_pairings" in data


def test_api_causal_effects_endpoint(api_client):
    """Verify GET /metrics/causal-effects endpoint."""
    resp = api_client.get("/metrics/causal-effects")
    assert resp.status_code == 200
    data = resp.json()
    assert "interventions" in data
    assert "low_volatile_acidity" in data["interventions"]
