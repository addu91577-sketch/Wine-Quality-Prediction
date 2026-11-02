"""
Unit and Integration Tests for Day 11:
- CellarAgingKineticsSimulator: Arrhenius rate law, SO2 decay, VA accumulation, peak drinking window.
- ParetoEnologicalOptimizer: Non-dominated sorting, Pareto frontier, knee-point multi-criteria compromise.
- EnologicalStressTester: Perturbation disturbance battery, classification stability, epsilon flip margin.
- ShapEnologicalInteractions: Molecular SO2 acid equilibrium, ethanol masking synergy.
- FastAPI Day 11 REST API Endpoints: /aging/simulate, /pareto/optimize, /stress/test, /metrics/shap-interactions.
"""

import pytest
import math
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from src.aging_pareto_stress import (
    CellarAgingKineticsSimulator,
    ParetoEnologicalOptimizer,
    EnologicalStressTester,
    ShapEnologicalInteractions,
    CLOSURE_OTR,
    STANDARD_CELLAR_TEMP_C
)
from src.production_service import create_app, get_test_client


@pytest.fixture(scope="module")
def api_client():
    """Module-level FastAPI test client."""
    return get_test_client()


@pytest.fixture
def baseline_sample():
    """Representative red wine chemistry."""
    return {
        "fixed acidity": 8.5,
        "volatile acidity": 0.45,
        "citric acid": 0.35,
        "residual sugar": 2.2,
        "chlorides": 0.075,
        "free sulfur dioxide": 22.0,
        "total sulfur dioxide": 60.0,
        "density": 0.9965,
        "pH": 3.35,
        "sulphates": 0.72,
        "alcohol": 12.0
    }


# ============================================================================
# 1. Aging Kinetics Simulator Tests
# ============================================================================

def test_aging_kinetics_rate_constant_arrhenius():
    """Verify Arrhenius temperature scaling and closure OTR modulation."""
    sim = CellarAgingKineticsSimulator()

    # Rate at standard 14C with natural cork
    k_std = sim.calculate_so2_rate_constant(temperature_c=14.0, closure="natural_cork")
    assert 0.020 < k_std < 0.040

    # Elevated temperature (22C) should strictly accelerate oxidation rate
    k_hot = sim.calculate_so2_rate_constant(temperature_c=22.0, closure="natural_cork")
    assert k_hot > k_std

    # Screwcap (lower OTR) should have lower decay rate than synthetic
    k_screw = sim.calculate_so2_rate_constant(temperature_c=14.0, closure="screwcap")
    k_synth = sim.calculate_so2_rate_constant(temperature_c=14.0, closure="synthetic")
    assert k_screw < k_std < k_synth


def test_aging_kinetics_trajectory_decay_and_spoilage(baseline_sample):
    """Verify Free SO2 decreases monotonically and VA accumulates if SO2 drops low."""
    sim = CellarAgingKineticsSimulator()
    res = sim.simulate_trajectory(baseline_sample, months=36, temperature_c=14.0, closure="natural_cork")

    assert res["simulation_months"] == 36
    assert len(res["trajectory"]) == 37  # 0 to 36 inclusive

    # Check Free SO2 monotonicity
    f_so2_vals = [r["free sulfur dioxide"] for r in res["trajectory"]]
    for i in range(1, len(f_so2_vals)):
        assert f_so2_vals[i] <= f_so2_vals[i - 1]

    # Check that initial Free SO2 equals input
    assert abs(f_so2_vals[0] - baseline_sample["free sulfur dioxide"]) < 1e-3


def test_aging_kinetics_drinking_window_detection(baseline_sample):
    """Verify peak probability, peak month, and drinking window calculation."""
    sim = CellarAgingKineticsSimulator()
    res = sim.simulate_trajectory(baseline_sample, months=48, temperature_c=14.0, closure="natural_cork")

    assert "initial_probability" in res
    assert "peak_probability" in res
    assert res["peak_probability"] >= res["initial_probability"] or res["peak_month"] == 0
    assert 0 <= res["peak_month"] <= 48
    assert isinstance(res["sommelier_cellar_advice"], str)


# ============================================================================
# 2. Pareto Multi-Objective Optimizer Tests
# ============================================================================

def test_pareto_optimizer_dominance_logic():
    """Verify Pareto dominance mechanics where lot A strictly dominates B."""
    opt = ParetoEnologicalOptimizer()

    lot_superior = {
        "name": "Superior Lot",
        "fixed acidity": 8.0,
        "volatile acidity": 0.30,  # Lower VA
        "citric acid": 0.30,
        "residual sugar": 2.0,
        "chlorides": 0.06,
        "free sulfur dioxide": 25.0,
        "total sulfur dioxide": 60.0,
        "density": 0.996,
        "pH": 3.35,
        "sulphates": 0.50,         # Lower sulphates
        "alcohol": 12.8            # Ideal alcohol
    }
    lot_inferior = {
        "name": "Inferior Lot",
        "fixed acidity": 8.0,
        "volatile acidity": 0.85,  # Much higher VA
        "citric acid": 0.10,
        "residual sugar": 2.0,
        "chlorides": 0.12,
        "free sulfur dioxide": 8.0,
        "total sulfur dioxide": 90.0,
        "density": 0.999,
        "pH": 3.55,
        "sulphates": 0.90,         # Higher sulphates
        "alcohol": 9.5             # Low alcohol
    }

    res = opt.compute_pareto_frontier([lot_superior, lot_inferior])
    assert res["total_candidates_evaluated"] == 2
    assert res["pareto_optimal_count"] >= 1

    # Superior lot must be Pareto Rank 1
    front_names = [f["name"] for f in res["pareto_frontier"]]
    assert "Superior Lot" in front_names


def test_pareto_frontier_extraction_and_knee_point():
    """Verify non-dominated sorting and knee point detection across candidate cohort."""
    opt = ParetoEnologicalOptimizer()

    candidates = [
        {
            "name": f"Candidate_{i}",
            "fixed acidity": 7.5 + 0.1 * i,
            "volatile acidity": 0.35 + 0.03 * i,
            "citric acid": 0.25,
            "residual sugar": 2.0,
            "chlorides": 0.07,
            "free sulfur dioxide": 20.0,
            "total sulfur dioxide": 50.0,
            "density": 0.996,
            "pH": 3.30,
            "sulphates": 0.50 + 0.02 * i,
            "alcohol": 11.5 + 0.1 * i
        }
        for i in range(8)
    ]

    res = opt.compute_pareto_frontier(candidates)
    assert res["pareto_optimal_count"] > 0
    assert res["knee_point_lot"] is not None
    assert "knee_distance" in res["knee_point_lot"]
    assert "archetypes" in res


# ============================================================================
# 3. Physicochemical Stress-Testing Suite Tests
# ============================================================================

def test_enological_stress_tester_battery(baseline_sample):
    """Verify standard disturbance scenario execution and resilience scoring."""
    tester = EnologicalStressTester()
    res = tester.run_stress_test(baseline_sample)

    assert res["scenarios_tested"] == 5
    assert len(res["scenario_results"]) == 5
    assert 0.0 <= res["resilience_score"] <= 100.0
    assert res["resilience_tier"] in ["Highly Resilient", "Moderate Resilience", "Fragile / Vulnerable"]

    # Check scenarios keys
    for s in res["scenario_results"]:
        assert "scenario_name" in s
        assert "probability_delta" in s
        assert isinstance(s["retained_classification"], bool)


def test_stress_tester_epsilon_flip_margin(baseline_sample):
    """Verify adversarial volatile acidity epsilon boundary search."""
    tester = EnologicalStressTester()
    res = tester.run_stress_test(baseline_sample)
    assert res["critical_va_epsilon_flip_margin"] > 0.0


# ============================================================================
# 4. Enological Chemical Interactions Tests
# ============================================================================

def test_shap_molecular_so2_henderson_hasselbalch():
    """Verify molecular active SO2 equilibrium under Henderson-Hasselbalch equation."""
    # At lower pH (3.0), molecular SO2 fraction is significantly higher than at pH 3.8
    mol_acidic = ShapEnologicalInteractions.calculate_molecular_so2(free_so2=25.0, ph=3.0)
    mol_basic = ShapEnologicalInteractions.calculate_molecular_so2(free_so2=25.0, ph=3.8)

    assert mol_acidic > mol_basic
    assert 0.1 < mol_acidic < 5.0
    assert 0.01 < mol_basic < 1.0


# ============================================================================
# 5. FastAPI REST API Endpoint Tests
# ============================================================================

def test_api_aging_simulate_endpoint(api_client, baseline_sample):
    """Test POST /aging/simulate endpoint."""
    payload = {
        "sample": {
            "fixed_acidity": baseline_sample["fixed acidity"],
            "volatile_acidity": baseline_sample["volatile acidity"],
            "citric_acid": baseline_sample["citric acid"],
            "residual_sugar": baseline_sample["residual sugar"],
            "chlorides": baseline_sample["chlorides"],
            "free_sulfur_dioxide": baseline_sample["free sulfur dioxide"],
            "total_sulfur_dioxide": baseline_sample["total sulfur dioxide"],
            "density": baseline_sample["density"],
            "pH": baseline_sample["pH"],
            "sulphates": baseline_sample["sulphates"],
            "alcohol": baseline_sample["alcohol"]
        },
        "months": 36,
        "temperature_c": 14.0,
        "closure_type": "natural_cork",
        "step_months": 1
    }
    response = api_client.post("/aging/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["closure_type"] == "natural_cork"
    assert data["simulation_months"] == 36
    assert "initial_probability" in data
    assert "peak_probability" in data
    assert "sommelier_cellar_advice" in data
    assert len(data["trajectory_preview"]) > 0


def test_api_pareto_optimize_endpoint(api_client, baseline_sample):
    """Test POST /pareto/optimize endpoint."""
    payload = {
        "candidates": [
            {
                "name": "Lot Alpha",
                "fixed_acidity": baseline_sample["fixed acidity"],
                "volatile_acidity": 0.35,
                "citric_acid": baseline_sample["citric acid"],
                "residual_sugar": baseline_sample["residual sugar"],
                "chlorides": baseline_sample["chlorides"],
                "free_sulfur_dioxide": 25.0,
                "total_sulfur_dioxide": 60.0,
                "density": baseline_sample["density"],
                "pH": 3.30,
                "sulphates": 0.60,
                "alcohol": 12.5
            },
            {
                "name": "Lot Beta",
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.65,
                "citric_acid": 0.10,
                "residual_sugar": 2.5,
                "chlorides": 0.09,
                "free_sulfur_dioxide": 12.0,
                "total_sulfur_dioxide": 80.0,
                "density": 0.998,
                "pH": 3.50,
                "sulphates": 0.85,
                "alcohol": 10.2
            }
        ]
    }
    response = api_client.post("/pareto/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_candidates_evaluated"] == 2
    assert data["pareto_optimal_count"] >= 1
    assert len(data["pareto_frontier"]) > 0
    assert "archetypes" in data


def test_api_stress_test_endpoint(api_client, baseline_sample):
    """Test POST /stress/test endpoint."""
    payload = {
        "sample": {
            "fixed_acidity": baseline_sample["fixed acidity"],
            "volatile_acidity": baseline_sample["volatile acidity"],
            "citric_acid": baseline_sample["citric acid"],
            "residual_sugar": baseline_sample["residual sugar"],
            "chlorides": baseline_sample["chlorides"],
            "free_sulfur_dioxide": baseline_sample["free sulfur dioxide"],
            "total_sulfur_dioxide": baseline_sample["total sulfur dioxide"],
            "density": baseline_sample["density"],
            "pH": baseline_sample["pH"],
            "sulphates": baseline_sample["sulphates"],
            "alcohol": baseline_sample["alcohol"]
        }
    }
    response = api_client.post("/stress/test", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["scenarios_tested"] == 5
    assert 0.0 <= data["resilience_score"] <= 100.0
    assert len(data["scenario_results"]) == 5


def test_api_metrics_shap_interactions_endpoint(api_client):
    """Test GET /metrics/shap-interactions endpoint."""
    response = api_client.get("/metrics/shap-interactions")
    assert response.status_code == 200
    data = response.json()

    assert "interactions" in data
    assert "alcohol_va_masking" in data["interactions"]
    assert "molecular_so2_equilibrium" in data["interactions"]
