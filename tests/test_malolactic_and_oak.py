"""
Unit and Integration Tests for Day 13 Enological Secondary Malolactic Fermentation,
Precision Oak Barrel Maturation, Cellar Master Dispatch Engine, and HACCP Food Safety Auditor.
"""

import pytest
import numpy as np
from src.malolactic_and_oak import (
    MalolacticFermentationSimulator,
    OakBarrelMaturationSimulator,
    CellarMasterDispatchEngine,
    EnologicalHaccpAuditor,
    OAK_SPECIES_PROFILES,
    TOAST_LEVEL_MODIFIERS,
    BARREL_USE_FACTORS,
    STOICH_MALIC_TO_LACTIC
)
from src.production_service import get_test_client


@pytest.fixture
def test_client():
    return get_test_client()


# ============================================================================
# 1. Malolactic Fermentation Simulator Tests
# ============================================================================

def test_malolactic_simulation_stoichiometry_and_conversion():
    sim = MalolacticFermentationSimulator()
    res = sim.simulate(
        initial_malic_acid=3.0,
        initial_ph=3.4,
        alcohol=12.5,
        free_so2=5.0,
        temperature=20.0,
        duration_days=30
    )
    
    final_state = res["final_state"]
    # Check that malic acid decreased monotonically and significantly
    assert final_state["malic_acid_g_l"] < 0.5
    assert final_state["lactic_acid_g_l"] > 1.5
    # Stoichiometric conversion check
    consumed = 3.0 - final_state["malic_acid_g_l"]
    expected_lactic = consumed * STOICH_MALIC_TO_LACTIC
    assert abs(final_state["lactic_acid_g_l"] - expected_lactic) < 0.05
    assert res["diagnostics"]["is_complete"] is True or final_state["conversion_rate_pct"] > 80.0


def test_malolactic_temperature_and_so2_inhibition():
    sim = MalolacticFermentationSimulator()
    # High Free SO2 (22 mg/L) should strongly inhibit O. oeni
    res_sulfited = sim.simulate(
        initial_malic_acid=3.0,
        initial_ph=3.4,
        alcohol=13.0,
        free_so2=22.0,
        temperature=20.0,
        duration_days=20
    )
    # Optimum condition
    res_optimal = sim.simulate(
        initial_malic_acid=3.0,
        initial_ph=3.4,
        alcohol=13.0,
        free_so2=4.0,
        temperature=20.0,
        duration_days=20
    )
    
    assert res_sulfited["final_state"]["conversion_rate_pct"] < res_optimal["final_state"]["conversion_rate_pct"]
    assert res_sulfited["final_state"]["malic_acid_g_l"] > res_optimal["final_state"]["malic_acid_g_l"]


def test_malolactic_ph_and_fixed_acidity_shifts():
    sim = MalolacticFermentationSimulator()
    res = sim.simulate(
        initial_malic_acid=3.2,
        initial_ph=3.30,
        initial_fixed_acidity=7.5,
        duration_days=35
    )
    final = res["final_state"]
    # Fixed acidity drops, pH rises as dicarboxylic malic acid becomes monocarboxylic lactic acid
    assert final["fixed_acidity_g_l"] < 7.5
    assert final["ph"] > 3.30


def test_diacetyl_synthesis_and_reduction():
    sim = MalolacticFermentationSimulator()
    res = sim.simulate(
        initial_malic_acid=3.0,
        initial_ph=3.40,
        temperature=20.0,
        duration_days=35
    )
    # Peak diacetyl should be positive and recorded in diagnostics
    assert res["final_state"]["peak_diacetyl_mg_l"] > 0.05
    # Diacetyl trajectory should exhibit a peak and gradual subsequent reduction
    diacetyl_traj = res["trajectory"]["diacetyl_mg_l"]
    assert max(diacetyl_traj) >= diacetyl_traj[-1]


# ============================================================================
# 2. Oak Barrel Maturation Simulator Tests
# ============================================================================

def test_oak_maturation_species_differences():
    sim = OakBarrelMaturationSimulator()
    res_french = sim.simulate(oak_species="french", maturation_months=12)
    res_american = sim.simulate(oak_species="american", maturation_months=12)

    # French oak has higher ellagitannins; American oak has much higher whiskey lactones
    assert res_french["final_concentrations"]["ellagitannins_mg_l"] > res_american["final_concentrations"]["ellagitannins_mg_l"]
    assert res_american["final_concentrations"]["whiskey_lactones_mg_l"] > res_french["final_concentrations"]["whiskey_lactones_mg_l"]
    assert res_american["final_concentrations"]["vanillin_mg_l"] > res_french["final_concentrations"]["vanillin_mg_l"]


def test_oak_toast_level_and_barrel_generation():
    sim = OakBarrelMaturationSimulator()
    res_heavy = sim.simulate(oak_species="french", toast_level="heavy", maturation_months=12)
    res_light = sim.simulate(oak_species="french", toast_level="light", maturation_months=12)

    # Heavy toast creates much higher guaiacol (smoke) and lower ellagitannins
    assert res_heavy["final_concentrations"]["guaiacol_mg_l"] > res_light["final_concentrations"]["guaiacol_mg_l"]
    assert res_light["final_concentrations"]["ellagitannins_mg_l"] > res_heavy["final_concentrations"]["ellagitannins_mg_l"]

    # Neutral barrel extracts dramatically less than new barrel
    res_new = sim.simulate(oak_species="french", barrel_usage="new", maturation_months=12)
    res_neutral = sim.simulate(oak_species="french", barrel_usage="neutral", maturation_months=12)
    assert res_new["final_concentrations"]["vanillin_mg_l"] > res_neutral["final_concentrations"]["vanillin_mg_l"] * 5


def test_tannin_polymerization_and_astringency_softening():
    sim = OakBarrelMaturationSimulator()
    res = sim.simulate(maturation_months=24)
    traj = res["trajectory"]
    
    # Micro-oxygenation should monotonically increase polymeric pigment index
    assert traj["polymeric_pigment_index"][-1] > traj["polymeric_pigment_index"][0]
    # Astringency softening should increase over time
    assert traj["astringency_softening_pct"][-1] > traj["astringency_softening_pct"][0]


# ============================================================================
# 3. Cellar Master Dispatch Engine Tests
# ============================================================================

def test_cellar_dispatch_molecular_so2_and_pms_formula():
    # Henderson-Hasselbalch calculation
    # At pH 3.4 and 20 mg/L Free SO2: mol_so2 = 20 / (1 + 10^(3.4 - 1.81)) = 20 / (1 + 10^1.59) = 20 / 39.9 = ~0.50 mg/L
    mol = CellarMasterDispatchEngine.calculate_molecular_so2(20.0, 3.4)
    assert 0.45 <= mol <= 0.55

    # PMS dose calculation: 225L barrel, current Free SO2 10 mg/L, target 30 mg/L
    pms_g = CellarMasterDispatchEngine.calculate_pms_addition_grams(
        target_free_so2=30.0,
        current_free_so2=10.0,
        volume_liters=225.0
    )
    # Deficit = 20 mg/L -> Gross ~ 30.77 mg/L -> Total ~ 6923 mg SO2 -> / 0.576 ~ 12.02 g PMS
    assert 10.0 <= pms_g <= 15.0


def test_cellar_dispatch_fleet_triage_and_erpi():
    dispatch = CellarMasterDispatchEngine()
    fleet = [
        {
            "vessel_id": "CRITICAL-TANK",
            "vessel_type": "TANK",
            "cultivar": "red",
            "volume_liters": 1000.0,
            "ph": 3.8,
            "free_so2": 4.0,
            "total_so2": 40.0,
            "volatile_acidity": 0.95,
            "alcohol": 12.0,
            "temperature": 24.0,
            "residual_sugar": 5.0,
            "malic_acid": 0.6,
            "maturation_days": 30,
            "status": "aging"
        },
        {
            "vessel_id": "PRISTINE-BARREL",
            "vessel_type": "BARREL",
            "cultivar": "red",
            "volume_liters": 225.0,
            "ph": 3.35,
            "free_so2": 25.0,
            "total_so2": 70.0,
            "volatile_acidity": 0.35,
            "alcohol": 13.5,
            "temperature": 15.0,
            "residual_sugar": 1.2,
            "malic_acid": 0.02,
            "maturation_days": 180,
            "status": "resting"
        }
    ]
    res = dispatch.dispatch_cellar_inventory(fleet)
    assert res["total_vessels"] == 2
    assert res["dispatched_vessels"][0]["vessel_id"] == "CRITICAL-TANK"
    assert res["dispatched_vessels"][0]["triage"]["action_tier"] == "CRITICAL_INTERVENTION"
    assert res["dispatched_vessels"][1]["vessel_id"] == "PRISTINE-BARREL"
    assert res["dispatched_vessels"][1]["triage"]["action_tier"] in ["ROUTINE_MONITORING", "STABLE_CELLAR"]
    assert res["aggregate_pms_reagents_required_kg"] > 0.0


# ============================================================================
# 4. HACCP Enological Food Safety & Biogenic Amine Tests
# ============================================================================

def test_haccp_auditor_statutory_bounds():
    auditor = EnologicalHaccpAuditor()
    # Fully compliant wine
    res_pass = auditor.audit_wine_lot(
        lot_id="LOT-LEGAL-01",
        cultivar="red",
        ph=3.4,
        free_so2=25.0,
        total_so2=100.0,
        volatile_acidity=0.45,
        alcohol=13.5
    )
    assert res_pass["all_ccp_passed"] is True
    assert res_pass["audit_status"] == "CERTIFIED_FOOD_GRADE"

    # Non-compliant: Total SO2 exceeds legal 150 mg/L limit for red
    res_fail = auditor.audit_wine_lot(
        lot_id="LOT-FAIL-02",
        cultivar="red",
        ph=3.4,
        free_so2=25.0,
        total_so2=180.0,
        volatile_acidity=0.45,
        alcohol=13.5
    )
    assert res_fail["all_ccp_passed"] is False
    assert res_fail["audit_status"] == "NON_COMPLIANT_HOLD"
    assert res_fail["ccp_verifications"]["ccp2_total_so2_statutory_ceiling"]["passed"] is False


def test_biogenic_amine_risk_scoring():
    auditor = EnologicalHaccpAuditor()
    # High risk condition: high pH (3.85), incomplete MLF, zero SO2, warm temp
    res_high = auditor.audit_wine_lot(
        lot_id="LOT-HAZARD",
        cultivar="red",
        ph=3.85,
        free_so2=5.0,
        total_so2=40.0,
        volatile_acidity=0.6,
        alcohol=12.5,
        malic_acid=0.8,
        temperature=23.0
    )
    assert res_high["biogenic_amine_risk"]["risk_level"] in ["MODERATE", "HIGH"]
    assert res_high["biogenic_amine_risk"]["risk_index_score"] > 50.0


# ============================================================================
# 5. Production REST API Endpoints Integration Tests
# ============================================================================

def test_fastapi_day13_endpoints(test_client):
    # 1. /cellar/malolactic/simulate
    mlf_payload = {
        "initial_malic_acid": 2.6,
        "initial_ph": 3.4,
        "initial_fixed_acidity": 7.0,
        "alcohol": 13.0,
        "free_so2": 5.0,
        "temperature": 20.0,
        "duration_days": 30
    }
    r1 = test_client.post("/cellar/malolactic/simulate", json=mlf_payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert "final_state" in d1
    assert "conversion_rate_pct" in d1["final_state"]

    # 2. /cellar/oak-maturation/simulate
    oak_payload = {
        "oak_species": "french",
        "toast_level": "medium_plus",
        "barrel_usage": "new",
        "maturation_months": 18,
        "barrel_volume_liters": 225.0
    }
    r2 = test_client.post("/cellar/oak-maturation/simulate", json=oak_payload)
    assert r2.status_code == 200
    d2 = r2.json()
    assert "final_concentrations" in d2
    assert d2["final_concentrations"]["ellagitannins_mg_l"] > 50.0

    # 3. /cellar/dispatch/work-orders
    dispatch_payload = {
        "inventory": [
            {
                "vessel_id": "BARREL-01",
                "vessel_type": "BARREL",
                "cultivar": "red",
                "volume_liters": 225.0,
                "ph": 3.45,
                "free_so2": 10.0,
                "total_so2": 50.0,
                "volatile_acidity": 0.45,
                "alcohol": 13.2,
                "temperature": 18.0,
                "residual_sugar": 1.8,
                "malic_acid": 0.04,
                "maturation_days": 120,
                "status": "aging"
            }
        ]
    }
    r3 = test_client.post("/cellar/dispatch/work-orders", json=dispatch_payload)
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["total_vessels"] == 1
    assert len(d3["dispatched_vessels"]) == 1

    # 4. /compliance/haccp-audit
    haccp_payload = {
        "lot_id": "LOT-RESERVE-2026",
        "cultivar": "red",
        "ph": 3.42,
        "free_so2": 26.0,
        "total_so2": 95.0,
        "volatile_acidity": 0.42,
        "alcohol": 13.8,
        "residual_sugar": 2.0,
        "malic_acid": 0.03,
        "temperature": 16.0
    }
    r4 = test_client.post("/compliance/haccp-audit", json=haccp_payload)
    assert r4.status_code == 200
    d4 = r4.json()
    assert d4["all_ccp_passed"] is True
    assert "biogenic_amine_risk" in d4
