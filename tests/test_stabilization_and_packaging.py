"""
Unit and Integration Tests for Day 14 Tartrate & Protein Stabilization Kinetics,
Crossflow Membrane Microfiltration, Dissolved Gas Packaging Dynamics, and
Packaging Closure Shelf-Life Forecaster.
"""

import pytest
import numpy as np

from src.stabilization_and_packaging import (
    TartrateStabilizationSimulator,
    ProteinHazeAndBentoniteOptimizer,
    CrossflowFiltrationSimulator,
    DissolvedGasPackagingEngine,
    ClosureShelfLifeForecaster,
    BottlingLineQCAuditor,
    CLOSURE_PROFILES,
    STABILIZATION_ADDITIVES
)
from src.production_service import get_test_client


@pytest.fixture
def test_client():
    return get_test_client()


# ============================================================================
# 1. Tartrate Stabilization Tests
# ============================================================================

def test_bitartrate_dissociation_and_solubility_product():
    sim = TartrateStabilizationSimulator()

    # Bitartrate fraction peaks near pH 3.7
    f_30 = sim.calculate_bitartrate_fraction(3.0)
    f_37 = sim.calculate_bitartrate_fraction(3.7)
    f_44 = sim.calculate_bitartrate_fraction(4.4)

    assert f_37 > f_30
    assert f_37 > f_44
    assert 0.0 < f_37 <= 1.0

    # Solubility product decreases with lower temperature
    ksp_20c = sim.calculate_solubility_product(20.0, 12.0)
    ksp_0c = sim.calculate_solubility_product(0.0, 12.0)
    ksp_sub4c = sim.calculate_solubility_product(-4.0, 12.0)

    assert ksp_20c > ksp_0c > ksp_sub4c

    # Higher ethanol decreases K_sp (salting out effect)
    ksp_10pct = sim.calculate_solubility_product(14.0, 10.0)
    ksp_15pct = sim.calculate_solubility_product(14.0, 15.0)
    assert ksp_10pct > ksp_15pct


def test_tartrate_stability_evaluation_thresholds():
    sim = TartrateStabilizationSimulator()

    # Highly supersaturated specimen
    unstable_res = sim.evaluate_tartrate_stability(
        tartaric_acid_g_l=8.0,
        potassium_mg_l=1400.0,
        ph=3.65,
        alcohol_pct=14.0,
        storage_temp_c=14.0
    )
    assert unstable_res["stability_status"] == "UNSTABLE"
    assert unstable_res["action_required"] is True
    assert unstable_res["minicontact_delta_chi_us_cm"] > 55.0

    # Stable specimen (low potassium and moderate tartaric)
    stable_res = sim.evaluate_tartrate_stability(
        tartaric_acid_g_l=4.5,
        potassium_mg_l=500.0,
        ph=3.20,
        alcohol_pct=11.5,
        storage_temp_c=14.0
    )
    assert stable_res["stability_status"] == "STABLE"
    assert stable_res["action_required"] is False
    assert stable_res["minicontact_delta_chi_us_cm"] < 30.0


def test_cold_stabilization_kinetics_contact_process():
    sim = TartrateStabilizationSimulator()

    # Seeding should dramatically speed up crystallization compared to unseeded
    res_seeded = sim.simulate_cold_stabilization_kinetics(
        tartaric_acid_g_l=7.0,
        potassium_mg_l=1100.0,
        ph=3.55,
        alcohol_pct=13.0,
        chilling_temp_c=-4.0,
        seed_crystal_g_l=4.0,
        duration_hours=12
    )

    res_unseeded = sim.simulate_cold_stabilization_kinetics(
        tartaric_acid_g_l=7.0,
        potassium_mg_l=1100.0,
        ph=3.55,
        alcohol_pct=13.0,
        chilling_temp_c=-4.0,
        seed_crystal_g_l=0.0,
        duration_hours=12
    )

    final_seeded = res_seeded["final_state"]
    final_unseeded = res_unseeded["final_state"]

    assert final_seeded["total_kht_precipitated_g_l"] > final_unseeded["total_kht_precipitated_g_l"]
    assert final_seeded["potassium_mg_l"] < final_unseeded["potassium_mg_l"]
    assert final_seeded["stability_status"] in ["STABLE", "MODERATE_RISK"]


# ============================================================================
# 2. Protein Haze & Bentonite Fining Tests
# ============================================================================

def test_protein_haze_risk_evaluation_heat_test():
    opt = ProteinHazeAndBentoniteOptimizer()

    # High PR protein white wine
    haze_white = opt.evaluate_protein_haze_risk(
        protein_content_mg_l=150.0,
        ph=3.20,
        cultivar="white"
    )
    assert haze_white["is_stable"] is False
    assert haze_white["estimated_heat_delta_ntu"] > 10.0
    assert haze_white["risk_category"] == "SEVERE_HAZE_RISK"

    # Red wine naturally stabilized by tannins
    haze_red = opt.evaluate_protein_haze_risk(
        protein_content_mg_l=30.0,
        ph=3.60,
        phenolics_total_mg_l=1200.0,
        cultivar="red"
    )
    assert haze_red["estimated_heat_delta_ntu"] < haze_white["estimated_heat_delta_ntu"]


def test_langmuir_bentonite_dosage_optimization():
    opt = ProteinHazeAndBentoniteOptimizer()

    res = opt.optimize_bentonite_dosage(
        protein_content_mg_l=120.0,
        ph=3.25,
        cultivar="white",
        bentonite_type="sodium_bentonite"
    )

    assert res["status"] == "OPTIMIZATION_SUCCESSFUL"
    assert res["recommended_bentonite_g_hl"] > 20.0
    assert res["recommended_bentonite_g_hl"] <= 180.0
    assert res["heat_delta_ntu_final"] <= 1.8
    assert res["aroma_stripping_penalty_pct"] > 0.0

    # Test already stable wine
    stable_case = opt.optimize_bentonite_dosage(
        protein_content_mg_l=5.0,
        ph=3.5,
        cultivar="white"
    )
    assert stable_case["status"] == "ALREADY_STABLE"
    assert stable_case["recommended_bentonite_g_hl"] == 0.0


# ============================================================================
# 3. Crossflow Membrane Microfiltration Tests
# ============================================================================

def test_crossflow_filtration_flux_decay_and_turbidity():
    filt = CrossflowFiltrationSimulator(membrane_surface_area_m2=40.0)

    res = filt.simulate_crossflow_run(
        initial_turbidity_ntu=45.0,
        transmembrane_pressure_bar=1.6,
        target_volume_liters=10000.0,
        colloidal_index=1.2,
        backwash_interval_hours=2.0,
        run_duration_hours=6.0
    )

    summary = res["summary"]
    traj = res["trajectory"]

    assert summary["total_volume_filtered_l"] > 2000.0
    assert summary["final_permeate_turbidity_ntu"] < 0.60
    assert summary["clarity_status"] == "BRILLIANT_STERILE"
    assert len(traj["flux_lmh"]) > 5
    # Flux should generally decrease initially
    assert traj["flux_lmh"][1] >= traj["flux_lmh"][10]


def test_filterability_index_clogging_risk():
    filt = CrossflowFiltrationSimulator()

    low_colloid_run = filt.simulate_crossflow_run(colloidal_index=0.6, initial_turbidity_ntu=15.0)
    high_colloid_run = filt.simulate_crossflow_run(colloidal_index=2.8, initial_turbidity_ntu=90.0)

    fi_low = low_colloid_run["summary"]["filterability_index_s"]
    fi_high = high_colloid_run["summary"]["filterability_index_s"]

    assert fi_high > fi_low
    assert low_colloid_run["summary"]["clogging_risk"] == "LOW"
    assert high_colloid_run["summary"]["clogging_risk"] in ["MEDIUM", "HIGH"]


# ============================================================================
# 4. Dissolved Gas Packaging & Sparging Tests
# ============================================================================

def test_henry_gas_solubility_do_and_co2():
    eng = DissolvedGasPackagingEngine()

    # Oxygen solubility drops with higher temperature
    do_cold = eng.calculate_henry_oxygen_solubility(10.0, 12.0)
    do_warm = eng.calculate_henry_oxygen_solubility(25.0, 12.0)
    assert do_cold > do_warm
    assert 5.0 <= do_warm <= 12.0

    # CO2 solubility decreases with temperature
    co2_cold = eng.calculate_henry_co2_solubility(10.0, 1.0)
    co2_warm = eng.calculate_henry_co2_solubility(25.0, 1.0)
    assert co2_cold > co2_warm


def test_microbubble_sparging_deoxygenation():
    eng = DissolvedGasPackagingEngine()

    sparge_res = eng.simulate_nitrogen_sparging(
        initial_do_mg_l=1.60,
        target_do_mg_l=0.40,
        nitrogen_flow_ratio_nl_l=0.8,
        temperature_c=18.0
    )

    assert sparge_res["final_do_mg_l"] < 0.45
    assert sparge_res["oxygen_removed_mg_l"] > 1.0
    assert sparge_res["deoxygenation_efficiency_pct"] > 70.0
    assert sparge_res["target_reached"] is True


def test_total_packaging_oxygen_tpo_calculation():
    eng = DissolvedGasPackagingEngine()

    # Standard non-inerted bottling (20.9% air in headspace)
    tpo_air = eng.calculate_total_packaging_oxygen(
        dissolved_oxygen_wine_mg_l=0.8,
        headspace_volume_ml=7.0,
        bottle_volume_ml=750.0,
        headspace_o2_pct=20.95
    )

    # Vacuum flushed inerted bottling (1.5% O2 in headspace)
    tpo_inert = eng.calculate_total_packaging_oxygen(
        dissolved_oxygen_wine_mg_l=0.35,
        headspace_volume_ml=6.0,
        bottle_volume_ml=750.0,
        headspace_o2_pct=1.5
    )

    assert tpo_air["total_packaging_oxygen_tpo_mg_l"] > 2.0
    assert tpo_air["is_within_premium_spec"] is False
    assert tpo_inert["total_packaging_oxygen_tpo_mg_l"] < 0.85
    assert tpo_inert["is_within_premium_spec"] is True
    # 4 mg SO2 consumed per mg TPO
    assert abs(tpo_inert["stoichiometric_free_so2_depletion_mg_l"] - tpo_inert["total_packaging_oxygen_tpo_mg_l"] * 4.0) < 0.05


# ============================================================================
# 5. Packaging Closure Shelf-Life Tests
# ============================================================================

def test_closure_profiles_and_steady_otr():
    assert "natural_cork_fleur" in CLOSURE_PROFILES
    assert "screwcap_saran_tin" in CLOSURE_PROFILES
    assert "diam_5_microagglomerated" in CLOSURE_PROFILES

    # Screwcap Saran-Tin has lowest steady OTR
    otr_tin = CLOSURE_PROFILES["screwcap_saran_tin"]["steady_otr_mg_year"]
    otr_cork = CLOSURE_PROFILES["natural_cork_fleur"]["steady_otr_mg_year"]
    assert otr_tin < otr_cork


def test_bottle_shelf_life_so2_decay_and_premox():
    shelf = ClosureShelfLifeForecaster()

    res = shelf.simulate_bottle_shelf_life(
        closure_type="diam_5_microagglomerated",
        initial_free_so2=30.0,
        initial_ph=3.45,
        initial_tpo_mg_l=0.85,
        storage_temperature_c=14.0,
        duration_months=48
    )

    milestones = res["milestones"]
    final = res["final_state"]
    traj = res["trajectory"]

    assert final["free_so2_mg_l"] < 30.0
    assert final["a420_browning"] > 0.12
    assert len(traj["month"]) == 49
    assert milestones["peak_maturity_month"] > 0

    # Heat abuse at 28C should accelerate Free SO2 decay
    res_hot = shelf.simulate_bottle_shelf_life(
        closure_type="diam_5_microagglomerated",
        initial_free_so2=30.0,
        initial_ph=3.45,
        initial_tpo_mg_l=0.85,
        storage_temperature_c=28.0,
        duration_months=48
    )
    assert res_hot["final_state"]["free_so2_mg_l"] < final["free_so2_mg_l"]
    assert res_hot["final_state"]["a420_browning"] > final["a420_browning"]


# ============================================================================
# 6. Bottling Line SPC & Metrology Tests
# ============================================================================

def test_bottling_line_spc_audit_capability_and_metrology():
    qc = BottlingLineQCAuditor()

    # Good compliant batch (750 +/- 1.5 mL)
    good_vols = [749.8, 750.2, 750.5, 749.5, 750.1, 750.4, 749.9, 750.3]
    good_depths = [1.0, 1.1, 0.9, 1.05, 1.0, 1.15, 0.95, 1.0]
    good_vacs = [-0.25, -0.28, -0.24, -0.26, -0.27, -0.25, -0.26, -0.28]

    report = qc.audit_bottling_batch(
        lot_id="LOT_TEST_PERFECT",
        fill_volumes_ml=good_vols,
        closure_depths_mm=good_depths,
        headspace_vacuums_bar=good_vacs,
        filter_bubble_point_bar=3.45
    )

    assert report["is_batch_approved"] is True
    assert report["fill_volume_spc"]["defect_count"] == 0
    assert report["fill_volume_spc"]["cpk"] > 1.33
    assert report["audit_tier"] == "SIX_SIGMA_COMPLIANT"
    assert report["membrane_integrity"]["passed"] is True

    # Batch with failed bubble point (< 3.2 bar)
    failed_report = qc.audit_bottling_batch(
        lot_id="LOT_FAIL_FILTER",
        fill_volumes_ml=good_vols,
        closure_depths_mm=good_depths,
        headspace_vacuums_bar=good_vacs,
        filter_bubble_point_bar=2.85
    )
    assert failed_report["is_batch_approved"] is False
    assert failed_report["membrane_integrity"]["passed"] is False


# ============================================================================
# 7. FastAPI Day 14 Endpoints Integration Tests
# ============================================================================

def test_fastapi_day14_endpoints(test_client):
    # 1. /stabilization/evaluate-tartrate
    r1 = test_client.post("/stabilization/evaluate-tartrate", json={
        "tartaric_acid_g_l": 6.8,
        "potassium_mg_l": 1100.0,
        "ph": 3.45,
        "alcohol_pct": 13.0,
        "storage_temp_c": 14.0
    })
    assert r1.status_code == 200
    assert "minicontact_delta_chi_us_cm" in r1.json()

    # 2. /stabilization/simulate-cold
    r2 = test_client.post("/stabilization/simulate-cold", json={
        "tartaric_acid_g_l": 7.0,
        "potassium_mg_l": 1150.0,
        "ph": 3.55,
        "alcohol_pct": 13.5,
        "chilling_temp_c": -4.0,
        "seed_crystal_g_l": 4.0,
        "duration_hours": 8
    })
    assert r2.status_code == 200
    assert "total_kht_precipitated_g_l" in r2.json()["final_state"]

    # 3. /stabilization/optimize-bentonite
    r3 = test_client.post("/stabilization/optimize-bentonite", json={
        "protein_content_mg_l": 110.0,
        "ph": 3.25,
        "cultivar": "white",
        "bentonite_type": "sodium_bentonite"
    })
    assert r3.status_code == 200
    assert r3.json()["recommended_bentonite_g_hl"] > 0

    # 4. /filtration/crossflow-simulate
    r4 = test_client.post("/filtration/crossflow-simulate", json={
        "initial_turbidity_ntu": 35.0,
        "transmembrane_pressure_bar": 1.6,
        "target_volume_liters": 5000.0,
        "colloidal_index": 1.1,
        "backwash_interval_hours": 2.0,
        "run_duration_hours": 4.0
    })
    assert r4.status_code == 200
    assert r4.json()["summary"]["final_permeate_turbidity_ntu"] < 0.60

    # 5. /packaging/gas-sparging-simulate
    r5 = test_client.post("/packaging/gas-sparging-simulate", json={
        "initial_do_mg_l": 1.5,
        "target_do_mg_l": 0.40,
        "nitrogen_flow_ratio_nl_l": 0.6,
        "temperature_c": 18.0
    })
    assert r5.status_code == 200
    assert r5.json()["target_reached"] is True

    # 6. /packaging/calculate-tpo
    r6 = test_client.post("/packaging/calculate-tpo", json={
        "dissolved_oxygen_wine_mg_l": 0.40,
        "headspace_volume_ml": 6.0,
        "bottle_volume_ml": 750.0,
        "headspace_o2_pct": 1.8,
        "wine_temperature_c": 18.0
    })
    assert r6.status_code == 200
    assert r6.json()["total_packaging_oxygen_tpo_mg_l"] < 1.0

    # 7. /packaging/closure-shelf-life
    r7 = test_client.post("/packaging/closure-shelf-life", json={
        "closure_type": "diam_5_microagglomerated",
        "initial_free_so2": 32.0,
        "initial_ph": 3.45,
        "initial_tpo_mg_l": 0.85,
        "storage_temperature_c": 14.0,
        "duration_months": 36
    })
    assert r7.status_code == 200
    assert len(r7.json()["trajectory"]["month"]) == 37

    # 8. /packaging/bottling-qc-audit
    r8 = test_client.post("/packaging/bottling-qc-audit", json={
        "lot_id": "LOT_API_TEST",
        "fill_volumes_ml": [750.1, 749.9, 750.3, 750.0, 750.2],
        "closure_depths_mm": [1.0, 1.1, 0.95, 1.0, 1.05],
        "headspace_vacuums_bar": [-0.25, -0.27, -0.24, -0.26, -0.25],
        "filter_bubble_point_bar": 3.4
    })
    assert r8.status_code == 200
    assert r8.json()["is_batch_approved"] is True
