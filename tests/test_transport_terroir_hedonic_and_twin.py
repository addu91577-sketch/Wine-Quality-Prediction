"""
Unit and Integration Tests for Day 15 Cold-Chain Logistics, Isotopic Terroir Fraud Detection,
Consumer Hedonic Econometrics, and Unified Digital Winery Twin Platform.
"""

import pytest
import numpy as np
import math

from src.transport_terroir_hedonic_and_twin import (
    ColdChainLogisticsSimulator,
    IsotopicTerroirAndFraudAuditor,
    ConsumerHedonicAndMarketPricingOptimizer,
    UnifiedDigitalWineryTwinPlatform,
    TERROIR_PROFILES,
    REGIONAL_PALATE_PROFILES,
    CLOSURE_PRESSURE_TOLERANCES,
    P_ATM_SEA_LEVEL,
    P_ATM_AIR_CARGO_CRUISE,
    P_ATM_MOUNTAIN_PASS
)
from src.production_service import get_test_client


@pytest.fixture
def test_client():
    return get_test_client()


# ============================================================================
# 1. Cold-Chain Transportation Stress Tests
# ============================================================================

def test_cold_chain_thermal_trajectory_and_arrhenius_tdu():
    sim = ColdChainLogisticsSimulator()

    # Reefer (14 C constant) vs Hot dry container (32 C with amplitude)
    reefer_df = sim.simulate_thermal_trajectory(duration_days=10, mean_temp_c=14.0, diurnal_amplitude_c=0.5)
    dry_df = sim.simulate_thermal_trajectory(duration_days=10, mean_temp_c=32.0, diurnal_amplitude_c=8.0)

    assert len(reefer_df) == 240
    assert len(dry_df) == 240

    # Hot container must accumulate dramatically higher TDU hours
    reefer_tdu = reefer_df["cumulative_tdu_hours"].iloc[-1]
    dry_tdu = dry_df["cumulative_tdu_hours"].iloc[-1]
    assert dry_tdu > reefer_tdu * 5.0

    # At 14 C (reference), rate should be approximately 1.0
    assert abs(reefer_df["rel_rate_ester"].mean() - 1.0) < 0.15


def test_headspace_pressure_differential_and_temperature_expansion():
    sim = ColdChainLogisticsSimulator()

    # Differential pressure increases with temperature due to wine expansion and CO2 degassing
    p_15c = sim.calculate_headspace_pressure_differential(liquid_temp_c=15.0, ambient_pressure_kpa=P_ATM_SEA_LEVEL)
    p_30c = sim.calculate_headspace_pressure_differential(liquid_temp_c=30.0, ambient_pressure_kpa=P_ATM_SEA_LEVEL)
    p_42c = sim.calculate_headspace_pressure_differential(liquid_temp_c=42.0, ambient_pressure_kpa=P_ATM_SEA_LEVEL)

    assert p_42c["delta_p_differential_kpa"] > p_30c["delta_p_differential_kpa"] > p_15c["delta_p_differential_kpa"]
    assert p_42c["effective_headspace_ml"] < p_15c["effective_headspace_ml"]


def test_barometric_altitude_pressure_drop_and_cork_extrusion():
    sim = ColdChainLogisticsSimulator()

    # In unpressurized or cargo hold air freight (75.2 kPa), Delta P spikes
    sea_res = sim.calculate_headspace_pressure_differential(liquid_temp_c=30.0, ambient_pressure_kpa=P_ATM_SEA_LEVEL)
    air_res = sim.calculate_headspace_pressure_differential(liquid_temp_c=30.0, ambient_pressure_kpa=P_ATM_AIR_CARGO_CRUISE)

    diff = air_res["delta_p_differential_kpa"] - sea_res["delta_p_differential_kpa"]
    # Should equal approximately the difference between sea level and cargo hold (101.3 - 75.2 = 26.1 kPa)
    assert abs(diff - (P_ATM_SEA_LEVEL - P_ATM_AIR_CARGO_CRUISE)) < 1.0
    assert air_res["delta_p_differential_kpa"] > 60.0


def test_intermodal_transit_scenarios_and_closure_seal_risk():
    sim = ColdChainLogisticsSimulator()

    reefer_eval = sim.evaluate_transit_scenario(
        preset_name="maritime_reefer_rotterdam_singapore",
        closure_type="natural_cork_fleur"
    )
    dry_eval = sim.evaluate_transit_scenario(
        preset_name="dry_container_deck_panama",
        closure_type="natural_cork_fleur"
    )

    assert reefer_eval["barometric_seal_integrity"]["seal_status"] == "PASS_OPTIMAL_COLD_CHAIN"
    assert reefer_eval["barometric_seal_integrity"]["overall_transit_risk_score"] < 25

    # Dry container on deck must flag critical thermal degradation or leakage
    assert dry_eval["barometric_seal_integrity"]["overall_transit_risk_score"] >= 65
    assert dry_eval["chemical_impact"]["so2_loss_mg_l"] > reefer_eval["chemical_impact"]["so2_loss_mg_l"]


# ============================================================================
# 2. Isotopic Terroir & Fraud Auditor Tests
# ============================================================================

def test_chaptalization_detection_carbon_isotope_mass_balance():
    auditor = IsotopicTerroirAndFraudAuditor()

    # Authentic pure C3 grape (typically -27.2 per mil)
    res_auth = auditor.calculate_chaptalization_fraction(-27.2)
    assert not res_auth["is_adulterated"]
    assert res_auth["estimated_c4_sugar_fraction_pct"] == 0.0
    assert res_auth["compliance_verdict"] == "LEGAL_AUTHENTIC"

    # Heavily adulterated with cane sugar (-18.0 per mil)
    res_fraud = auditor.calculate_chaptalization_fraction(-18.0)
    assert res_fraud["is_adulterated"]
    assert res_fraud["estimated_c4_sugar_fraction_pct"] > 50.0
    assert "SEVERE" in res_fraud["compliance_verdict"]


def test_water_dilution_detection_oxygen_isotope_fraction():
    auditor = IsotopicTerroirAndFraudAuditor()

    # Natural grape water transpiration (+3.8 per mil)
    res_auth = auditor.calculate_water_dilution_fraction(3.8)
    assert not res_auth["is_diluted"]
    assert res_auth["estimated_exogenous_water_pct"] == 0.0

    # Diluted with tap water (-4.5 per mil)
    res_diluted = auditor.calculate_water_dilution_fraction(-4.5)
    assert res_diluted["is_diluted"]
    assert res_diluted["estimated_exogenous_water_pct"] > 50.0


def test_appellation_terroir_verification_and_counterfeiting():
    auditor = IsotopicTerroirAndFraudAuditor()

    bordeaux_prof = TERROIR_PROFILES["bordeaux_left_bank"]
    sample_auth = bordeaux_prof["minerals_ppm"].copy()

    # Authentic sample matches profile
    res_auth = auditor.verify_appellation_terroir(
        claimed_appellation="bordeaux_left_bank",
        sample_minerals=sample_auth,
        measured_sr_ratio=bordeaux_prof["sr_87_86_ratio"],
        measured_delta13c=-27.2,
        measured_delta18o=3.8
    )
    assert res_auth["is_authentic"]
    assert res_auth["verdict"] == "VERIFIED_AUTHENTIC_TERROIR"
    assert res_auth["mahalanobis_terroir_distance"] < 1.5

    # Counterfeit: Napa Valley claiming to be Bordeaux
    napa_prof = TERROIR_PROFILES["napa_valley_rutherford"]
    res_counterfeit = auditor.verify_appellation_terroir(
        claimed_appellation="bordeaux_left_bank",
        sample_minerals=napa_prof["minerals_ppm"],
        measured_sr_ratio=napa_prof["sr_87_86_ratio"],
        measured_delta13c=-27.0,
        measured_delta18o=3.5
    )
    assert not res_counterfeit["is_authentic"]
    assert res_counterfeit["mahalanobis_terroir_distance"] > 3.0


# ============================================================================
# 3. Consumer Hedonic Econometrics & Market Optimization Tests
# ============================================================================

def test_hedonic_pricing_valuation_and_confidence_intervals():
    opt = ConsumerHedonicAndMarketPricingOptimizer()

    val_low = opt.estimate_bottle_market_value(predicted_quality=5.0, appellation_tier="table")
    val_high = opt.estimate_bottle_market_value(predicted_quality=8.5, appellation_tier="grand_cru")

    assert val_high["estimated_market_price_usd"] > val_low["estimated_market_price_usd"] * 5.0
    ci_low, ci_high = val_high["confidence_interval_90_usd"]
    assert ci_low < val_high["estimated_market_price_usd"] < ci_high


def test_cross_regional_consumer_palate_alignment():
    opt = ConsumerHedonicAndMarketPricingOptimizer()

    # Bold oaky high alcohol red (fits North America best)
    bold_red = opt.calculate_regional_palate_alignment(
        alcohol=14.5,
        ph=3.70,
        residual_sugar=3.5,
        volatile_acidity=0.42,
        sulphates=0.72,
        oak_index=0.90
    )

    assert bold_red["north_america"]["palate_alignment_score"] > 80.0
    assert bold_red["north_america"]["palate_alignment_score"] > bold_red["continental_europe"]["palate_alignment_score"]


def test_price_elasticity_and_margin_optimization():
    opt = ConsumerHedonicAndMarketPricingOptimizer()

    res = opt.optimize_release_pricing(
        production_cost_per_bottle_usd=10.0,
        market_valuation_usd=32.0,
        price_elasticity=-1.25,
        total_bottles_inventory=10000
    )

    assert res["optimal_release_price_usd"] > res["production_cost_usd"]
    assert res["gross_margin_pct"] > 50.0
    assert res["projected_gross_profit_usd"] > 0


# ============================================================================
# 4. Unified Digital Winery Twin & ESG Life Cycle Tests
# ============================================================================

def test_unified_digital_winery_twin_end_to_end():
    twin = UnifiedDigitalWineryTwinPlatform()

    sim = twin.simulate_end_to_end_winery_lifecycle(
        cultivar="red",
        harvest_sugar_brix=25.0,
        must_tartaric_acid=6.5,
        must_ph=3.50,
        oak_regime="french_medium_plus",
        oak_duration_months=12,
        production_lot_bottles=15000
    )

    assert sim["stage_1_fermentation"]["fermented_alcohol_pct"] >= 14.0
    assert sim["stage_4_quality_scoring"]["predicted_quality_score"] > 6.5
    assert "market_valuation" in sim["stage_7_hedonic_and_pricing"]
    assert "total_footprint_g_co2e_per_bottle" in sim["stage_8_esg_lca_sustainability"]


def test_sustainability_lca_carbon_footprint_and_glass_lightweighting():
    twin = UnifiedDigitalWineryTwinPlatform()

    # Heavy bottle (750g) vs Eco-lightweight (380g)
    heavy = twin.calculate_sustainability_lca(bottle_glass_weight_g=750.0)
    light = twin.calculate_sustainability_lca(bottle_glass_weight_g=380.0)

    # Glass makes up a huge part of Scope 3
    assert heavy["total_footprint_g_co2e_per_bottle"] > light["total_footprint_g_co2e_per_bottle"] + 250.0
    assert light["glass_bottle_contribution_pct"] < heavy["glass_bottle_contribution_pct"]


# ============================================================================
# 5. FastAPI Day 15 REST Endpoints Integration Tests
# ============================================================================

def test_fastapi_day15_endpoints(test_client):
    # 1. Transit simulation
    resp = test_client.post("/logistics/simulate-transit", json={
        "preset_name": "maritime_reefer_rotterdam_singapore",
        "closure_type": "technical_diam"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["transit_mode"] == "maritime_reefer"
    assert "total_tdu_hours" in data["kinetic_stress_metrics"]

    # 2. Barometric seal assessment
    resp = test_client.post("/logistics/assess-barometric-seal", json={
        "liquid_temp_c": 35.0,
        "ambient_pressure_kpa": 75.2
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["delta_p_differential_kpa"] > 40.0

    # 3. Terroir verification
    bordeaux = TERROIR_PROFILES["bordeaux_left_bank"]
    resp = test_client.post("/authenticity/verify-terroir", json={
        "claimed_appellation": "bordeaux_left_bank",
        "sample_minerals": bordeaux["minerals_ppm"],
        "measured_sr_ratio": bordeaux["sr_87_86_ratio"],
        "measured_delta13c": -27.2,
        "measured_delta18o": 3.8
    })
    assert resp.status_code == 200
    assert resp.json()["is_authentic"] is True

    # 4. Chaptalization detection
    resp = test_client.post("/authenticity/detect-chaptalization", json={
        "measured_delta13c": -20.5
    })
    assert resp.status_code == 200
    assert resp.json()["is_adulterated"] is True

    # 5. Hedonic valuation
    resp = test_client.post("/market/hedonic-valuation", json={
        "predicted_quality": 7.8,
        "appellation_tier": "doc_quality"
    })
    assert resp.status_code == 200
    assert resp.json()["estimated_market_price_usd"] > 20.0

    # 6. Regional palate alignment
    resp = test_client.post("/market/regional-palate-alignment", json={
        "alcohol": 14.2,
        "ph": 3.65,
        "residual_sugar": 3.0,
        "volatile_acidity": 0.42,
        "sulphates": 0.70
    })
    assert resp.status_code == 200
    assert "north_america" in resp.json()["regions"]

    # 7. End-to-end winery twin
    resp = test_client.post("/winery-twin/simulate-end-to-end", json={
        "cultivar": "red",
        "harvest_sugar_brix": 25.0,
        "production_lot_bottles": 10000
    })
    assert resp.status_code == 200
    assert "simulation_id" in resp.json()

    # 8. Sustainability LCA
    resp = test_client.post("/sustainability/lca-carbon-footprint", json={
        "bottle_glass_weight_g": 410.0,
        "is_organic_certified": True
    })
    assert resp.status_code == 200
    assert resp.json()["total_footprint_g_co2e_per_bottle"] > 0
