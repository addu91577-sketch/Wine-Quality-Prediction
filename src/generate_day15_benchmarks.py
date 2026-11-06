"""
Generates serialized enological benchmark JSON artifacts for Day 15:
- models/cold_chain_logistics_benchmarks.json
- models/terroir_authenticity_benchmarks.json
- models/consumer_hedonic_market_benchmarks.json
- models/digital_winery_twin_benchmarks.json
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import numpy as np

from src.transport_terroir_hedonic_and_twin import (
    ColdChainLogisticsSimulator,
    IsotopicTerroirAndFraudAuditor,
    ConsumerHedonicAndMarketPricingOptimizer,
    UnifiedDigitalWineryTwinPlatform,
    TERROIR_PROFILES,
    REGIONAL_PALATE_PROFILES,
    CLOSURE_PRESSURE_TOLERANCES
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_cold_chain_benchmarks():
    sim = ColdChainLogisticsSimulator()
    benchmarks = {
        "metadata": {
            "title": "Cold-Chain Intermodal Transportation Benchmarks",
            "version": "1.0",
            "day": 15,
            "closure_pressure_tolerances": CLOSURE_PRESSURE_TOLERANCES
        },
        "routes": {}
    }

    closure_types = ["natural_cork_fleur", "technical_diam", "synthetic_nomacorc", "screwcap_sarantin"]

    for preset_name, preset_info in sim.transit_presets.items():
        benchmarks["routes"][preset_name] = {
            "info": preset_info,
            "closures_evaluated": {}
        }
        for closure in closure_types:
            res = sim.evaluate_transit_scenario(
                preset_name=preset_name,
                closure_type=closure,
                initial_free_so2=35.0,
                initial_a420_browning=0.115,
                dissolved_co2_mg_l=820.0
            )
            benchmarks["routes"][preset_name]["closures_evaluated"][closure] = res

    out_path = MODELS_DIR / "cold_chain_logistics_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Generated: {out_path}")
    return benchmarks


def generate_terroir_benchmarks():
    auditor = IsotopicTerroirAndFraudAuditor()
    benchmarks = {
        "metadata": {
            "title": "Isotopic Terroir & Anti-Counterfeit Forensics Benchmarks",
            "version": "1.0",
            "day": 15,
            "terroirs_registered": list(TERROIR_PROFILES.keys())
        },
        "reference_profiles": TERROIR_PROFILES,
        "sample_verifications": {},
        "chaptalization_calibration_curve": [],
        "water_dilution_calibration_curve": []
    }

    # Chaptalization sweep
    for delta13c in [-27.5, -26.5, -25.5, -24.0, -22.5, -20.0, -18.0, -15.0]:
        benchmarks["chaptalization_calibration_curve"].append(
            auditor.calculate_chaptalization_fraction(delta13c)
        )

    # Water dilution sweep
    for delta18o in [4.5, 3.8, 2.5, 1.2, 0.0, -2.0, -4.5, -6.0]:
        benchmarks["water_dilution_calibration_curve"].append(
            auditor.calculate_water_dilution_fraction(delta18o)
        )

    # Appellation tests: Authentic sample, Foreign terroir counterfeit, Chaptalized sample, Watered sample
    for app_id, prof in TERROIR_PROFILES.items():
        ref_min = prof["minerals_ppm"]
        # 1. Authentic specimen
        sample_auth = {k: v * float(np.random.uniform(0.97, 1.03)) for k, v in ref_min.items()}
        res_auth = auditor.verify_appellation_terroir(
            claimed_appellation=app_id,
            sample_minerals=sample_auth,
            measured_sr_ratio=prof["sr_87_86_ratio"] + 0.0001,
            measured_delta13c=-27.2,
            measured_delta18o=3.7
        )

        # 2. Counterfeit specimen (claiming app_id, but actually minerals of another region)
        other_id = [x for x in TERROIR_PROFILES.keys() if x != app_id][0]
        other_prof = TERROIR_PROFILES[other_id]
        res_fraud = auditor.verify_appellation_terroir(
            claimed_appellation=app_id,
            sample_minerals=other_prof["minerals_ppm"],
            measured_sr_ratio=other_prof["sr_87_86_ratio"],
            measured_delta13c=-27.0,
            measured_delta18o=3.5
        )

        benchmarks["sample_verifications"][app_id] = {
            "authentic_test": res_auth,
            "counterfeit_test": res_fraud
        }

    out_path = MODELS_DIR / "terroir_authenticity_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Generated: {out_path}")
    return benchmarks


def generate_consumer_hedonic_benchmarks():
    optimizer = ConsumerHedonicAndMarketPricingOptimizer()
    benchmarks = {
        "metadata": {
            "title": "Consumer Hedonic Econometrics & Regional Palate Benchmarks",
            "version": "1.0",
            "day": 15,
            "regions": list(REGIONAL_PALATE_PROFILES.keys())
        },
        "hedonic_pricing_curve": [],
        "regional_palate_matrices": {},
        "margin_optimization_scenarios": []
    }

    # Quality vs Price curve across tiers
    for q in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]:
        for tier in ["table", "igt", "doc_quality", "grand_cru"]:
            res = optimizer.estimate_bottle_market_value(
                predicted_quality=q,
                appellation_tier=tier,
                alcohol=13.5,
                cellar_aging_years=2.0
            )
            benchmarks["hedonic_pricing_curve"].append(res)

    # Palate alignment for classic wine profiles
    specimens = {
        "bold_california_cabernet": {"alcohol": 14.5, "ph": 3.72, "residual_sugar": 3.8, "volatile_acidity": 0.44, "sulphates": 0.75, "oak_index": 0.95},
        "crisp_burgundy_pinot": {"alcohol": 12.8, "ph": 3.32, "residual_sugar": 1.2, "volatile_acidity": 0.46, "sulphates": 0.58, "oak_index": 0.65},
        "delicate_japanese_koshu_style": {"alcohol": 12.0, "ph": 3.25, "residual_sugar": 2.0, "volatile_acidity": 0.35, "sulphates": 0.50, "oak_index": 0.20},
        "commercial_mass_market_red": {"alcohol": 13.0, "ph": 3.55, "residual_sugar": 6.5, "volatile_acidity": 0.52, "sulphates": 0.62, "oak_index": 0.50}
    }

    for name, params in specimens.items():
        benchmarks["regional_palate_matrices"][name] = optimizer.calculate_regional_palate_alignment(**params)

    # Margin optimization scenarios
    for cost in [4.50, 8.00, 14.50, 28.00]:
        val = cost * 2.8
        for elast in [-2.4, -1.25, -0.65]:
            opt = optimizer.optimize_release_pricing(
                production_cost_per_bottle_usd=cost,
                market_valuation_usd=val,
                price_elasticity=elast
            )
            benchmarks["margin_optimization_scenarios"].append(opt)

    out_path = MODELS_DIR / "consumer_hedonic_market_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Generated: {out_path}")
    return benchmarks


def generate_digital_winery_twin_benchmarks():
    twin = UnifiedDigitalWineryTwinPlatform()
    benchmarks = {
        "metadata": {
            "title": "Unified Digital Winery Twin & ESG Life Cycle Benchmarks",
            "version": "1.0",
            "day": 15
        },
        "lifecycle_simulations": {},
        "esg_lca_comparisons": {}
    }

    scenarios = [
        {
            "id": "lot_bordeaux_grand_vin",
            "cultivar": "red",
            "harvest_sugar_brix": 25.2,
            "must_tartaric_acid": 6.9,
            "must_ph": 3.48,
            "oak_regime": "french_medium_plus",
            "oak_duration_months": 18,
            "closure_choice": "technical_diam",
            "shipping_route": "maritime_reefer_rotterdam_singapore",
            "appellation_claimed": "bordeaux_left_bank",
            "production_lot_bottles": 20000
        },
        {
            "id": "lot_napa_valley_reserve",
            "cultivar": "red",
            "harvest_sugar_brix": 26.0,
            "must_tartaric_acid": 6.2,
            "must_ph": 3.65,
            "oak_regime": "american_heavy",
            "oak_duration_months": 14,
            "closure_choice": "natural_cork_fleur",
            "shipping_route": "cross_country_truck_express",
            "appellation_claimed": "napa_valley_rutherford",
            "production_lot_bottles": 15000
        },
        {
            "id": "lot_burgundy_village_air_express",
            "cultivar": "white",
            "harvest_sugar_brix": 23.5,
            "must_tartaric_acid": 7.4,
            "must_ph": 3.25,
            "oak_regime": "french_light",
            "oak_duration_months": 8,
            "closure_choice": "screwcap_sarantin",
            "shipping_route": "air_freight_cargo_london_tokyo",
            "appellation_claimed": "burgundy_cote_dor",
            "production_lot_bottles": 8000
        }
    ]

    for sc in scenarios:
        sc_id = sc.pop("id")
        sim_res = twin.simulate_end_to_end_winery_lifecycle(**sc)
        benchmarks["lifecycle_simulations"][sc_id] = sim_res

    # ESG LCA comparisons across glass weights and transport modes
    glass_weights = [380.0, 450.0, 580.0, 720.0]
    for gw in glass_weights:
        for mode in ["maritime", "truck", "air_cargo"]:
            res = twin.calculate_sustainability_lca(
                bottle_glass_weight_g=gw,
                freight_mode=mode,
                transit_distance_km=8000.0
            )
            key = f"glass_{int(gw)}g_{mode}"
            benchmarks["esg_lca_comparisons"][key] = res

    out_path = MODELS_DIR / "digital_winery_twin_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Generated: {out_path}")
    return benchmarks


def main():
    print("=" * 70)
    print("DAY 15 BENCHMARK GENERATION PIPELINE")
    print("=" * 70)
    generate_cold_chain_benchmarks()
    generate_terroir_benchmarks()
    generate_consumer_hedonic_benchmarks()
    generate_digital_winery_twin_benchmarks()
    print("All Day 15 benchmark JSON models serialized successfully.")


if __name__ == "__main__":
    main()
