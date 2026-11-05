"""
Generates serialized enological benchmark JSON artifacts for Day 14:
- models/stabilization_benchmarks.json
- models/crossflow_filtration_benchmarks.json
- models/packaging_closure_benchmarks.json
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
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

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_stabilization_benchmarks():
    tartrate_sim = TartrateStabilizationSimulator()
    bentonite_opt = ProteinHazeAndBentoniteOptimizer()

    # 1. Tartrate evaluations across standard wine matrices
    sample_specimens = [
        {"lot": "LOT_RED_CAB_01", "cultivar": "red", "tartaric": 6.8, "potassium": 1150.0, "ph": 3.65, "alcohol": 14.0},
        {"lot": "LOT_RED_MERLOT_02", "cultivar": "red", "tartaric": 5.8, "potassium": 980.0, "ph": 3.52, "alcohol": 13.5},
        {"lot": "LOT_WHITE_CHARD_03", "cultivar": "white", "tartaric": 7.2, "potassium": 820.0, "ph": 3.28, "alcohol": 12.8},
        {"lot": "LOT_WHITE_SAUV_04", "cultivar": "white", "tartaric": 8.1, "potassium": 920.0, "ph": 3.15, "alcohol": 12.2},
        {"lot": "LOT_ROSE_PINOT_05", "cultivar": "white", "tartaric": 6.2, "potassium": 750.0, "ph": 3.32, "alcohol": 12.5},
    ]

    tartrate_evals = []
    for s in sample_specimens:
        ev = tartrate_sim.evaluate_tartrate_stability(
            tartaric_acid_g_l=s["tartaric"],
            potassium_mg_l=s["potassium"],
            ph=s["ph"],
            alcohol_pct=s["alcohol"],
            storage_temp_c=14.0
        )
        ev["lot"] = s["lot"]
        ev["cultivar"] = s["cultivar"]
        tartrate_evals.append(ev)

    # 2. Dynamic contact cold stabilization simulation
    cold_sim_result = tartrate_sim.simulate_cold_stabilization_kinetics(
        tartaric_acid_g_l=7.2,
        potassium_mg_l=1100.0,
        ph=3.55,
        alcohol_pct=13.5,
        chilling_temp_c=-4.0,
        seed_crystal_g_l=4.0,
        duration_hours=12
    )

    # 3. Protein haze and bentonite fining benchmarks
    protein_specimens = [
        {"lot": "LOT_WHITE_SAUV_04", "cultivar": "white", "protein": 120.0, "ph": 3.15},
        {"lot": "LOT_WHITE_CHARD_03", "cultivar": "white", "protein": 85.0, "ph": 3.28},
        {"lot": "LOT_WHITE_RIES_06", "cultivar": "white", "protein": 190.0, "ph": 3.05},
        {"lot": "LOT_RED_CAB_01", "cultivar": "red", "protein": 45.0, "ph": 3.65},
    ]

    fining_benchmarks = []
    for p in protein_specimens:
        dose_res = bentonite_opt.optimize_bentonite_dosage(
            protein_content_mg_l=p["protein"],
            ph=p["ph"],
            cultivar=p["cultivar"],
            bentonite_type="sodium_bentonite"
        )
        dose_res["lot"] = p["lot"]
        fining_benchmarks.append(dose_res)

    stabilization_payload = {
        "title": "Day 14 Tartrate and Protein Stabilization Benchmarks",
        "reference_additives": STABILIZATION_ADDITIVES,
        "sample_tartrate_evaluations": tartrate_evals,
        "cold_stabilization_simulation": cold_sim_result,
        "bentonite_fining_benchmarks": fining_benchmarks
    }

    out_file = MODELS_DIR / "stabilization_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stabilization_payload, f, indent=2)
    print(f"Generated {out_file}")


def generate_filtration_benchmarks():
    filt_sim = CrossflowFiltrationSimulator(membrane_surface_area_m2=40.0)

    # Run simulations under 3 colloidal loading conditions
    scenarios = [
        {"name": "Low Turbidity White Wine", "initial_ntu": 18.0, "tmp_bar": 1.4, "colloidal_index": 0.8},
        {"name": "Medium Turbidity Red Wine (Racked)", "initial_ntu": 42.0, "tmp_bar": 1.6, "colloidal_index": 1.25},
        {"name": "High Colloid Press Wine", "initial_ntu": 95.0, "tmp_bar": 2.0, "colloidal_index": 2.1},
    ]

    filtration_results = []
    for sc in scenarios:
        res = filt_sim.simulate_crossflow_run(
            initial_turbidity_ntu=sc["initial_ntu"],
            transmembrane_pressure_bar=sc["tmp_bar"],
            target_volume_liters=10000.0,
            colloidal_index=sc["colloidal_index"],
            backwash_interval_hours=2.0,
            run_duration_hours=6.0
        )
        filtration_results.append({
            "scenario": sc["name"],
            "parameters": res["parameters"],
            "summary": res["summary"],
            "trajectory_sample": {
                "hour": res["trajectory"]["hour"][::2],
                "flux_lmh": res["trajectory"]["flux_lmh"][::2],
                "cumulative_volume_l": res["trajectory"]["cumulative_volume_l"][::2],
                "permeate_ntu": res["trajectory"]["permeate_ntu"][::2]
            }
        })

    filt_payload = {
        "title": "Day 14 Crossflow Membrane Microfiltration Benchmarks",
        "membrane_specifications": {
            "pore_size_um": 0.45,
            "membrane_material": "Polyethersulfone (PES) hollow fiber",
            "standard_surface_area_m2": 40.0,
            "bubble_point_integrity_min_bar": 3.20
        },
        "scenarios": filtration_results
    }

    out_file = MODELS_DIR / "crossflow_filtration_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(filt_payload, f, indent=2)
    print(f"Generated {out_file}")


def generate_packaging_benchmarks():
    gas_eng = DissolvedGasPackagingEngine()
    shelf_sim = ClosureShelfLifeForecaster()
    qc_auditor = BottlingLineQCAuditor()

    # 1. Total Packaging Oxygen (TPO) benchmarks
    tpo_cases = [
        {"name": "Non-inerted Standard Bottling", "do_wine": 1.2, "hs_vol": 7.0, "hs_o2": 20.9},
        {"name": "Nitrogen Sparged + Vacuum Corker", "do_wine": 0.45, "hs_vol": 6.0, "hs_o2": 1.8},
        {"name": "Ultra-Low Oxygen Red Wine Packaging", "do_wine": 0.28, "hs_vol": 5.0, "hs_o2": 0.8},
    ]
    tpo_benchmarks = []
    for c in tpo_cases:
        tpo_res = gas_eng.calculate_total_packaging_oxygen(
            dissolved_oxygen_wine_mg_l=c["do_wine"],
            headspace_volume_ml=c["hs_vol"],
            headspace_o2_pct=c["hs_o2"]
        )
        tpo_res["case_name"] = c["name"]
        tpo_benchmarks.append(tpo_res)

    # 2. Multi-closure shelf-life simulation comparison
    closure_benchmarks = {}
    for c_key in CLOSURE_PROFILES.keys():
        sim_res = shelf_sim.simulate_bottle_shelf_life(
            closure_type=c_key,
            initial_free_so2=32.0,
            initial_ph=3.45,
            initial_tpo_mg_l=0.85,
            storage_temperature_c=14.0,
            duration_months=48
        )
        closure_benchmarks[c_key] = {
            "closure_name": sim_res["closure_info"]["name"],
            "steady_otr_mg_year": sim_res["closure_info"]["steady_otr_mg_year"],
            "milestones": sim_res["milestones"],
            "final_state": sim_res["final_state"],
            "trajectory_summary": {
                "month": sim_res["trajectory"]["month"][::4],
                "free_so2_mg_l": sim_res["trajectory"]["free_so2_mg_l"][::4],
                "a420_browning": sim_res["trajectory"]["a420_browning"][::4],
                "freshness_score": sim_res["trajectory"]["freshness_score"][::4]
            }
        }

    # 3. Bottling Line SPC Audit Benchmark
    np.random.seed(42)
    # 25 production samples
    sample_vols = [float(round(v, 2)) for v in np.random.normal(750.4, 1.8, 25)]
    sample_depths = [float(round(d, 2)) for d in np.random.normal(1.1, 0.15, 25)]
    sample_vacuums = [float(round(vac, 3)) for vac in np.random.normal(-0.28, 0.04, 25)]

    spc_audit_res = qc_auditor.audit_bottling_batch(
        lot_id="LOT_ESTATE_CAB_B2026_01",
        fill_volumes_ml=sample_vols,
        closure_depths_mm=sample_depths,
        headspace_vacuums_bar=sample_vacuums,
        filter_bubble_point_bar=3.45
    )

    packaging_payload = {
        "title": "Day 14 Dissolved Gas, Closure Permeation and Bottling QC Benchmarks",
        "closure_specifications": CLOSURE_PROFILES,
        "tpo_benchmarks": tpo_benchmarks,
        "closure_shelf_life_benchmarks": closure_benchmarks,
        "sample_spc_bottling_audit": spc_audit_res
    }

    out_file = MODELS_DIR / "packaging_closure_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(packaging_payload, f, indent=2)
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_stabilization_benchmarks()
    generate_filtration_benchmarks()
    generate_packaging_benchmarks()
