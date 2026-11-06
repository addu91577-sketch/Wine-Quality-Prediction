"""
Day 15 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day15_Cold_Chain_Terroir_Authenticity_and_Digital_Twin.ipynb
and executes all cells to populate verified outputs: intermodal cold-chain thermal
excursions and Arrhenius TDU kinetics, high-altitude barometric headspace Delta P
and closure seal integrity, carbon and oxygen stable isotope ratio forensics (delta13C, delta18O),
bedrock strontium 87Sr/86Sr and ICP-MS multielemental Mahalanobis terroir verification,
Rosen hedonic pricing econometric model, cross-regional palate alignment, and the
complete 15-stage unified digital winery twin with Scope 1/2/3 ESG Life Cycle Assessment.
"""

import os
import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(".").resolve()
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Day15_Cold_Chain_Terroir_Authenticity_and_Digital_Twin.ipynb"


def create_day15_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.8"
        }
    }

    cells = []

    # Markdown: Title & Objectives
    cells.append(nbf.v4.new_markdown_cell(
        "# Day 15: Cold-Chain Logistics, Isotopic Terroir Fraud Detection, Consumer Hedonics & Unified Winery Twin\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Author:** ML Engineering & Production Deployment Team\n"
        "**Milestone:** Day 15 Capstone: Enterprise Logistics, Provenance Forensics & Digital Winery Twin\n\n"
        "### Key Architectural Capabilities:\n"
        "1. **Intermodal Cold-Chain Logistics & Thermal Stress Forecaster:** Models diurnal shipping container thermal cycles, Arrhenius Thermal Damage Units ($TDU$), Free $SO_2$ oxidative exhaustion, ester hydrolysis, and non-enzymatic browning ($A_{420}$).\n"
        "2. **Barometric Altitude Headspace Dynamics & Closure Integrity:** Solves ideal gas and Henry's Law degassing under low-pressure air cargo ($75.2\\text{ kPa}$) or high mountain passes ($80.0\\text{ kPa}$), quantifying trans-closure differential pressure $\\Delta P = P_{hs} - P_{atm}$ and cork extrusion / weeping risks.\n"
        "3. **Stable Isotope Ratio Analysis (SIRA) Forensics:** Evaluates $\\delta^{13}\\text{C}_{VPDB}$ mass balance to detect exogenous C4 cane/corn sugar chaptalization ($< -25.5‰$ authentic threshold) and $\\delta^{18}\\text{O}_{VSMOW}$ for exogenous water dilution ($> +1.5‰$ authentic berry water threshold).\n"
        "4. **Bedrock $^{87}\\text{Sr}/^{86}\\text{Sr}$ & ICP-MS Geochemical Terroir Verification:** Computes Mahalanobis distance in orthogonalized elemental space (Li, Rb, Sr, Ba, Mn, B) and radiogenic strontium ratios to authenticate AOC/DOC/AVA geographical provenance against counterfeits.\n"
        "5. **Rosen Hedonic Econometric Pricing Model:** Decomposes predicted wine quality, aging years, and chemical balance into fair market bottle release valuations ($/750\\text{ mL}$) with 90% confidence corridors.\n"
        "6. **Cross-Regional Consumer Palate Alignment & Elasticity Optimization:** Scores finished wines across North America, Continental Europe, East Asia, and Nordic Monopolies, maximizing cellar gross margin under price elasticity of demand.\n"
        "7. **Unified 15-Stage Digital Winery Twin & ESG Life Cycle Assessment (LCA):** Holistically simulates must intake, primary/secondary fermentation, oak maturation, stabilization, filtration, bottling, shipping, and retail valuation, combined with Scope 1, 2, and 3 Greenhouse Gas (GHG) calculations ($g\\text{ CO}_2e / \\text{bottle}$) and water footprints."
    ))

    # Code: Setup & Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('..').resolve()) if Path('..').resolve().name != 'Wine Quality Prediction' else str(Path('.').resolve()))\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from src.transport_terroir_hedonic_and_twin import (\n"
        "    ColdChainLogisticsSimulator,\n"
        "    IsotopicTerroirAndFraudAuditor,\n"
        "    ConsumerHedonicAndMarketPricingOptimizer,\n"
        "    UnifiedDigitalWineryTwinPlatform,\n"
        "    TERROIR_PROFILES,\n"
        "    REGIONAL_PALATE_PROFILES,\n"
        "    CLOSURE_PRESSURE_TOLERANCES,\n"
        "    P_ATM_SEA_LEVEL,\n"
        "    P_ATM_AIR_CARGO_CRUISE,\n"
        "    P_ATM_MOUNTAIN_PASS\n"
        ")\n"
        "print('Successfully loaded Day 15 Logistics, Terroir, Hedonics & Winery Twin Modules!')"
    ))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Intermodal Cold-Chain Logistics, Thermal Shock & Barometric Ingress\n\n"
        "Simulates dynamic thermal trajectories, Arrhenius TDU hours, and trans-closure pressure differentials."
    ))

    # Code: Section 1
    cells.append(nbf.v4.new_code_cell(
        "log_sim = ColdChainLogisticsSimulator()\n\n"
        "# 1. Evaluate Maritime Reefer Route (Rotterdam to Singapore - 28 Days)\n"
        "reefer_res = log_sim.evaluate_transit_scenario(\n"
        "    preset_name='maritime_reefer_rotterdam_singapore',\n"
        "    closure_type='technical_diam',\n"
        "    initial_free_so2=35.0\n"
        ")\n"
        "print('--- MARITIME REEFER LOGISTICS AUDIT ---')\n"
        "print(f\"Route:                  {reefer_res['route_description']}\")\n"
        "print(f\"Duration:               {reefer_res['duration_days']} Days\")\n"
        "print(f\"Mean Temperature:       {reefer_res['temperature_summary']['mean_c']} °C\")\n"
        "print(f\"Arrhenius TDU Hours:    {reefer_res['kinetic_stress_metrics']['total_tdu_hours']} hrs (~{reefer_res['kinetic_stress_metrics']['equivalent_cellar_aging_months']} cellar months)\")\n"
        "print(f\"Free SO2 Loss:          {reefer_res['chemical_impact']['so2_loss_mg_l']} mg/L (Final: {reefer_res['chemical_impact']['final_free_so2_mg_l']} mg/L)\")\n"
        "print(f\"Peak Differential ΔP:   {reefer_res['barometric_seal_integrity']['peak_delta_p_kpa']} kPa\")\n"
        "print(f\"Seal Status / Risk:     {reefer_res['barometric_seal_integrity']['seal_status']} ({reefer_res['barometric_seal_integrity']['overall_transit_risk_score']}/100)\\n\")\n\n"
        "# 2. Evaluate Dry Container Trans-Panama Route (Critical Thermal Stress)\n"
        "dry_res = log_sim.evaluate_transit_scenario(\n"
        "    preset_name='dry_container_deck_panama',\n"
        "    closure_type='natural_cork_fleur',\n"
        "    initial_free_so2=35.0\n"
        ")\n"
        "print('--- DRY CONTAINER HIGH-STRESS ROUTE ---')\n"
        "print(f\"Route:                  {dry_res['route_description']}\")\n"
        "print(f\"Max Temperature Peak:   {dry_res['temperature_summary']['max_c']} °C\")\n"
        "print(f\"Arrhenius TDU Hours:    {dry_res['kinetic_stress_metrics']['total_tdu_hours']} hrs (~{dry_res['kinetic_stress_metrics']['equivalent_cellar_aging_months']} cellar months!)\")\n"
        "print(f\"Degree-Hours > 35°C:    {dry_res['kinetic_stress_metrics']['degree_hours_above_35c']} deg-hrs\")\n"
        "print(f\"Free SO2 Loss:          {dry_res['chemical_impact']['so2_loss_mg_l']} mg/L\")\n"
        "print(f\"Browning ΔA420:         +{dry_res['chemical_impact']['delta_a420']} OD\")\n"
        "print(f\"Fruity Ester Retention: {dry_res['chemical_impact']['fruity_ester_retention_pct']} %\")\n"
        "print(f\"Peak Differential ΔP:   {dry_res['barometric_seal_integrity']['peak_delta_p_kpa']} kPa (vs Cork Weep Limit {dry_res['barometric_seal_integrity']['weep_threshold_kpa']} kPa)\")\n"
        "print(f\"Seal Status / Risk:     {dry_res['barometric_seal_integrity']['seal_status']} ({dry_res['barometric_seal_integrity']['overall_transit_risk_score']}/100)\")"
    ))

    # Code: Section 1 part B (Air freight barometric calculation)
    cells.append(nbf.v4.new_code_cell(
        "# High-altitude air freight cargo hold pressure calculation (Patm = 75.2 kPa)\n"
        "air_seal = log_sim.calculate_headspace_pressure_differential(\n"
        "    liquid_temp_c=32.0,\n"
        "    ambient_pressure_kpa=P_ATM_AIR_CARGO_CRUISE,\n"
        "    dissolved_co2_mg_l=950.0,\n"
        "    initial_headspace_vacuum_kpa=15.0\n"
        ")\n"
        "print('--- AIR FREIGHT BAROMETRIC SEAL ASSESSMENT ---')\n"
        "print(f\"Ambient Cargo Pressure:    {air_seal['ambient_pressure_kpa']} kPa (Cruise Altitude)\")\n"
        "print(f\"Internal Headspace Pres:   {air_seal['total_internal_pressure_kpa']} kPa\")\n"
        "print(f\"Net Trans-Closure ΔP:      {air_seal['delta_p_differential_kpa']} kPa\")\n"
        "print(f\"Effective Headspace:       {air_seal['effective_headspace_ml']} mL\")"
    ))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Isotopic Terroir & Anti-Counterfeiting Forensics (δ¹³C, δ¹⁸O, ⁸⁷Sr/⁸⁶Sr & ICP-MS)\n\n"
        "Detects C4 chaptalization, water dilution, and verifies appellation d'origine authenticity."
    ))

    # Code: Section 2
    cells.append(nbf.v4.new_code_cell(
        "auditor = IsotopicTerroirAndFraudAuditor()\n\n"
        "# 1. Chaptalization detection across Carbon stable isotope values\n"
        "for d13c in [-27.2, -24.5, -19.0]:\n"
        "    ch_res = auditor.calculate_chaptalization_fraction(d13c)\n"
        "    print(f\"δ13C = {d13c} ‰ -> C4 Sugar: {ch_res['estimated_c4_sugar_fraction_pct']}% | Verdict: {ch_res['compliance_verdict']}\")\n\n"
        "# 2. Water addition detection across Oxygen stable isotope values\n"
        "for d18o in [3.8, 0.5, -4.5]:\n"
        "    w_res = auditor.calculate_water_dilution_fraction(d18o)\n"
        "    print(f\"δ18O = {d18o} ‰ -> Water Added: {w_res['estimated_exogenous_water_pct']}% | Verdict: {w_res['compliance_verdict']}\")\n\n"
        "# 3. Appellation Provenance Verification for Bordeaux Left Bank\n"
        "bordeaux_prof = TERROIR_PROFILES['bordeaux_left_bank']\n"
        "sample_auth = bordeaux_prof['minerals_ppm'].copy()\n"
        "verif_auth = auditor.verify_appellation_terroir(\n"
        "    claimed_appellation='bordeaux_left_bank',\n"
        "    sample_minerals=sample_auth,\n"
        "    measured_sr_ratio=bordeaux_prof['sr_87_86_ratio'],\n"
        "    measured_delta13c=-27.1,\n"
        "    measured_delta18o=3.7\n"
        ")\n"
        "print('\\n--- AUTHENTIC BORDEAUX VERIFICATION ---')\n"
        "print(f\"Appellation: {verif_auth['appellation_name']}\")\n"
        "print(f\"Mahalanobis Distance: {verif_auth['mahalanobis_terroir_distance']} (p-value: {verif_auth['terroir_p_value']})\")\n"
        "print(f\"Verdict: {verif_auth['verdict']} (Confidence: {verif_auth['confidence_pct']}%)\")\n\n"
        "# Counterfeit sample (Napa Valley claiming Bordeaux)\n"
        "napa_prof = TERROIR_PROFILES['napa_valley_rutherford']\n"
        "verif_fake = auditor.verify_appellation_terroir(\n"
        "    claimed_appellation='bordeaux_left_bank',\n"
        "    sample_minerals=napa_prof['minerals_ppm'],\n"
        "    measured_sr_ratio=napa_prof['sr_87_86_ratio'],\n"
        "    measured_delta13c=-27.0,\n"
        "    measured_delta18o=3.5\n"
        ")\n"
        "print('\\n--- COUNTERFEIT SAMPLE AUDIT ---')\n"
        "print(f\"Mahalanobis Distance: {verif_fake['mahalanobis_terroir_distance']}\")\n"
        "print(f\"Verdict: {verif_fake['verdict']} (Confidence: {verif_fake['confidence_pct']}%)\")"
    ))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Consumer Hedonic Econometrics, Regional Palate Fit & Release Pricing Optimization\n\n"
        "Decomposes bottle value into marginal attributes and optimizes cellar gross profit."
    ))

    # Code: Section 3
    cells.append(nbf.v4.new_code_cell(
        "hed_opt = ConsumerHedonicAndMarketPricingOptimizer()\n\n"
        "# 1. Fair Market Bottle Valuation across Quality Tiers\n"
        "for q, tier in [(5.2, 'table'), (6.5, 'igt'), (7.8, 'doc_quality'), (8.8, 'grand_cru')]:\n"
        "    val = hed_opt.estimate_bottle_market_value(predicted_quality=q, appellation_tier=tier, alcohol=13.8)\n"
        "    ci = val['confidence_interval_90_usd']\n"
        "    print(f\"Quality {q:.1f} ({tier.upper()}): ${val['estimated_market_price_usd']:.2f} (90% CI: ${ci[0]:.2f} - ${ci[1]:.2f})\")\n\n"
        "# 2. Cross-Regional Palate Alignment\n"
        "palate_res = hed_opt.calculate_regional_palate_alignment(\n"
        "    alcohol=14.2,\n"
        "    ph=3.68,\n"
        "    residual_sugar=3.2,\n"
        "    volatile_acidity=0.42,\n"
        "    sulphates=0.72,\n"
        "    oak_index=0.90\n"
        ")\n"
        "print('\\n--- REGIONAL PALATE ALIGNMENT ---')\n"
        "for r_id, r_info in palate_res.items():\n"
        "    print(f\"{r_info['region_name']}: Score {r_info['palate_alignment_score']}/100 ({r_info['market_fit_verdict']}) | Proj. Retail: ${r_info['projected_retail_usd']:.2f}\")\n\n"
        "# 3. Margin Optimization under Price Elasticity\n"
        "margin_res = hed_opt.optimize_release_pricing(\n"
        "    production_cost_per_bottle_usd=9.50,\n"
        "    market_valuation_usd=32.00,\n"
        "    price_elasticity=-1.25,\n"
        "    total_bottles_inventory=20000\n"
        ")\n"
        "print('\\n--- CELLAR RELEASE PRICING OPTIMIZER ---')\n"
        "print(f\"Recommended Release Price: ${margin_res['optimal_release_price_usd']:.2f}\")\n"
        "print(f\"Projected Sales Demand:    {margin_res['projected_sales_bottles']:,} bottles\")\n"
        "print(f\"Projected Gross Revenue:   ${margin_res['projected_gross_revenue_usd']:,.2f}\")\n"
        "print(f\"Projected Gross Profit:    ${margin_res['projected_gross_profit_usd']:,.2f} ({margin_res['gross_margin_pct']:.1f}% margin)\")"
    ))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Unified 15-Stage Digital Winery Twin & Scope 1/2/3 ESG Life Cycle Platform\n\n"
        "Connects all 15 operational stages into an end-to-end simulation with carbon and water footprint analysis."
    ))

    # Code: Section 4
    cells.append(nbf.v4.new_code_cell(
        "twin = UnifiedDigitalWineryTwinPlatform()\n\n"
        "full_twin = twin.simulate_end_to_end_winery_lifecycle(\n"
        "    cultivar='red',\n"
        "    harvest_sugar_brix=25.2,\n"
        "    must_tartaric_acid=6.8,\n"
        "    must_ph=3.52,\n"
        "    oak_regime='french_medium_plus',\n"
        "    oak_duration_months=14,\n"
        "    closure_choice='technical_diam',\n"
        "    shipping_route='maritime_reefer_rotterdam_singapore',\n"
        "    appellation_claimed='bordeaux_left_bank',\n"
        "    production_lot_bottles=25000\n"
        ")\n\n"
        "print('=== UNIFIED DIGITAL WINERY TWIN REPORT ===')\n"
        "print(f\"Simulation Lot ID:     {full_twin['simulation_id']}\")\n"
        "print(f\"Lot Size:              {full_twin['lot_size_bottles']:,} bottles\")\n"
        "print(f\"Fermented Alcohol:     {full_twin['stage_1_fermentation']['fermented_alcohol_pct']} % vol\")\n"
        "print(f\"Post-MLF pH:           {full_twin['stage_2_maturation']['post_mlf_ph']}\")\n"
        "print(f\"Final Clarification:   {full_twin['stage_3_stabilization']['final_turbidity_ntu']} NTU\")\n"
        "print(f\"Predicted Quality:     {full_twin['stage_4_quality_scoring']['predicted_quality_score']} ({full_twin['stage_4_quality_scoring']['quality_tier']})\")\n"
        "print(f\"Logistics Status:      {full_twin['stage_5_cold_chain_logistics']['barometric_seal_integrity']['seal_status']}\")\n"
        "print(f\"Terroir Verdict:       {full_twin['stage_6_terroir_authenticity']['verdict']}\")\n"
        "print(f\"Release Price:         ${full_twin['stage_7_hedonic_and_pricing']['optimized_pricing']['optimal_release_price_usd']:.2f}\")\n\n"
        "lca = full_twin['stage_8_esg_lca_sustainability']\n"
        "print('--- ESG CARBON & WATER LIFE CYCLE ASSESSMENT ---')\n"
        "print(f\"Scope 1 Direct:        {lca['scope_1_direct_g_co2e']} g CO2e\")\n"
        "print(f\"Scope 2 Electricity:   {lca['scope_2_electricity_g_co2e']} g CO2e\")\n"
        "print(f\"Scope 3 Supply Chain:  {lca['scope_3_supply_chain_g_co2e']} g CO2e ({lca['glass_bottle_contribution_pct']}% from glass bottle)\")\n"
        "print(f\"Total Footprint:       {lca['total_footprint_g_co2e_per_bottle']} g CO2e / bottle ({lca['sustainability_rating']})\")\n"
        "print(f\"Water Footprint:       {lca['water_footprint_liters_per_liter']} L H2O / L Wine\")"
    ))

    # Save notebook to disk
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created unexecuted notebook: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


def execute_day15_notebook():
    print("Executing Day 15 notebook via nbconvert...")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved populated notebook: {NOTEBOOK_PATH}")


def main():
    create_day15_notebook()
    execute_day15_notebook()


if __name__ == "__main__":
    main()
