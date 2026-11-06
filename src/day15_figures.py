"""
Day 15 Diagnostic Figures Generator.

Generates 4 publication-quality enological figures:
1. reports/figures/day15_cold_chain_thermal_and_barometric_stress.png
2. reports/figures/day15_isotopic_terroir_and_fraud_detection.png
3. reports/figures/day15_consumer_hedonic_market_econometrics.png
4. reports/figures/day15_unified_digital_winery_twin_cockpit.png
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.transport_terroir_hedonic_and_twin import (
    ColdChainLogisticsSimulator,
    IsotopicTerroirAndFraudAuditor,
    ConsumerHedonicAndMarketPricingOptimizer,
    UnifiedDigitalWineryTwinPlatform,
    TERROIR_PROFILES,
    REGIONAL_PALATE_PROFILES,
    CLOSURE_PRESSURE_TOLERANCES,
    P_ATM_SEA_LEVEL,
    P_ATM_AIR_CARGO_CRUISE
)

REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling standards
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 15,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def generate_figure_1_logistics():
    """Generates 4-panel diagnostic figure for Cold-Chain Transportation Stress."""
    sim = ColdChainLogisticsSimulator()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 15: Intermodal Cold-Chain Logistics, Thermal Shock & Barometric Ingress Forecaster", fontsize=15, fontweight="bold", y=0.98)

    # 1. Thermal Trajectory: Maritime Reefer vs Dry Container vs Truck
    ax = axes[0, 0]
    reefer_df = sim.simulate_thermal_trajectory(duration_days=25, mean_temp_c=14.0, diurnal_amplitude_c=1.0)
    dry_df = sim.simulate_thermal_trajectory(duration_days=25, mean_temp_c=31.0, diurnal_amplitude_c=11.0)
    truck_df = sim.simulate_thermal_trajectory(duration_days=5, mean_temp_c=24.0, diurnal_amplitude_c=9.0)

    ax.plot(reefer_df["day"], reefer_df["temp_c"], color="#2ca02c", lw=2, label="Maritime Reefer (14±1°C)")
    ax.plot(dry_df["day"], dry_df["temp_c"], color="#d62728", lw=1.8, label="Dry Deck Container (31±11°C)")
    ax.axhline(25.0, color="#ff7f0e", linestyle=":", lw=1.5, label="Quality Degradation Threshold (25°C)")
    ax.axhline(35.0, color="#9467bd", linestyle="--", lw=1.5, label="Critical PremOx / Heat Damage (35°C)")
    ax.set_title("Container Thermal Exposure Trajectories")
    ax.set_xlabel("Transit Elapsed Time (Days)")
    ax.set_ylabel("Liquid Wine Temperature (°C)")
    ax.set_ylim(8, 48)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    # 2. Cumulative Arrhenius Thermal Damage Units (TDU) & SO2 Depletion
    ax = axes[0, 1]
    ax.plot(reefer_df["day"], reefer_df["cumulative_tdu_hours"], color="#2ca02c", lw=2, label="Reefer TDU Hours")
    ax.plot(dry_df["day"], dry_df["cumulative_tdu_hours"], color="#d62728", lw=2, label="Dry Container TDU Hours")
    ax.set_title("Arrhenius Thermal Damage Units (TDU Accumulation)")
    ax.set_xlabel("Transit Elapsed Time (Days)")
    ax.set_ylabel("Normalized Aging Hours (TDU)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    # Secondary axis for Free SO2 decay
    ax_so2 = ax.twinx()
    so2_dry = np.maximum(5.0, 35.0 - (dry_df["cumulative_tdu_hours"] / (24.0 * 30.4)) * 0.45 - (dry_df["cumulative_deg_hrs_gt35"] / 100.0) * 1.5)
    so2_reefer = np.maximum(5.0, 35.0 - (reefer_df["cumulative_tdu_hours"] / (24.0 * 30.4)) * 0.45)
    ax_so2.plot(reefer_df["day"], so2_reefer, color="#1f77b4", linestyle="--", label="Reefer Free SO2 (mg/L)")
    ax_so2.plot(dry_df["day"], so2_dry, color="#e377c2", linestyle="--", label="Dry Deck Free SO2 (mg/L)")
    ax_so2.set_ylabel("Residual Free SO2 (mg/L)", color="#1f77b4")
    ax_so2.tick_params(axis="y", labelcolor="#1f77b4")
    ax_so2.set_ylim(0, 40)

    # 3. Headspace Pressure Differential (Delta P) vs Liquid Temperature across Altitudes
    ax = axes[1, 0]
    temps = np.linspace(10.0, 45.0, 36)
    delta_p_sea = []
    delta_p_air = []
    for t in temps:
        delta_p_sea.append(sim.calculate_headspace_pressure_differential(t, ambient_pressure_kpa=P_ATM_SEA_LEVEL)["delta_p_differential_kpa"])
        delta_p_air.append(sim.calculate_headspace_pressure_differential(t, ambient_pressure_kpa=P_ATM_AIR_CARGO_CRUISE)["delta_p_differential_kpa"])

    ax.plot(temps, delta_p_sea, color="#1f77b4", lw=2, label="Sea Level Transit (Patm = 101.3 kPa)")
    ax.plot(temps, delta_p_air, color="#ff7f0e", lw=2, linestyle="-.", label="Air Cargo Hold (Patm = 75.2 kPa)")
    ax.axhline(CLOSURE_PRESSURE_TOLERANCES["natural_cork_fleur"]["weep_threshold_kpa"], color="#d62728", linestyle=":", lw=1.5, label="Cork Weeping Threshold (90 kPa)")
    ax.axhline(CLOSURE_PRESSURE_TOLERANCES["natural_cork_fleur"]["extrude_threshold_kpa"], color="#8c564b", linestyle="--", lw=1.5, label="Cork Extrusion Threshold (135 kPa)")
    ax.set_title("Trans-Closure Pressure Differential (ΔP = Phs - Patm)")
    ax.set_xlabel("Liquid Temperature (°C)")
    ax.set_ylabel("Pressure Differential (kPa)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)

    # 4. Multi-Route Transit Risk Scorecard Comparison
    ax = axes[1, 1]
    routes = list(sim.transit_presets.keys())
    route_labels = ["Maritime Reefer", "Dry Container", "Intermodal Truck", "Air Freight"]
    cork_risks = []
    diam_risks = []
    for r in routes:
        cork_risks.append(sim.evaluate_transit_scenario(r, closure_type="natural_cork_fleur")["barometric_seal_integrity"]["overall_transit_risk_score"])
        diam_risks.append(sim.evaluate_transit_scenario(r, closure_type="screwcap_sarantin")["barometric_seal_integrity"]["overall_transit_risk_score"])

    x = np.arange(len(routes))
    width = 0.35
    ax.bar(x - width/2, cork_risks, width, label="Natural Cork Fleur", color="#e28743")
    ax.bar(x + width/2, diam_risks, width, label="Screwcap Saran-Tin", color="#1f77b4")
    ax.axhline(50, color="gray", linestyle="--", lw=1, label="High-Risk Gate (50)")
    ax.set_title("Transit Vulnerability Risk Score by Route & Closure")
    ax.set_xticks(x)
    ax.set_xticklabels(route_labels, rotation=15, ha="right")
    ax.set_ylabel("Overall Transit Risk Score (0 - 100)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out_path = REPORTS_DIR / "day15_cold_chain_thermal_and_barometric_stress.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure_2_terroir():
    """Generates 4-panel diagnostic figure for Isotopic Terroir & Anti-Counterfeit Audits."""
    auditor = IsotopicTerroirAndFraudAuditor()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 15: Isotopic Terroir Fingerprinting & Anti-Counterfeiting Forensics", fontsize=15, fontweight="bold", y=0.98)

    # 1. Delta 13C VPDB Chaptalization Forensics (C3 Grape vs C4 Cane/Corn Sugar)
    ax = axes[0, 0]
    delta13c_vals = np.linspace(-28.0, -13.0, 50)
    c4_fractions = [auditor.calculate_chaptalization_fraction(d)["estimated_c4_sugar_fraction_pct"] for d in delta13c_vals]

    ax.plot(delta13c_vals, c4_fractions, color="#1f77b4", lw=2.5, label="C4 Sugar Fraction (%)")
    ax.axvspan(-28.5, -25.5, color="#2ca02c", alpha=0.25, label="Authentic C3 Grape Domain (<-25.5 ‰)")
    ax.axvspan(-25.5, -22.0, color="#ff7f0e", alpha=0.25, label="Suspect Moderate Chaptalization")
    ax.axvspan(-22.0, -12.5, color="#d62728", alpha=0.25, label="Severe Illegal Chaptalization")
    ax.set_title("Carbon Stable Isotope (δ¹³C VPDB) Chaptalization Curve")
    ax.set_xlabel("Measured δ¹³C (‰ VPDB)")
    ax.set_ylabel("Estimated Exogenous C4 Sugar (%)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)

    # 2. Delta 18O VSMOW Water Dilution Forensics
    ax = axes[0, 1]
    delta18o_vals = np.linspace(-8.0, 6.0, 50)
    water_fractions = [auditor.calculate_water_dilution_fraction(d)["estimated_exogenous_water_pct"] for d in delta18o_vals]

    ax.plot(delta18o_vals, water_fractions, color="#d62728", lw=2.5, label="Exogenous Tap Water (%)")
    ax.axvspan(1.5, 6.0, color="#2ca02c", alpha=0.25, label="Natural Transpiration Enriched (>+1.5 ‰)")
    ax.axvspan(-1.0, 1.5, color="#ff7f0e", alpha=0.25, label="Suspect Water Dilution")
    ax.axvspan(-8.0, -1.0, color="#9467bd", alpha=0.25, label="Severe Watering-Down Adulteration")
    ax.set_title("Oxygen Stable Isotope (δ¹⁸O VSMOW) Water Addition Curve")
    ax.set_xlabel("Measured δ¹⁸O (‰ VSMOW)")
    ax.set_ylabel("Estimated Exogenous Water Fraction (%)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)

    # 3. Bedrock 87Sr/86Sr Isotope & Rubidium ICP-MS Scatter Plot by Terroir
    ax = axes[1, 0]
    colors = {"bordeaux_left_bank": "#1f77b4", "burgundy_cote_dor": "#d62728", "napa_valley_rutherford": "#2ca02c", "rioja_alta": "#ff7f0e", "douro_schist": "#9467bd"}
    for t_id, prof in TERROIR_PROFILES.items():
        sr_mean = prof["sr_87_86_ratio"]
        rb_mean = prof["minerals_ppm"]["rubidium"]
        # Generate representative cluster
        sr_pts = np.random.normal(sr_mean, prof["sr_87_86_std"], 30)
        rb_pts = np.random.normal(rb_mean, prof["mineral_stds"]["rubidium"], 30)
        ax.scatter(sr_pts, rb_pts, label=prof["appellation"].split("(")[0].strip(), color=colors[t_id], alpha=0.75, edgecolors="none")

    ax.set_title("Bedrock ⁸⁷Sr/⁸⁶Sr Radiogenic Ratio vs Rubidium (ICP-MS)")
    ax.set_xlabel("⁸⁷Sr/⁸⁶Sr Bedrock Isotope Ratio")
    ax.set_ylabel("Rubidium Concentration (ppm)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)

    # 4. Multi-Mineral Geochemical Fingerprint Spider / Radar Profile
    ax = axes[1, 1]
    elements = ["lithium", "rubidium", "strontium", "barium", "manganese", "boron"]
    elem_labels = ["Li (x20)", "Rb", "Sr", "Ba (x5)", "Mn", "B"]
    x_indices = np.arange(len(elements))

    for t_id, prof in list(TERROIR_PROFILES.items())[:3]:
        vals = []
        for e in elements:
            raw = prof["minerals_ppm"][e]
            if e == "lithium": raw *= 20
            elif e == "barium": raw *= 5
            vals.append(raw)
        ax.plot(x_indices, vals, marker="o", lw=2, label=prof["appellation"].split("(")[0].strip(), color=colors[t_id])

    ax.set_title("Normalized Trace Element Provenance Profiles")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(elem_labels)
    ax.set_ylabel("Calibrated Element Intensity (ppm)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path = REPORTS_DIR / "day15_isotopic_terroir_and_fraud_detection.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure_3_hedonic():
    """Generates 4-panel diagnostic figure for Consumer Hedonics & Market Econometrics."""
    optimizer = ConsumerHedonicAndMarketPricingOptimizer()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 15: Global Consumer Hedonics, Regional Palate Alignment & Price Elasticity", fontsize=15, fontweight="bold", y=0.98)

    # 1. Rosen Hedonic Pricing Curve: Predicted Quality vs Fair Market Bottle Price
    ax = axes[0, 0]
    qualities = np.linspace(4.5, 9.0, 30)
    tiers = [("table", "Table / Everyday", "#7f7f7f", ":"),
             ("igt", "IGT / Varietal", "#1f77b4", "--"),
             ("doc_quality", "DOC / Premium", "#2ca02c", "-"),
             ("grand_cru", "Grand Cru / Icon", "#d62728", "-")]

    for tier_key, tier_label, c, ls in tiers:
        prices = [optimizer.estimate_bottle_market_value(q, appellation_tier=tier_key)["estimated_market_price_usd"] for q in qualities]
        ax.plot(qualities, prices, color=c, linestyle=ls, lw=2.2, label=tier_label)

    ax.set_title("Hedonic Price Valuation by Quality Score & Appellation Tier")
    ax.set_xlabel("Predicted Physicochemical Quality Score")
    ax.set_ylabel("Estimated Fair Market Price ($ / 750mL)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    # 2. Regional Consumer Palate Alignment Comparison
    ax = axes[0, 1]
    specimens = {
        "Napa Valley Cab": {"alcohol": 14.5, "ph": 3.72, "residual_sugar": 3.8, "volatile_acidity": 0.44, "sulphates": 0.75, "oak_index": 0.95},
        "Burgundy Pinot": {"alcohol": 12.8, "ph": 3.32, "residual_sugar": 1.2, "volatile_acidity": 0.46, "sulphates": 0.58, "oak_index": 0.65},
        "Commercial Red": {"alcohol": 13.0, "ph": 3.55, "residual_sugar": 6.5, "volatile_acidity": 0.52, "sulphates": 0.62, "oak_index": 0.50}
    }

    regions = list(REGIONAL_PALATE_PROFILES.keys())
    region_labels = ["North America", "Cont. Europe", "East Asia", "Nordic"]
    x = np.arange(len(regions))
    width = 0.25

    for idx, (spec_name, spec_params) in enumerate(specimens.items()):
        scores = [optimizer.calculate_regional_palate_alignment(**spec_params)[r]["palate_alignment_score"] for r in regions]
        ax.bar(x + (idx - 1) * width, scores, width, label=spec_name)

    ax.set_title("Cross-Regional Palate Alignment Score (0 - 100)")
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel("Palate Fit Score")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8.5)

    # 3. Demand Price Elasticity & Revenue Surface
    ax = axes[1, 0]
    p_range = np.linspace(10.0, 80.0, 50)
    cost = 12.0
    for eps, label, c in [(-2.4, "Everyday (ε = -2.4)", "#1f77b4"),
                          (-1.25, "Premium Cru (ε = -1.25)", "#2ca02c"),
                          (-0.45, "Icon / Luxury (ε = -0.45)", "#d62728")]:
        q_dem = 10000.0 * ((p_range / 30.0) ** eps)
        profit = (p_range - cost) * q_dem
        # Normalize to index
        ax.plot(p_range, profit / 1000.0, color=c, lw=2.2, label=label)

    ax.set_title("Cellar Gross Profit vs Release Price Curve")
    ax.set_xlabel("Release Price ($ / Bottle)")
    ax.set_ylabel("Projected Gross Profit ($k)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8.5)

    # 4. Willingness-to-Pay (WTP) vs Palate Fit Distribution
    ax = axes[1, 1]
    scores_sweep = np.linspace(30.0, 98.0, 35)
    for r_key, r_prof in REGIONAL_PALATE_PROFILES.items():
        wtp = [r_prof["base_wtp_usd"] * (s / 80.0) for s in scores_sweep]
        ax.plot(scores_sweep, wtp, lw=2, label=r_prof["name"].split("(")[0].strip())

    ax.set_title("Projected Retail WTP vs Palate Fit Score")
    ax.set_xlabel("Consumer Palate Fit Score")
    ax.set_ylabel("Projected Retail WTP ($ / Bottle)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out_path = REPORTS_DIR / "day15_consumer_hedonic_market_econometrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure_4_digital_twin():
    """Generates 4-panel diagnostic figure for Unified Digital Winery Twin & ESG Life Cycle."""
    twin = UnifiedDigitalWineryTwinPlatform()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 15: Unified Digital Winery Twin & ESG Carbon Life Cycle Assessment", fontsize=15, fontweight="bold", y=0.98)

    # 1. Multi-Stage Quality & Physicochemical Trajectory across Winemaking Gates
    ax = axes[0, 0]
    stages = ["Intake Must", "Post-Ferm", "Post-MLF", "Post-Oak", "Post-Stab", "Bottled", "Post-Transit"]
    alcohol_traj = [0.0, 14.1, 14.1, 14.2, 14.2, 14.2, 14.2]
    ph_traj = [3.55, 3.52, 3.68, 3.68, 3.66, 3.66, 3.67]
    so2_traj = [0.0, 0.0, 15.0, 25.0, 30.0, 35.0, 32.5]

    ax.plot(stages, alcohol_traj, marker="s", color="#1f77b4", lw=2, label="Alcohol (% vol)")
    ax.set_title("End-to-End Physicochemical Evolution")
    ax.set_ylabel("Alcohol (% vol)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.grid(True)

    ax_ph = ax.twinx()
    ax_ph.plot(stages, ph_traj, marker="o", color="#d62728", lw=2, linestyle="--", label="pH Value")
    ax_ph.set_ylabel("pH Level", color="#d62728")
    ax_ph.tick_params(axis="y", labelcolor="#d62728")
    ax_ph.set_ylim(3.0, 4.0)

    # 2. Scope 1, 2, 3 GHG Carbon Breakdown per 750 mL Bottle
    ax = axes[0, 1]
    lca_standard = twin.calculate_sustainability_lca(bottle_glass_weight_g=550.0, freight_mode="truck", is_organic_certified=False)
    lca_eco = twin.calculate_sustainability_lca(bottle_glass_weight_g=390.0, freight_mode="maritime", is_organic_certified=True)

    categories = ["Scope 1 (Direct)", "Scope 2 (Electricity)", "Scope 3 (Glass)", "Scope 3 (Freight)", "Total Carbon"]
    std_vals = [lca_standard["scope_1_direct_g_co2e"], lca_standard["scope_2_electricity_g_co2e"], 550.0 * 0.85, 4500.0 * 0.095, lca_standard["total_footprint_g_co2e_per_bottle"]]
    eco_vals = [lca_eco["scope_1_direct_g_co2e"], lca_eco["scope_2_electricity_g_co2e"], 390.0 * 0.85, 11000.0 * 0.018, lca_eco["total_footprint_g_co2e_per_bottle"]]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, std_vals, width, label="Conventional Heavy Bottle", color="#7f7f7f")
    ax.bar(x + width/2, eco_vals, width, label="Eco-Optimized Lightweight", color="#2ca02c")
    ax.set_title("Carbon Footprint (g CO₂e / Bottle) by LCA Scope")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Emissions (g CO₂e / 750mL Bottle)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=8.5)

    # 3. Water Footprint & Glass Lightweighting Savings Matrix
    ax = axes[1, 0]
    glass_weights = np.array([350, 420, 500, 600, 750, 900])
    co2_glass = glass_weights * 0.85
    savings_pct = (1.0 - co2_glass / co2_glass[-1]) * 100.0

    ax.plot(glass_weights, co2_glass, marker="o", color="#1f77b4", lw=2.2, label="Emissions (g CO2e)")
    ax.set_title("Bottle Glass Mass vs Embodied Carbon & Savings")
    ax.set_xlabel("Empty Glass Bottle Weight (grams)")
    ax.set_ylabel("Embodied Carbon (g CO₂e)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True)

    ax_sav = ax.twinx()
    ax_sav.plot(glass_weights, savings_pct, marker="s", color="#2ca02c", lw=2, linestyle=":", label="Carbon Savings (%)")
    ax_sav.set_ylabel("Relative Carbon Savings vs Ultra-Heavy (%)", color="#2ca02c")
    ax_sav.tick_params(axis="y", labelcolor="#2ca02c")

    # 4. Integrated Winery Profit vs Sustainability Eco-Efficiency Tradeoff
    ax = axes[1, 1]
    profit_margins = [38.5, 45.2, 52.0, 58.4, 64.1]
    eco_ratings = [980, 840, 760, 690, 620]
    bubble_sizes = [250, 350, 500, 700, 900] # Production volume (k bottles)

    scatter = ax.scatter(eco_ratings, profit_margins, s=bubble_sizes, c=profit_margins, cmap="viridis", alpha=0.8, edgecolors="black")
    ax.set_title("Winery Gross Margin (%) vs Product Carbon Intensity")
    ax.set_xlabel("Total Lifecycle Carbon (g CO₂e / Bottle)")
    ax.set_ylabel("Gross Operating Margin (%)")
    ax.invert_xaxis()  # Lower carbon is better (to the right)
    ax.grid(True)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Margin %")

    plt.tight_layout()
    out_path = REPORTS_DIR / "day15_unified_digital_winery_twin_cockpit.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("=" * 70)
    print("DAY 15 DIAGNOSTIC FIGURES GENERATION PIPELINE")
    print("=" * 70)
    generate_figure_1_logistics()
    generate_figure_2_terroir()
    generate_figure_3_hedonic()
    generate_figure_4_digital_twin()
    print("All Day 15 diagnostic figures generated successfully.")


if __name__ == "__main__":
    main()
