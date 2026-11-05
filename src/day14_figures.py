"""
Day 14 Diagnostic Figures Generator.

Generates 4 publication-quality enological figures:
1. reports/figures/day14_tartrate_and_protein_stabilization.png
2. reports/figures/day14_crossflow_membrane_filtration.png
3. reports/figures/day14_dissolved_gas_and_sparging.png
4. reports/figures/day14_closure_permeation_and_shelf_life.png
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

REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling standards
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def generate_figure_1_stabilization():
    """Generates Tartrate and Protein Stabilization 4-panel diagnostic figure."""
    tartrate_sim = TartrateStabilizationSimulator()
    bentonite_opt = ProteinHazeAndBentoniteOptimizer()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 14: Enological Tartrate & Protein Stabilization Dynamics", fontsize=16, fontweight="bold", y=0.98)

    # ----------------------------------------------------
    # Panel 1: Tartrate Saturation & Conductivity Phase Space
    # ----------------------------------------------------
    ax = axes[0, 0]
    potassium_range = np.linspace(600, 1400, 40)
    ph_levels = [3.1, 3.3, 3.5, 3.7]
    colors = ["#2b5c8f", "#2ca02c", "#ff7f0e", "#d62728"]

    for ph_val, c in zip(ph_levels, colors):
        drops = [
            tartrate_sim.evaluate_tartrate_stability(6.5, k_val, ph_val, 13.0)["minicontact_delta_chi_us_cm"]
            for k_val in potassium_range
        ]
        ax.plot(potassium_range, drops, label=f"pH {ph_val}", color=c, lw=2.2)

    ax.axhline(30.0, color="#2ca02c", linestyle="--", lw=1.5, label="Stable Boundary (30 uS/cm)")
    ax.axhline(55.0, color="#d62728", linestyle="--", lw=1.5, label="Unstable Threshold (55 uS/cm)")
    ax.fill_between(potassium_range, 0, 30, color="#2ca02c", alpha=0.10)
    ax.fill_between(potassium_range, 30, 55, color="#ff7f0e", alpha=0.10)
    ax.fill_between(potassium_range, 55, 110, color="#d62728", alpha=0.08)

    ax.set_title("Minicontact Conductivity Drop vs Potassium & pH")
    ax.set_xlabel("Potassium Content (mg/L)")
    ax.set_ylabel(r"Conductivity Drop $\Delta \chi$ ($\mu$S/cm)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 2: Cold Stabilization Crystallization Kinetics (-4C)
    # ----------------------------------------------------
    ax = axes[0, 1]
    res_seeded = tartrate_sim.simulate_cold_stabilization_kinetics(
        tartaric_acid_g_l=7.2, potassium_mg_l=1150.0, ph=3.55, alcohol_pct=13.5,
        chilling_temp_c=-4.0, seed_crystal_g_l=4.0, duration_hours=12
    )
    res_unseeded = tartrate_sim.simulate_cold_stabilization_kinetics(
        tartaric_acid_g_l=7.2, potassium_mg_l=1150.0, ph=3.55, alcohol_pct=13.5,
        chilling_temp_c=-4.0, seed_crystal_g_l=0.0, duration_hours=12
    )

    t_hrs = res_seeded["trajectory"]["hour"]
    ax.plot(t_hrs, res_seeded["trajectory"]["kht_precipitated_g_l"], label="Contact Process (4 g/L Seed)", color="#1f77b4", lw=2.5)
    ax.plot(t_hrs, res_unseeded["trajectory"]["kht_precipitated_g_l"], label="Static Chilling (Unseeded)", color="#7f7f7f", linestyle="--", lw=2.0)

    ax.set_title(r"Potassium Bitartrate Precipitation Kinetics at $-4^\circ$C")
    ax.set_xlabel("Contact Time (Hours)")
    ax.set_ylabel("Precipitated KHT (g/L)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    # Secondary axis for Potassium concentration
    ax2 = ax.twinx()
    ax2.plot(t_hrs, res_seeded["trajectory"]["potassium_mg_l"], color="#d62728", lw=1.8, linestyle=":")
    ax2.set_ylabel("Residual Potassium (mg/L)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # ----------------------------------------------------
    # Panel 3: Protein Haze vs Bentonite Fining Dosage (Langmuir)
    # ----------------------------------------------------
    ax = axes[1, 0]
    dosages = np.linspace(0, 140, 30)
    initial_proteins = [80.0, 130.0, 180.0]
    p_colors = ["#17becf", "#9467bd", "#e377c2"]

    for init_p, col in zip(initial_proteins, p_colors):
        delta_ntu_curve = []
        for d in dosages:
            # Langmuir binding removal
            q_ads = (85.0 * 0.045 * init_p) / (1.0 + 0.045 * init_p)
            prot_removed = min(init_p, (d / 100.0) * q_ads)
            res_p = init_p - prot_removed
            haze = bentonite_opt.evaluate_protein_haze_risk(res_p, 3.25)["estimated_heat_delta_ntu"]
            delta_ntu_curve.append(haze)
        ax.plot(dosages, delta_ntu_curve, label=f"Initial Protein {int(init_p)} mg/L", color=col, lw=2.2)

    ax.axhline(1.8, color="#2ca02c", linestyle="--", lw=1.5, label="Stability Standard (<= 1.8 NTU)")
    ax.set_title("Heat-Induced Turbidity vs Bentonite Fining Dosage")
    ax.set_xlabel("Sodium Bentonite Dosage (g/hL)")
    ax.set_ylabel(r"Heat Test Haze $\Delta$ NTU (80$^\circ$C / 2h)")
    ax.set_ylim(0, 18)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 4: Alternative Tartrate Stabilizers Comparison
    # ----------------------------------------------------
    ax = axes[1, 1]
    add_names = ["Cold Contact", "CMC (E466)", "KPA (E456)", "Metatartaric", "Electrodialysis"]
    durabilities = [60, 36, 48, 9, 72]
    costs = [1.45, 0.42, 0.85, 0.28, 1.95]
    energy = [3.8, 0.1, 0.1, 0.05, 1.2]

    x = np.arange(len(add_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, durabilities, width, label="Stability Duration (Months)", color="#2b5c8f", alpha=0.85)
    ax.set_ylabel("Stability Longevity (Months)", color="#2b5c8f")
    ax.set_xticks(x)
    ax.set_xticklabels(add_names, rotation=15, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#2b5c8f")

    ax_cost = ax.twinx()
    rects2 = ax_cost.bar(x + width/2, costs, width, label="Process Cost (EUR/hL)", color="#ff7f0e", alpha=0.85)
    ax_cost.set_ylabel("Operating Cost (€/hL)", color="#ff7f0e")
    ax_cost.tick_params(axis="y", labelcolor="#ff7f0e")

    ax.set_title("Comparison of Tartrate Stabilization Methodologies")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "day14_tartrate_and_protein_stabilization.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure_2_filtration():
    """Generates Crossflow Membrane Filtration 4-panel diagnostic figure."""
    filt_sim = CrossflowFiltrationSimulator(membrane_surface_area_m2=40.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 14: Crossflow Membrane Microfiltration (0.45 um PES)", fontsize=16, fontweight="bold", y=0.98)

    # ----------------------------------------------------
    # Panel 1: Permeate Flux Decline & Backwash Hydrodynamic Restoration
    # ----------------------------------------------------
    ax = axes[0, 0]
    res = filt_sim.simulate_crossflow_run(
        initial_turbidity_ntu=38.0, transmembrane_pressure_bar=1.6,
        target_volume_liters=15000.0, colloidal_index=1.2,
        backwash_interval_hours=2.0, run_duration_hours=6.0
    )
    traj = res["trajectory"]
    ax.plot(traj["hour"], traj["flux_lmh"], color="#1f77b4", lw=2.5, label="Actual Permeate Flux (LMH)")
    ax.axvline(2.0, color="#2ca02c", linestyle=":", lw=1.8, label="Hydraulic Backwash Cycle")
    ax.axvline(4.0, color="#2ca02c", linestyle=":", lw=1.8)

    ax.set_title("Permeate Flux Decay J(t) & Backwash Regeneration")
    ax.set_xlabel("Filtration Time (Hours)")
    ax.set_ylabel("Permeate Flux (L / m² / h)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 2: Cumulative Volume & Turbidity Clarification
    # ----------------------------------------------------
    ax = axes[0, 1]
    ax.plot(traj["hour"], traj["cumulative_volume_l"], color="#2ca02c", lw=2.5, label="Cumulative Permeate (L)")
    ax.set_title("Cumulative Volume Throughput & Permeate Turbidity")
    ax.set_xlabel("Filtration Time (Hours)")
    ax.set_ylabel("Filtered Volume (L)", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(traj["hour"], traj["permeate_ntu"], color="#d62728", lw=2.0, linestyle="--", label="Permeate Turbidity (NTU)")
    ax2.axhline(0.60, color="#9467bd", linestyle=":", lw=1.5, label="Sterile Bottling Limit (0.6 NTU)")
    ax2.set_ylabel("Nephelometric Turbidity (NTU)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc="upper left", fontsize=8.5)

    # ----------------------------------------------------
    # Panel 3: Hermia's 4 Fouling Mechanisms Comparison
    # ----------------------------------------------------
    ax = axes[1, 0]
    t_fouling = np.linspace(0, 5, 50)
    # Complete blocking (n=2): J = J0 * exp(-k*t)
    j_complete = 120.0 * np.exp(-0.35 * t_fouling)
    # Intermediate blocking (n=1): 1/J = 1/J0 + k*t
    j_inter = 120.0 / (1.0 + 0.35 * t_fouling)
    # Standard pore constriction (n=1.5): 1/sqrt(J) = 1/sqrt(J0) + k*t
    j_standard = (1.0 / (1.0 / np.sqrt(120.0) + 0.035 * t_fouling)) ** 2
    # Cake filtration (n=0): 1/J^2 = 1/J0^2 + k*t
    j_cake = np.sqrt(1.0 / (1.0 / (120.0**2) + 0.00045 * t_fouling))

    ax.plot(t_fouling, j_complete, label="Complete Blocking (n=2)", color="#d62728", lw=2.0)
    ax.plot(t_fouling, j_inter, label="Intermediate Blocking (n=1)", color="#ff7f0e", lw=2.0)
    ax.plot(t_fouling, j_standard, label="Standard Pore Constriction (n=1.5)", color="#9467bd", lw=2.0)
    ax.plot(t_fouling, j_cake, label="Surface Cake Layer (n=0)", color="#1f77b4", lw=2.0)

    ax.set_title("Hermia's Classic Membrane Fouling Rate Laws")
    ax.set_xlabel("Relative Operating Time (t)")
    ax.set_ylabel("Normalized Flux (LMH)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 4: Filterability Index (FI) & Colloidal Fouling Bands
    # ----------------------------------------------------
    ax = axes[1, 1]
    colloid_levels = np.linspace(0.4, 2.5, 30)
    fi_white = [8.0 + 12.0 * cl + 0.15 * 18.0 for cl in colloid_levels]
    fi_red = [8.0 + 12.0 * cl + 0.15 * 45.0 for cl in colloid_levels]

    ax.plot(colloid_levels, fi_white, label="White Wine Matrix (18 NTU)", color="#2ca02c", lw=2.2)
    ax.plot(colloid_levels, fi_red, label="Red Wine Matrix (45 NTU)", color="#d62728", lw=2.2)

    ax.axhline(15.0, color="#2ca02c", linestyle="--", lw=1.5, label="Excellent Filterability (< 15 s)")
    ax.axhline(30.0, color="#ff7f0e", linestyle="--", lw=1.5, label="Good / Acceptable (30 s)")
    ax.axhline(50.0, color="#d62728", linestyle="--", lw=1.5, label="Severe Membrane Clogging (> 50 s)")
    ax.fill_between(colloid_levels, 0, 15, color="#2ca02c", alpha=0.10)
    ax.fill_between(colloid_levels, 15, 30, color="#1f77b4", alpha=0.08)
    ax.fill_between(colloid_levels, 30, 50, color="#ff7f0e", alpha=0.10)
    ax.fill_between(colloid_levels, 50, 70, color="#d62728", alpha=0.08)

    ax.set_title(r"Filterability Index ($FI = t_{400} - 2 t_{200}$) vs Colloidal Load")
    ax.set_xlabel("Colloidal Fouling Load Factor")
    ax.set_ylabel("Filterability Index (Seconds)")
    ax.set_ylim(0, 65)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "day14_crossflow_membrane_filtration.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure_3_dissolved_gas():
    """Generates Dissolved Gas Packaging & Sparging 4-panel diagnostic figure."""
    gas_eng = DissolvedGasPackagingEngine()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 14: Dissolved Gas Packaging Dynamics & Sparging", fontsize=16, fontweight="bold", y=0.98)

    # ----------------------------------------------------
    # Panel 1: Henry's Law DO and CO2 Solubility vs Temperature
    # ----------------------------------------------------
    ax = axes[0, 0]
    temps = np.linspace(5, 30, 30)
    do_12pct = [gas_eng.calculate_henry_oxygen_solubility(t, 12.0) for t in temps]
    do_14pct = [gas_eng.calculate_henry_oxygen_solubility(t, 14.5) for t in temps]

    ax.plot(temps, do_12pct, label="DO Saturation (12.0% v/v EtOH)", color="#1f77b4", lw=2.2)
    ax.plot(temps, do_14pct, label="DO Saturation (14.5% v/v EtOH)", color="#2ca02c", linestyle="--", lw=2.0)
    ax.set_title("Equilibrium Dissolved Oxygen (DO) Solubility")
    ax.set_xlabel("Wine Temperature (°C)")
    ax.set_ylabel("Saturated Dissolved Oxygen (mg/L)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    ax_co2 = ax.twinx()
    co2_sol = [gas_eng.calculate_henry_co2_solubility(t, 1.0) for t in temps]
    ax_co2.plot(temps, co2_sol, label="CO2 Solubility (1 atm)", color="#d62728", lw=2.0, linestyle=":")
    ax_co2.set_ylabel("CO2 Henry Solubility (g/L)", color="#d62728")
    ax_co2.tick_params(axis="y", labelcolor="#d62728")

    # ----------------------------------------------------
    # Panel 2: Counter-Current N2 Sparging Deoxygenation Curve
    # ----------------------------------------------------
    ax = axes[0, 1]
    n2_ratios = np.linspace(0.0, 1.2, 40)
    init_dos = [1.8, 1.2, 0.8]
    do_colors = ["#d62728", "#ff7f0e", "#1f77b4"]

    for init_do, col in zip(init_dos, do_colors):
        do_curve = [
            gas_eng.simulate_nitrogen_sparging(init_do, nitrogen_flow_ratio_nl_l=r)["final_do_mg_l"]
            for r in n2_ratios
        ]
        ax.plot(n2_ratios, do_curve, label=f"Initial DO = {init_do} mg/L", color=col, lw=2.2)

    ax.axhline(0.40, color="#2ca02c", linestyle="--", lw=1.5, label="Target Bottling Spec (< 0.40 mg/L)")
    ax.set_title("Inline Micro-Bubble N2 Sparging Deoxygenation")
    ax.set_xlabel("Nitrogen Ratio (Norm-Liters N2 / Liter Wine)")
    ax.set_ylabel("Residual Dissolved Oxygen (mg/L)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 3: Total Packaging Oxygen (TPO) Breakdown
    # ----------------------------------------------------
    ax = axes[1, 0]
    categories = ["Non-Inerted Air", "Inert N2 Purge", "Vacuum Corker", "Ultra-Low Spec"]
    do_wine_vals = [1.10, 0.50, 0.40, 0.25]
    hs_oxygen_vals = [2.25, 0.42, 0.22, 0.12]

    x = np.arange(len(categories))
    width = 0.5

    p1 = ax.bar(x, do_wine_vals, width, label="Dissolved Oxygen (DO wine)", color="#1f77b4", alpha=0.85)
    p2 = ax.bar(x, hs_oxygen_vals, width, bottom=do_wine_vals, label="Headspace Oxygen (eq mg/L)", color="#ff7f0e", alpha=0.85)

    ax.axhline(1.0, color="#2ca02c", linestyle="--", lw=1.5, label="Premium TPO Benchmark (< 1.0 mg/L)")
    ax.set_title("Total Packaging Oxygen (TPO) Compartmental Partitioning")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylabel("Total Packaging Oxygen (mg/L)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 4: Antioxidant Free SO2 Depletion Impact
    # ----------------------------------------------------
    ax = axes[1, 1]
    tpo_levels = [0.6, 1.2, 2.0, 3.2]
    colors_tpo = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    labels_tpo = ["TPO 0.6 mg/L (Ultra-Low)", "TPO 1.2 mg/L (Commercial)", "TPO 2.0 mg/L (Elevated)", "TPO 3.2 mg/L (Air Ingress)"]

    months = np.arange(0, 25)
    for tpo_val, c, lbl in zip(tpo_levels, colors_tpo, labels_tpo):
        # 1 mg TPO consumes 4 mg Free SO2 rapidly (first 2-3 months)
        initial_so2 = 30.0
        so2_trajectory = []
        for m in months:
            tpo_consumed = tpo_val * (1.0 - math.exp(-0.8 * m))
            so2_left = max(0.0, initial_so2 - tpo_consumed * 4.0 - 0.25 * m)
            so2_trajectory.append(so2_left)
        ax.plot(months, so2_trajectory, label=lbl, color=c, lw=2.2)

    ax.axhline(10.0, color="#d62728", linestyle="--", lw=1.5, label="PremOx Boundary (< 10 mg/L)")
    ax.set_title("Antioxidant Free SO2 Rapid Depletion from TPO")
    ax.set_xlabel("Time Post-Bottling (Months)")
    ax.set_ylabel("Free SO2 Concentration (mg/L)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "day14_dissolved_gas_and_sparging.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure_4_closure_shelf_life():
    """Generates Packaging Closure Shelf-Life & Browning 4-panel diagnostic figure."""
    shelf_sim = ClosureShelfLifeForecaster()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 14: Packaging Closure Permeation & Predictive Shelf-Life", fontsize=16, fontweight="bold", y=0.98)

    # ----------------------------------------------------
    # Panel 1: Multi-Closure OTR Profiles (Steady vs Desorption)
    # ----------------------------------------------------
    ax = axes[0, 0]
    closures_to_plot = ["natural_cork_fleur", "diam_5_microagglomerated", "synthetic_nomacorc_select", "screwcap_saran_tin", "screwcap_saranex"]
    c_names = ["Natural Cork", "Diam 5", "Nomacorc", "Screwcap Saran-Tin", "Screwcap Saranex"]
    c_colors = ["#8c564b", "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

    m_range = np.arange(0, 37)
    for c_key, name, col in zip(closures_to_plot, c_names, c_colors):
        prof = CLOSURE_PROFILES[c_key]
        steady_m = prof["steady_otr_mg_year"] / 12.0
        init_desorp = prof["initial_desorption_mg"]
        otr_vals = [(steady_m + (init_desorp / 3.0 if m < 3 else 0.0)) for m in m_range]
        ax.plot(m_range, otr_vals, label=name, color=col, lw=2.2)

    ax.set_title("Oxygen Ingress Rate Over Time (Desorption + Steady OTR)")
    ax.set_xlabel("Months in Bottle")
    ax.set_ylabel("Oxygen Ingress (mg O2 / bottle / month)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 2: Free SO2 Depletion Kinetics (48 Months)
    # ----------------------------------------------------
    ax = axes[0, 1]
    for c_key, name, col in zip(closures_to_plot, c_names, c_colors):
        res = shelf_sim.simulate_bottle_shelf_life(
            closure_type=c_key, initial_free_so2=32.0, initial_ph=3.45,
            initial_tpo_mg_l=0.85, storage_temperature_c=14.0, duration_months=48
        )
        traj = res["trajectory"]
        ax.plot(traj["month"], traj["free_so2_mg_l"], label=name, color=col, lw=2.2)

    ax.axhline(10.0, color="#d62728", linestyle="--", lw=1.5, label="PremOx Boundary (< 10 mg/L)")
    ax.set_title("Free SO2 Protection Depletion Across Closures (14°C)")
    ax.set_xlabel("Storage Duration (Months)")
    ax.set_ylabel("Free SO2 (mg/L)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 3: Browning Index (A420) Optical Density Evolution
    # ----------------------------------------------------
    ax = axes[1, 0]
    for c_key, name, col in zip(closures_to_plot, c_names, c_colors):
        res = shelf_sim.simulate_bottle_shelf_life(
            closure_type=c_key, initial_free_so2=32.0, initial_ph=3.45,
            initial_tpo_mg_l=0.85, storage_temperature_c=14.0, duration_months=48
        )
        traj = res["trajectory"]
        ax.plot(traj["month"], traj["a420_browning"], label=name, color=col, lw=2.2)

    ax.axhline(0.35, color="#ff7f0e", linestyle=":", lw=1.5, label="Noticeable Color Aging (0.35 OD)")
    ax.axhline(0.45, color="#d62728", linestyle="--", lw=1.5, label="Commercial Oxidation Limit (0.45 OD)")
    ax.set_title(r"Spectrophotometric Browning Index ($A_{420}$) Kinetics")
    ax.set_xlabel("Storage Duration (Months)")
    ax.set_ylabel(r"Optical Density $A_{420}$ (420 nm)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)

    # ----------------------------------------------------
    # Panel 4: Thermal Storage Acceleration Impact (Arrhenius)
    # ----------------------------------------------------
    ax = axes[1, 1]
    temps = [12.0, 18.0, 24.0, 30.0]
    temp_labels = ["12°C (Cold Cellar)", "18°C (Standard Cellar)", "24°C (Retail Ambient)", "30°C (Heat Abuse)"]
    temp_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    for t_val, lbl, col in zip(temps, temp_labels, temp_colors):
        res = shelf_sim.simulate_bottle_shelf_life(
            closure_type="diam_5_microagglomerated", initial_free_so2=32.0, initial_ph=3.45,
            initial_tpo_mg_l=0.85, storage_temperature_c=t_val, duration_months=48
        )
        traj = res["trajectory"]
        ax.plot(traj["month"], traj["freshness_score"], label=lbl, color=col, lw=2.2)

    ax.axhline(50.0, color="#d62728", linestyle="--", lw=1.5, label="End of Commercial Quality (< 50)")
    ax.set_title("Organoleptic Freshness Decay vs Storage Temperature")
    ax.set_xlabel("Storage Duration (Months)")
    ax.set_ylabel("Organoleptic Freshness Index (0 - 100)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "day14_closure_permeation_and_shelf_life.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_all_day14_figures():
    """Generates all four publication-grade Day 14 figures."""
    print("Generating Figure 1: Tartrate & Protein Stabilization...")
    generate_figure_1_stabilization()
    print("Generating Figure 2: Crossflow Membrane Filtration...")
    generate_figure_2_filtration()
    print("Generating Figure 3: Dissolved Gas Packaging & Sparging...")
    generate_figure_3_dissolved_gas()
    print("Generating Figure 4: Packaging Closure Shelf-Life...")
    generate_figure_4_closure_shelf_life()
    print("All Day 14 figures generated successfully!")


if __name__ == "__main__":
    generate_all_day14_figures()
