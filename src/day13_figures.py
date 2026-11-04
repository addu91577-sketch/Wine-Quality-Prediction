"""
Day 13 Diagnostic Figures Generator.

Generates 4 publication-quality enological figures:
1. reports/figures/day13_malolactic_fermentation_kinetics.png
2. reports/figures/day13_oak_maturation_wood_extractives.png
3. reports/figures/day13_cellar_master_dispatch_matrix.png
4. reports/figures/day13_haccp_compliance_and_biogenic_risk.png
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

from src.malolactic_and_oak import (
    MalolacticFermentationSimulator,
    OakBarrelMaturationSimulator,
    CellarMasterDispatchEngine,
    EnologicalHaccpAuditor
)

REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling standards
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def generate_figure_1_malolactic_kinetics():
    """Generates 4-panel secondary malolactic fermentation bioprocess dynamics."""
    sim = MalolacticFermentationSimulator()
    res_std = sim.simulate(initial_malic_acid=2.8, initial_ph=3.35, alcohol=13.0, free_so2=5.0, temperature=20.0, duration_days=35)
    res_cold = sim.simulate(initial_malic_acid=2.8, initial_ph=3.35, alcohol=13.0, free_so2=5.0, temperature=13.0, duration_days=35)
    res_so2 = sim.simulate(initial_malic_acid=2.8, initial_ph=3.35, alcohol=13.0, free_so2=16.0, temperature=20.0, duration_days=35)

    traj = res_std["trajectory"]
    days = traj["day"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 13: Secondary Malolactic Fermentation (MLF) Biotransformation Kinetics", fontsize=16, fontweight="bold", y=0.98)

    # 1. Malic depletion vs Lactic formation
    ax1 = axes[0, 0]
    ax1.plot(days, traj["malic_acid_g_l"], color="#800020", linewidth=2.5, label="L-Malic Acid (g/L)")
    ax1.plot(days, traj["lactic_acid_g_l"], color="#2E7D32", linewidth=2.5, linestyle="-.", label="L-Lactic Acid (g/L)")
    ax1.axhline(0.10, color="gray", linestyle=":", linewidth=1.5, label="Dry MLF Threshold (0.10 g/L)")
    ax1.set_title("1. Organic Acid Bioconversion ($C_4H_6O_5 \\to C_3H_6O_3$)", fontweight="bold")
    ax1.set_xlabel("Fermentation Time (Days)")
    ax1.set_ylabel("Acid Concentration (g/L)")
    ax1.grid(True)
    ax1.legend(loc="center right", framealpha=0.9)

    # 2. Diacetyl synthesis & enzymatic reduction
    ax2 = axes[0, 1]
    ax2.plot(days, traj["diacetyl_mg_l"], color="#D4AF37", linewidth=2.5, label="Diacetyl (Buttery Aroma)")
    ax2.axhline(0.20, color="#E65100", linestyle="--", linewidth=1.5, label="Sensory Perception Threshold (~0.2 mg/L)")
    ax2.axhline(2.0, color="#C62828", linestyle=":", linewidth=1.5, label="Excessive Spoilage Ceiling (~2.0 mg/L)")
    peak_val = max(traj["diacetyl_mg_l"])
    peak_idx = traj["diacetyl_mg_l"].index(peak_val)
    ax2.annotate(f"Peak Diacetyl: {peak_val:.2f} mg/L\n(Day {days[peak_idx]})",
                 xy=(days[peak_idx], peak_val), xytext=(days[peak_idx] + 4, peak_val * 0.8),
                 arrowprops=dict(facecolor="#333", shrink=0.05, width=1.2, headwidth=6),
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D"))
    ax2.set_title("2. Diacetyl Flavor Synthesis & Reductase Clearance", fontweight="bold")
    ax2.set_xlabel("Fermentation Time (Days)")
    ax2.set_ylabel("Diacetyl Concentration (mg/L)")
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.9)

    # 3. Fixed Acidity drop and pH elevation
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    p1 = ax3.plot(days, traj["fixed_acidity_g_l"], color="#1565C0", linewidth=2.5, label="Fixed Acidity (g/L)")
    p2 = ax3_twin.plot(days, traj["ph"], color="#C2185B", linewidth=2.5, linestyle="--", label="Wine pH")
    ax3.set_title("3. Physicochemical Deacidification & pH Shift", fontweight="bold")
    ax3.set_xlabel("Fermentation Time (Days)")
    ax3.set_ylabel("Fixed Acidity (g/L tartaric eq)", color="#1565C0")
    ax3_twin.set_ylabel("Measured pH", color="#C2185B")
    ax3.grid(True)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="center right", framealpha=0.9)

    # 4. Stress inhibition comparison
    ax4 = axes[1, 1]
    ax4.plot(days, traj["conversion_rate_pct"], color="#2E7D32", linewidth=2.5, label="Optimal (20°C, 5 mg/L Free SO2)")
    ax4.plot(days, res_cold["trajectory"]["conversion_rate_pct"], color="#0288D1", linewidth=2.0, linestyle="--", label="Cold Cellar (13°C)")
    ax4.plot(days, res_so2["trajectory"]["conversion_rate_pct"], color="#D32F2F", linewidth=2.0, linestyle=":", label="Inhibited (16 mg/L Free SO2)")
    ax4.axhline(90.0, color="gray", linestyle="-.", label="Commercial Completion (90%)")
    ax4.set_title("4. Environmental Gating & Inhibition Impact", fontweight="bold")
    ax4.set_xlabel("Fermentation Time (Days)")
    ax4.set_ylabel("MLF Malic Conversion (%)")
    ax4.set_ylim(-5, 105)
    ax4.grid(True)
    ax4.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "day13_malolactic_fermentation_kinetics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_2_oak_maturation():
    """Generates 4-panel precision oak wood phenolics extraction and color stabilization figure."""
    sim = OakBarrelMaturationSimulator()
    oak_fr = sim.simulate("french", "medium_plus", "new", 24)
    oak_us = sim.simulate("american", "medium", "new", 24)
    oak_hu = sim.simulate("hungarian", "medium_plus", "new", 24)

    months = oak_fr["trajectory"]["month"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Day 13: Precision Oak Barrel Maturation & Wood Phenolics Extraction Dynamics", fontsize=16, fontweight="bold", y=0.98)

    # 1. Ellagitannins extraction (Structure & Mouthfeel)
    ax1 = axes[0, 0]
    ax1.plot(months, oak_fr["trajectory"]["ellagitannins_mg_l"], color="#4A148C", linewidth=2.5, label="French Oak (Q. petraea) - High Tannin")
    ax1.plot(months, oak_hu["trajectory"]["ellagitannins_mg_l"], color="#E65100", linewidth=2.5, linestyle="--", label="Hungarian Oak - Balanced")
    ax1.plot(months, oak_us["trajectory"]["ellagitannins_mg_l"], color="#1565C0", linewidth=2.5, linestyle=":", label="American Oak (Q. alba) - Low Tannin")
    ax1.set_title("1. Ellagitannins Dissolution Kinetics (g/L)", fontweight="bold")
    ax1.set_xlabel("Maturation Duration (Months)")
    ax1.set_ylabel("Ellagitannins (mg/L)")
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. Sweet Aromatics: Whiskey lactone & Vanillin
    ax2 = axes[0, 1]
    ax2.plot(months, oak_us["trajectory"]["whiskey_lactones_mg_l"], color="#BF360C", linewidth=2.5, label="American Oak Lactones (Coconut/Sweet)")
    ax2.plot(months, oak_fr["trajectory"]["whiskey_lactones_mg_l"], color="#4A148C", linewidth=2.0, linestyle="--", label="French Oak Lactones")
    ax2.plot(months, oak_us["trajectory"]["vanillin_mg_l"], color="#F57F17", linewidth=2.5, linestyle="-.", label="American Oak Vanillin")
    ax2.plot(months, oak_fr["trajectory"]["vanillin_mg_l"], color="#33691E", linewidth=2.0, linestyle=":", label="French Oak Vanillin")
    ax2.set_title("2. Oak Lactone & Vanillin Aromatic Extraction", fontweight="bold")
    ax2.set_xlabel("Maturation Duration (Months)")
    ax2.set_ylabel("Aroma Concentration (mg/L)")
    ax2.grid(True)
    ax2.legend(loc="upper left", framealpha=0.9)

    # 3. Toast Level Impact on Guaiacol Smoke
    ax3 = axes[1, 0]
    toast_levels = ["light", "medium", "medium_plus", "heavy"]
    guaiacol_vals = [sim.simulate("french", t, "new", 18)["final_concentrations"]["guaiacol_mg_l"] for t in toast_levels]
    furfural_vals = [sim.simulate("french", t, "new", 18)["final_concentrations"]["furfural_mg_l"] for t in toast_levels]
    
    x = np.arange(len(toast_levels))
    width = 0.35
    ax3.bar(x - width/2, guaiacol_vals, width, color="#424242", label="Guaiacol (Roasted Smoke, mg/L)")
    ax3.bar(x + width/2, furfural_vals, width, color="#D84315", label="Furfural (Toasted Caramel, mg/L)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Light", "Medium", "Medium+", "Heavy"])
    ax3.set_title("3. Toast Pyro-Conversion Yields (at 18 Months)", fontweight="bold")
    ax3.set_ylabel("Concentration (mg/L)")
    ax3.grid(True, axis="y")
    ax3.legend(loc="upper left", framealpha=0.9)

    # 4. Color Stabilization & Astringency Softening
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    p1 = ax4.plot(months, oak_fr["trajectory"]["polymeric_pigment_index"], color="#880E4F", linewidth=2.5, label="Polymeric Pigment Index")
    p2 = ax4_twin.plot(months, oak_fr["trajectory"]["astringency_softening_pct"], color="#00695C", linewidth=2.5, linestyle="--", label="Astringency Softening (%)")
    ax4.set_title("4. Micro-Oxygenation & Tannin Polymerization", fontweight="bold")
    ax4.set_xlabel("Maturation Duration (Months)")
    ax4.set_ylabel("Polymeric Pigment Index", color="#880E4F")
    ax4_twin.set_ylabel("Sensory Softening (%)", color="#00695C")
    ax4.grid(True)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="lower right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "day13_oak_maturation_wood_extractives.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_3_cellar_dispatch():
    """Generates cellar master automated dispatch triage matrix and work order priority plot."""
    dispatch = CellarMasterDispatchEngine()
    fleet = [
        {"vessel_id": "T-FERM-01", "vessel_type": "TANK", "cultivar": "red", "volume_liters": 5000.0, "ph": 3.75, "free_so2": 6.0, "total_so2": 45.0, "volatile_acidity": 0.88, "alcohol": 12.8, "temperature": 23.0, "residual_sugar": 5.2, "malic_acid": 0.65, "maturation_days": 25, "status": "aging"},
        {"vessel_id": "B-RES-04", "vessel_type": "BARREL", "cultivar": "red", "volume_liters": 225.0, "ph": 3.45, "free_so2": 12.0, "total_so2": 60.0, "volatile_acidity": 0.48, "alcohol": 13.5, "temperature": 17.5, "residual_sugar": 1.9, "malic_acid": 0.04, "maturation_days": 180, "status": "aging"},
        {"vessel_id": "B-NEW-09", "vessel_type": "BARREL", "cultivar": "red", "volume_liters": 225.0, "ph": 3.38, "free_so2": 24.0, "total_so2": 75.0, "volatile_acidity": 0.38, "alcohol": 13.8, "temperature": 16.0, "residual_sugar": 1.6, "malic_acid": 0.02, "maturation_days": 90, "status": "aging"},
        {"vessel_id": "T-WHT-02", "vessel_type": "TANK", "cultivar": "white", "volume_liters": 8000.0, "ph": 3.18, "free_so2": 22.0, "total_so2": 95.0, "volatile_acidity": 0.32, "alcohol": 12.2, "temperature": 12.5, "residual_sugar": 4.5, "malic_acid": 1.60, "maturation_days": 45, "status": "resting"},
        {"vessel_id": "B-OLD-14", "vessel_type": "BARREL", "cultivar": "red", "volume_liters": 225.0, "ph": 3.82, "free_so2": 7.0, "total_so2": 85.0, "volatile_acidity": 1.10, "alcohol": 14.2, "temperature": 21.0, "residual_sugar": 2.2, "malic_acid": 0.40, "maturation_days": 320, "status": "aging"},
        {"vessel_id": "T-COU-05", "vessel_type": "TANK", "cultivar": "red", "volume_liters": 12000.0, "ph": 3.52, "free_so2": 16.0, "total_so2": 70.0, "volatile_acidity": 0.55, "alcohol": 13.1, "temperature": 18.0, "residual_sugar": 2.4, "malic_acid": 0.12, "maturation_days": 75, "status": "aging"},
        {"vessel_id": "B-CHA-11", "vessel_type": "BARREL", "cultivar": "white", "volume_liters": 225.0, "ph": 3.32, "free_so2": 18.0, "total_so2": 80.0, "volatile_acidity": 0.42, "alcohol": 13.0, "temperature": 16.5, "residual_sugar": 2.0, "malic_acid": 0.05, "maturation_days": 140, "status": "aging"},
        {"vessel_id": "P-PUN-02", "vessel_type": "PUNCHEON", "cultivar": "red", "volume_liters": 500.0, "ph": 3.42, "free_so2": 26.0, "total_so2": 78.0, "volatile_acidity": 0.36, "alcohol": 13.6, "temperature": 15.5, "residual_sugar": 1.5, "malic_acid": 0.03, "maturation_days": 210, "status": "resting"},
    ]
    res = dispatch.dispatch_cellar_inventory(fleet)
    vessels = res["dispatched_vessels"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Day 13: Digital Cellar Master Fleet Triage & Automated Dispatch Matrix", fontsize=16, fontweight="bold", y=0.98)

    # 1. Bubble plot of ERPI risk vs Free SO2 deficit
    color_map = {
        "CRITICAL_INTERVENTION": "#D32F2F",
        "ACTIVE_MANAGEMENT": "#F57C00",
        "ROUTINE_MONITORING": "#1976D2",
        "STABLE_CELLAR": "#388E3C"
    }

    for v in vessels:
        tier = v["triage"]["action_tier"]
        erpi = v["metrics"]["erpi_score"]
        deficit = v["metrics"]["free_so2_deficit_mg_l"]
        vol = v["volume_liters"]
        size = 80 + math.sqrt(vol) * 8
        color = color_map.get(tier, "#757575")

        ax1.scatter(deficit, erpi, s=size, color=color, alpha=0.75, edgecolors="#212121", linewidth=1.5)
        ax1.annotate(v["vessel_id"], xy=(deficit, erpi), xytext=(deficit + 0.3, erpi + 1.2),
                     fontsize=9, fontweight="bold")

    ax1.axhline(68.0, color="#D32F2F", linestyle="--", linewidth=1.5, label="Critical Emergency Ceiling (ERPI 68)")
    ax1.axhline(42.0, color="#F57C00", linestyle=":", linewidth=1.5, label="Active Triage Boundary (ERPI 42)")
    ax1.set_title("1. Fleet Risk Priority Index (ERPI) vs Preservative Deficit", fontweight="bold")
    ax1.set_xlabel("Free SO2 Deficit Below Target (mg/L)")
    ax1.set_ylabel("Enological Risk Priority Index (0 - 100)")
    ax1.set_xlim(-1, 25)
    ax1.set_ylim(0, 105)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. Reagent Requirements & Triage Distribution
    tiers = ["CRITICAL_INTERVENTION", "ACTIVE_MANAGEMENT", "ROUTINE_MONITORING", "STABLE_CELLAR"]
    tier_labels = ["Critical (24h)", "Active (72h)", "Routine (7d)", "Stable"]
    tier_counts = [res["tier_distribution"].get(t, 0) for t in tiers]
    colors = [color_map[t] for t in tiers]

    bars = ax2.bar(tier_labels, tier_counts, color=colors, width=0.55, edgecolor="#212121", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{int(h)} vessels", ha="center", va="bottom", fontweight="bold")

    ax2.set_title(f"2. Cellar Action Triage (Total SO2 Reagent: {res['aggregate_pms_reagents_required_kg']:.3f} kg PMS)", fontweight="bold")
    ax2.set_ylabel("Vessel Count")
    ax2.set_ylim(0, max(tier_counts) + 2)
    ax2.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = REPORTS_DIR / "day13_cellar_master_dispatch_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_4_haccp_safety():
    """Generates HACCP food safety compliance boundaries and biogenic amine risk contour."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Day 13: HACCP Critical Control Point (CCP) & Biogenic Amine Safety Audit", fontsize=16, fontweight="bold", y=0.98)

    # 1. Molecular SO2 Safety Surface across pH and Free SO2
    ph_grid = np.linspace(3.0, 4.0, 100)
    so2_grid = np.linspace(5.0, 45.0, 100)
    PH, SO2 = np.meshgrid(ph_grid, so2_grid)
    MOL_SO2 = SO2 / (1.0 + 10.0 ** (PH - 1.81))

    contour1 = ax1.contourf(PH, SO2, MOL_SO2, levels=12, cmap="YlGnBu")
    cbar1 = fig.colorbar(contour1, ax=ax1)
    cbar1.set_label("Molecular SO2 (mg/L)", fontweight="bold")

    # Regulatory lines
    cs1 = ax1.contour(PH, SO2, MOL_SO2, levels=[0.50], colors=["#D32F2F"], linewidths=2.5, linestyles=["--"])
    ax1.clabel(cs1, inline=True, fmt="Red Minimum: 0.50 mg/L", fontsize=9)
    cs2 = ax1.contour(PH, SO2, MOL_SO2, levels=[0.80], colors=["#C2185B"], linewidths=2.5, linestyles=[":"])
    ax1.clabel(cs2, inline=True, fmt="White Minimum: 0.80 mg/L", fontsize=9)

    ax1.set_title("1. CCP-1 Antimicrobial Molecular SO2 Envelope", fontweight="bold")
    ax1.set_xlabel("Wine pH")
    ax1.set_ylabel("Measured Free SO2 (mg/L)")
    ax1.grid(True)

    # 2. Biogenic Amine Hazard Index Contour (pH vs Lingering Malic)
    malic_grid = np.linspace(0.0, 2.0, 100)
    PH2, MALIC = np.meshgrid(ph_grid, malic_grid)
    # BARI calculation proxy at 10 mg/L Free SO2 and 19°C
    BARI = (
        30.0 * np.clip((PH2 - 3.65) / 0.45, 0.0, 1.0) +
        25.0 * np.clip((MALIC - 0.15) / 1.5, 0.0, 1.0) +
        30.0 * (1.0 - (10.0 / 18.0)) +
        15.0 * np.clip((19.0 - 18.0) / 7.0, 0.0, 1.0)
    )

    contour2 = ax2.contourf(PH2, MALIC, BARI, levels=12, cmap="OrRd")
    cbar2 = fig.colorbar(contour2, ax=ax2)
    cbar2.set_label("Biogenic Amine Risk Index (0 - 100)", fontweight="bold")

    cs_safe = ax2.contour(PH2, MALIC, BARI, levels=[25.0], colors=["#2E7D32"], linewidths=2.0)
    ax2.clabel(cs_safe, inline=True, fmt="Safe Ceiling (25)", fontsize=9)
    cs_danger = ax2.contour(PH2, MALIC, BARI, levels=[60.0], colors=["#B71C1C"], linewidths=2.5, linestyles=["--"])
    ax2.clabel(cs_danger, inline=True, fmt="High Hazard Zone (60)", fontsize=9)

    ax2.set_title("2. Biogenic Amine Decarboxylation Hazard Contours", fontweight="bold")
    ax2.set_xlabel("Wine pH")
    ax2.set_ylabel("Residual Malic Acid (g/L)")
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = REPORTS_DIR / "day13_haccp_compliance_and_biogenic_risk.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_day13_figures():
    """Generates all Day 13 visual diagnostic figures."""
    print("Generating Day 13 figures...")
    generate_figure_1_malolactic_kinetics()
    generate_figure_2_oak_maturation()
    generate_figure_3_cellar_dispatch()
    generate_figure_4_haccp_safety()
    print("All Day 13 diagnostic figures generated successfully!")


if __name__ == "__main__":
    generate_all_day13_figures()
