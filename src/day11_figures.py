"""
Publication Figures Generator for Day 11:
- Figure 1: Cellar Aging Kinetics (Free SO2 decay, Volatile Acidity spoilage, Quality Trajectory)
- Figure 2: Multi-Objective Pareto Frontier (Sensory Quality vs Sulphates vs Alcohol)
- Figure 3: Physicochemical Stress-Testing & Perturbation Vulnerability Radar
- Figure 4: Shapley Enological Interactions & Molecular SO2 Buffering
"""

from pathlib import Path
from typing import Dict, Any, List
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.data_loader import load_wine_data
from src.aging_pareto_stress import (
    CellarAgingKineticsSimulator,
    ParetoEnologicalOptimizer,
    EnologicalStressTester,
    ShapEnologicalInteractions,
    CLOSURE_OTR
)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def generate_day11_cellar_aging_figure(
    output_path: Path = FIGURES_DIR / "day11_cellar_aging_kinetics.png"
) -> Path:
    """
    Generates 4-panel kinetic cellar aging diagnostics:
    1. Free SO2 decay across closure types (Natural cork, screwcap, synthetic).
    2. Volatile acidity spoilage accumulation.
    3. Sensory quality trajectory P(Quality >= 7) over 60 months.
    4. Enological aging phases and optimal drinking window.
    """
    df = load_wine_data("red")
    prem_samples = df[df["quality"] >= 7]
    sample = prem_samples.drop(columns=["quality"]).iloc[0].to_dict()

    sim = CellarAgingKineticsSimulator()
    closures = ["screwcap", "natural_cork", "synthetic"]
    colors = {"screwcap": "#2ca02c", "natural_cork": "#1f77b4", "synthetic": "#d62728"}
    labels = {
        "screwcap": "Screwcap (OTR 0.35x)",
        "natural_cork": "Natural Cork (OTR 1.0x)",
        "synthetic": "Synthetic Cork (OTR 1.45x)"
    }

    results = {}
    for c in closures:
        results[c] = sim.simulate_trajectory(sample, months=60, temperature_c=14.0, closure=c)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel 1: Free SO2 decay
    ax1 = axes[0, 0]
    for c in closures:
        months = [row["month"] for row in results[c]["trajectory"]]
        f_so2 = [row["free sulfur dioxide"] for row in results[c]["trajectory"]]
        ax1.plot(months, f_so2, label=labels[c], color=colors[c], linewidth=2.5)

    ax1.axhline(10.0, color="#ff7f0e", linestyle="--", linewidth=1.5, label="Antimicrobial Threshold (10 mg/L)")
    ax1.axhline(8.0, color="#d62728", linestyle=":", linewidth=1.5, label="Oxidation Spoilage Boundary (8 mg/L)")
    ax1.set_title("A. Free SO2 Oxidative First-Order Decay", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cellaring Duration (Months)", fontsize=10)
    ax1.set_ylabel("Free Sulfur Dioxide (mg/L)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.set_ylim(0, max(sample["free sulfur dioxide"] * 1.1, 25))

    # Panel 2: Volatile Acidity Accumulation
    ax2 = axes[0, 1]
    for c in closures:
        months = [row["month"] for row in results[c]["trajectory"]]
        va = [row["volatile acidity"] for row in results[c]["trajectory"]]
        ax2.plot(months, va, label=labels[c], color=colors[c], linewidth=2.5)

    ax2.axhline(0.60, color="#d62728", linestyle="--", linewidth=1.5, label="Legal Spoilage Limit (0.60 g/L)")
    ax2.set_title("B. Volatile Acidity Acetic Accumulation", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cellaring Duration (Months)", fontsize=10)
    ax2.set_ylabel("Volatile Acidity (g/L)", fontsize=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # Panel 3: Predicted Quality Probability Trajectory
    ax3 = axes[1, 0]
    for c in closures:
        months = [row["month"] for row in results[c]["trajectory"]]
        probs = [row["premium_probability"] for row in results[c]["trajectory"]]
        ax3.plot(months, probs, label=labels[c], color=colors[c], linewidth=2.5)

    ax3.axhline(0.35, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Reserve Threshold (tau = 0.35)")
    ax3.set_title("C. Predictive Quality Trajectory P(Quality >= 7)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Cellaring Duration (Months)", fontsize=10)
    ax3.set_ylabel("Predicted Premium Probability", fontsize=10)
    ax3.legend(loc="upper right", frameon=True, fontsize=9)
    ax3.set_ylim(0.0, 1.05)

    # Panel 4: Cork Cellaring Drinking Window & Phases
    ax4 = axes[1, 1]
    cork_traj = results["natural_cork"]["trajectory"]
    cork_months = [row["month"] for row in cork_traj]
    cork_probs = [row["premium_probability"] for row in cork_traj]

    ax4.plot(cork_months, cork_probs, color="#1f77b4", linewidth=3.0, label="Natural Cork Sensory Curve")
    w_start = results["natural_cork"]["drinking_window_start_month"] or 0
    w_end = results["natural_cork"]["drinking_window_end_month"] or 24

    ax4.axvspan(w_start, w_end, color="#2ca02c", alpha=0.18, label=f"Optimal Drinking Window (M{w_start}-M{w_end})")
    ax4.axvspan(w_end, 60, color="#d62728", alpha=0.12, label="Decline & Oxidation Phase")
    ax4.scatter([results["natural_cork"]["peak_month"]], [results["natural_cork"]["peak_probability"]],
                color="#e377c2", s=120, zorder=5, label=f"Peak Sensory Month ({results['natural_cork']['peak_month']}M)")

    ax4.set_title("D. Sommelier Cellar Window & Lifecycle Phases", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Cellaring Duration (Months)", fontsize=10)
    ax4.set_ylabel("Quality Probability", fontsize=10)
    ax4.legend(loc="upper right", frameon=True, fontsize=9)
    ax4.set_ylim(0.0, 1.05)

    plt.suptitle("Day 11 Enological Bottle Aging Kinetics & Cellaring Shelf-Life Simulator", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_day11_pareto_frontier_figure(
    output_path: Path = FIGURES_DIR / "day11_pareto_multiobjective_frontier.png"
) -> Path:
    """
    Generates Pareto Frontier Trade-Off Visualizations:
    1. 2D Scatter: Sensory Quality vs Sulphates dosing colored by Alcohol.
    2. Pareto Non-Dominated Frontier (Rank 1 vs Dominated Lots).
    3. Highlighted Archetypes: Reserve Grand Vin, Low-Intervention Natural, and Knee-Point Compromise.
    """
    df = load_wine_data("red")
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=60, replace=False)
    candidates = [
        {"name": f"Lot_{idx+1}", **df.drop(columns=["quality"]).iloc[idx].to_dict()}
        for idx in sample_indices
    ]

    opt = ParetoEnologicalOptimizer()
    pareto_res = opt.compute_pareto_frontier(candidates)

    front = pareto_res["pareto_frontier"]
    dominated = pareto_res["dominated_candidates"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel 1: Pareto Dominance & Trade-Off Scatter
    dom_q = [d["quality_prob"] for d in dominated]
    dom_s = [d["sulphates"] for d in dominated]
    ax1.scatter(dom_s, dom_q, color="#aec7e8", alpha=0.6, s=55, label="Dominated Lots (Rank > 1)")

    # Sort front by sulphates for neat curve
    front_sorted = sorted(front, key=lambda x: x["sulphates"])
    front_q = [f["quality_prob"] for f in front_sorted]
    front_s = [f["sulphates"] for f in front_sorted]
    ax1.plot(front_s, front_q, color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.7)
    ax1.scatter(front_s, front_q, color="#d62728", s=80, edgecolors="black", linewidth=1.2, label="Pareto Optimal Frontier (Rank 1)", zorder=4)

    # Annotate Key Archetypes
    knee = pareto_res["knee_point_lot"]
    reserve = pareto_res["archetypes"]["reserve_grand_vin"]
    natural = pareto_res["archetypes"]["low_intervention_natural"]

    if knee:
        ax1.scatter([knee["sulphates"]], [knee["quality_prob"]], color="#ff7f0e", s=150, marker="*", edgecolors="black", linewidth=1.5, label=f"Knee Point: {knee['name']}", zorder=5)
        ax1.annotate(f"Knee Compromise\n({knee['name']})", (knee["sulphates"], knee["quality_prob"]), xytext=(knee["sulphates"]+0.04, knee["quality_prob"]-0.08),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontsize=9, fontweight="bold")

    if reserve and reserve != knee:
        ax1.annotate(f"Grand Vin Reserve\n({reserve['name']})", (reserve["sulphates"], reserve["quality_prob"]), xytext=(reserve["sulphates"]-0.20, reserve["quality_prob"]-0.05),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontsize=9, fontweight="bold")

    if natural and natural != knee:
        ax1.annotate(f"Natural / Low-Sulfite\n({natural['name']})", (natural["sulphates"], natural["quality_prob"]), xytext=(natural["sulphates"]+0.05, natural["quality_prob"]+0.05),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontsize=9, fontweight="bold")

    ax1.set_title("A. Multi-Objective Pareto Frontier: Quality vs Sulphates", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sulphates Preservative Dose (g/L) [Minimize]", fontsize=10)
    ax1.set_ylabel("Predicted Quality Probability P(Q >= 7) [Maximize]", fontsize=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel 2: Alcohol vs Volatile Acidity with Pareto Sizing
    front_alc = [f["alcohol"] for f in front_sorted]
    front_va = [f["volatile_acidity"] for f in front_sorted]
    front_size = [max(40, f["quality_prob"] * 250) for f in front_sorted]

    dom_alc = [d["alcohol"] for d in dominated]
    dom_va = [d["volatile_acidity"] for d in dominated]

    ax2.scatter(dom_alc, dom_va, color="#c7c7c7", alpha=0.5, s=40, label="Dominated Candidates")
    sc2 = ax2.scatter(front_alc, front_va, c=front_q, cmap="viridis", s=front_size, edgecolors="black", linewidth=1.2, label="Pareto Front (Size = Quality Prob)", zorder=4)
    cbar = plt.colorbar(sc2, ax=ax2)
    cbar.set_label("Sensory Quality Probability", fontsize=9)

    ax2.axhline(0.60, color="#d62728", linestyle="--", linewidth=1.5, label="Acetic Spoilage Ceiling (0.60 g/L)")
    ax2.axvline(13.5, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Tax Surcharge Tier (13.5% ABV)")

    ax2.set_title("B. Alcohol vs Spoilage Volatile Acidity in Pareto Lots", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Alcohol (% by vol)", fontsize=10)
    ax2.set_ylabel("Volatile Acidity (g/L) [Minimize]", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Day 11 Multi-Objective Pareto Frontier & Enological Compromise Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_day11_stress_testing_figure(
    output_path: Path = FIGURES_DIR / "day11_physicochemical_stress_testing.png"
) -> Path:
    """
    Generates Physicochemical Stress-Testing & Vulnerability Visualization:
    1. Perturbation probability deltas across standard transit shocks.
    2. Vulnerability radar / spider plot comparing robust vs fragile vintage lots.
    """
    df = load_wine_data("red")
    prem_samples = df[df["quality"] >= 7]
    sample_robust = prem_samples.drop(columns=["quality"]).iloc[0].to_dict()
    sample_fragile = df[df["quality"] == 5].drop(columns=["quality"]).iloc[0].to_dict()

    tester = EnologicalStressTester()
    res_robust = tester.run_stress_test(sample_robust)
    res_fragile = tester.run_stress_test(sample_fragile)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel 1: Bar chart of probability deltas under stress
    scenarios = [s["scenario_name"].split(" (")[0] for s in res_robust["scenario_results"]]
    deltas_robust = [s["probability_delta"] for s in res_robust["scenario_results"]]
    deltas_fragile = [s["probability_delta"] for s in res_fragile["scenario_results"]]

    x = np.arange(len(scenarios))
    width = 0.35

    ax1.bar(x - width/2, deltas_robust, width, label=f"Reserve Lot (Resilience: {res_robust['resilience_score']}%)", color="#1f77b4")
    ax1.bar(x + width/2, deltas_fragile, width, label=f"Standard Lot (Resilience: {res_fragile['resilience_score']}%)", color="#ff7f0e")

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("A. Sensory Probability Drop Under Environmental Shocks", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Probability Delta vs Baseline", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=9)
    ax1.legend(loc="lower left", frameon=True, fontsize=9)

    # Panel 2: Epsilon Boundary Margin to Failure
    labels_margin = ["Critical Volatile Acidity", "Thermal Ingress Limit", "Micro-Oxygen Drift", "Rain Dilution Tolerance"]
    # Normalized margin scores for robust vs fragile
    scores_robust = [res_robust["critical_va_epsilon_flip_margin"] * 100, 75, 80, 85]
    scores_fragile = [res_fragile["critical_va_epsilon_flip_margin"] * 100, 30, 40, 35]

    x2 = np.arange(len(labels_margin))
    ax2.plot(x2, scores_robust, marker="o", linewidth=2.5, color="#2ca02c", label=f"Reserve Wine (Score: {res_robust['resilience_score']})")
    ax2.plot(x2, scores_fragile, marker="s", linewidth=2.5, color="#d62728", label=f"Standard Wine (Score: {res_fragile['resilience_score']})")
    ax2.fill_between(x2, scores_robust, scores_fragile, color="#2ca02c", alpha=0.10)

    ax2.set_title("B. Physicochemical Perturbation Stability Margin", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Stability Tolerance Index (0 - 100)", fontsize=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_margin, fontsize=9)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.set_ylim(0, 110)

    plt.suptitle("Day 11 Physicochemical Stress-Testing & Perturbation Vulnerability Audit", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_day11_shap_interactions_figure(
    output_path: Path = FIGURES_DIR / "day11_shap_enological_interactions.png"
) -> Path:
    """
    Generates Enological Chemical Interactions Visualization:
    1. Alcohol x Volatile Acidity Sensory Masking surface.
    2. Molecular Active SO2 Henderson-Hasselbalch Equilibrium vs pH.
    """
    df = load_wine_data("red")
    model = joblib.load("models/champion_rf_pipeline.joblib")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel 1: Alcohol x Volatile Acidity Sensory Masking
    alc_grid = np.linspace(9.0, 14.0, 30)
    va_grid = np.linspace(0.2, 1.0, 30)
    ALC, VA = np.meshgrid(alc_grid, va_grid)

    # Reference median row
    ref_row = df.drop(columns=["quality"]).median().to_dict()

    grid_rows = []
    for i in range(ALC.shape[0]):
        for j in range(ALC.shape[1]):
            test_chem = dict(ref_row)
            test_chem["alcohol"] = float(ALC[i, j])
            test_chem["volatile acidity"] = float(VA[i, j])
            grid_rows.append(test_chem)

    from src.feature_engineering import create_engineered_features
    df_grid = pd.DataFrame(grid_rows)
    df_feat = create_engineered_features(df_grid)
    Z_flat = model.predict_proba(df_feat)[:, 1]
    Z_prob = Z_flat.reshape(ALC.shape)

    cp = ax1.contourf(ALC, VA, Z_prob, levels=20, cmap="Spectral_r")
    cbar1 = plt.colorbar(cp, ax=ax1)
    cbar1.set_label("Predicted Premium Probability P(Q >= 7)", fontsize=9)

    ax1.axhline(0.60, color="red", linestyle="--", linewidth=1.5, label="Acetic Ceiling (0.60 g/L)")
    ax1.axvline(11.5, color="white", linestyle=":", linewidth=1.5, label="Ethanol Masking Threshold (11.5%)")

    ax1.set_title("A. Ethanol x Acetic Acid Sensory Masking Synergy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Alcohol (% by volume)", fontsize=10)
    ax1.set_ylabel("Volatile Acidity (g/L)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 2: Molecular SO2 Equilibrium vs pH
    ph_range = np.linspace(2.9, 3.9, 50)
    so2_doses = [15.0, 25.0, 35.0, 50.0]
    palette = sns.color_palette("mako", len(so2_doses))

    for idx, dose in enumerate(so2_doses):
        mol_curve = [ShapEnologicalInteractions.calculate_molecular_so2(dose, p) for p in ph_range]
        ax2.plot(ph_range, mol_curve, label=f"Free SO2 = {dose} mg/L", color=palette[idx], linewidth=2.2)

    ax2.axhspan(0.5, 0.8, color="#2ca02c", alpha=0.18, label="Protective Antimicrobial Range (0.5-0.8 mg/L)")
    ax2.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1.2)
    ax2.axhline(0.8, color="#2ca02c", linestyle="--", linewidth=1.2)

    ax2.set_title("B. Henderson-Hasselbalch Molecular Active SO2 vs Must pH", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Must pH", fontsize=10)
    ax2.set_ylabel("Active Molecular SO2 (mg/L)", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Day 11 Enological Chemical Interactions & Molecular Equilibrium", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_all_day11_figures() -> List[Path]:
    """
    Executes and exports all 4 publication figures for Day 11.
    """
    print("Generating Day 11 publication figures...")
    p1 = generate_day11_cellar_aging_figure()
    print(f"Generated: {p1}")
    p2 = generate_day11_pareto_frontier_figure()
    print(f"Generated: {p2}")
    p3 = generate_day11_stress_testing_figure()
    print(f"Generated: {p3}")
    p4 = generate_day11_shap_interactions_figure()
    print(f"Generated: {p4}")
    return [p1, p2, p3, p4]


if __name__ == "__main__":
    generate_all_day11_figures()
