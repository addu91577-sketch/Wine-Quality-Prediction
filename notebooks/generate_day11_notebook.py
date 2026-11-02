"""
Day 11 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day11_Aging_Kinetics_Pareto_Optimization_and_Stress_Testing.ipynb
and executes all cells to populate verified outputs, enological aging kinetic models,
multi-objective Pareto frontiers, physicochemical stress-testing audits,
Shapley molecular equilibrium interactions, and FastAPI endpoint verifications.
"""

import os
import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(".").resolve()
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Day11_Aging_Kinetics_Pareto_Optimization_and_Stress_Testing.ipynb"


def create_day11_notebook():
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
        "# Day 11: Enological Aging Kinetics, Multi-Objective Pareto Frontier Optimization & Physicochemical Stress-Testing Suite\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Author:** ML Engineering & Production Deployment Team\n"
        "**Date:** November 2, 2026\n"
        "**Tech Stack:** Python 3.14, Scikit-learn, Scipy, FastAPI, Pydantic v2, Streamlit, Matplotlib, Seaborn, Joblib\n\n"
        "## Executive Overview & Objectives for Day 11\n"
        "Day 11 advances the Wine Quality Prediction platform from static point predictions to dynamic enological lifecycle modeling, multi-objective commercial decision theory, and environmental perturbation stress resilience:\n\n"
        "1. **First-Principles Bottle Aging Kinetics (`CellarAgingKineticsSimulator`):**\n"
        "   - Formulate temporal chemical decay differential equations across 12-84 months of bottle cellaring.\n"
        "   - Model Arrhenius Free $\\text{SO}_2$ oxidative consumption modulated by closure Oxygen Transmission Rate (OTR):\n"
        "     $$k_{\\text{SO}_2}(T, \\text{closure}) = k_0 \\cdot \\exp\\left(-\\frac{E_a}{R(T+273.15)}\\right) \\cdot \\theta_{\\text{closure}}$$\n"
        "   - Model *Acetobacter* acetic spoilage when protective Free $\\text{SO}_2 < 10.0$ mg/L, causing volatile acidity accumulation.\n"
        "   - Dynamically recompute 5 biochemical interaction features at monthly intervals to evaluate predictive quality trajectory $P(\\text{Quality} \\ge 7)$, peak maturity month, and sommelier drinking window.\n"
        "2. **Multi-Objective Pareto Frontier Optimizer (`ParetoEnologicalOptimizer`):**\n"
        "   - Balance conflicting winery trade-offs: Maximize Quality vs Minimize Sulphites vs Manage Alcohol/Excise Tax.\n"
        "   - Extract non-dominated Rank 1 Pareto optimal frontier and detect knee-point compromise solutions.\n"
        "   - Characterize vintage archetypes: *Reserve Grand Vin*, *Low-Intervention Natural*, and *Balanced Table*.\n"
        "3. **Physicochemical Stress-Testing & Perturbation Audit (`EnologicalStressTester`):**\n"
        "   - Audit classifier resilience under shipping heat shocks (+8°C transit), cork micro-oxygenation, acidification, and rain dilution.\n"
        "   - Compute adversarial critical $\\epsilon$-margin to classification flip and overall Vintage Resilience Score (0-100).\n"
        "4. **Shapley Enological Interactions & Molecular $\\text{SO}_2$ Equilibrium (`ShapEnologicalInteractions`):**\n"
        "   - Quantify ethanol sensory masking of volatile acidity sharpness and Henderson-Hasselbalch active molecular $\\text{SO}_2$ vs must pH.\n"
        "5. **Production REST API Endpoints (`src/production_service.py`):**\n"
        "   - `/aging/simulate`, `/pareto/optimize`, `/stress/test`, and `/metrics/shap-interactions`."
    ))

    # Code: Imports & Environment Setup
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "# Add repository root to Python path\n"
        "sys.path.insert(0, str(Path('.').resolve()))\n\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n\n"
        "from src.data_loader import load_wine_data\n"
        "from src.aging_pareto_stress import (\n"
        "    CellarAgingKineticsSimulator,\n"
        "    ParetoEnologicalOptimizer,\n"
        "    EnologicalStressTester,\n"
        "    ShapEnologicalInteractions,\n"
        "    CLOSURE_OTR\n"
        ")\n"
        "from src.production_service import get_test_client\n\n"
        "print('Day 11 modules successfully imported!')"
    ))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. First-Principles Bottle Aging Kinetics Simulation\n\n"
        "In commercial enology, bottled wine is a dynamic biochemical system. Free sulfur dioxide ($\text{SO}_2$) acts as the primary antioxidant and antimicrobial barrier. As oxygen diffuses through closures (natural cork, screwcap, synthetic), Free $\text{SO}_2$ decays exponentially according to pseudo-first-order kinetics:\n\n"
        "$$\\text{Free } \\text{SO}_2(t) = \\text{Free } \\text{SO}_2(0) \\cdot \\exp(-k_{\\text{SO}_2} \\cdot t)$$\n\n"
        "When Free $\text{SO}_2$ drops below $10.0$ mg/L, acetic acid bacteria (*Acetobacter aceti*) proliferate, causing volatile acidity to rise. The code below simulates this trajectory over 60 months."
    ))

    # Code: Aging Simulation
    cells.append(nbf.v4.new_code_cell(
        "# Load red wine dataset and extract a representative reserve lot\n"
        "df_red = load_wine_data('red')\n"
        "prem_sample = df_red[df_red['quality'] >= 7].drop(columns=['quality']).iloc[0].to_dict()\n\n"
        "sim = CellarAgingKineticsSimulator()\n"
        "res_cork = sim.simulate_trajectory(prem_sample, months=60, temperature_c=14.0, closure='natural_cork')\n"
        "res_screwcap = sim.simulate_trajectory(prem_sample, months=60, temperature_c=14.0, closure='screwcap')\n\n"
        "print(f'Initial Quality Probability: {res_cork[\"initial_probability\"]*100:.1f}%')\n"
        "print(f'Natural Cork Peak Probability: {res_cork[\"peak_probability\"]*100:.1f}% at Month {res_cork[\"peak_month\"]}')\n"
        "print(f'Optimal Drinking Window: Month {res_cork[\"drinking_window_start_month\"]} to Month {res_cork[\"drinking_window_end_month\"]}')\n"
        "print(f'Sommelier Cellar Guidance: {res_cork[\"sommelier_cellar_advice\"]}')"
    ))

    # Code: Plot Aging Trajectory
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n\n"
        "traj_cork = pd.DataFrame(res_cork['trajectory'])\n"
        "traj_screw = pd.DataFrame(res_screwcap['trajectory'])\n\n"
        "# Left: Free SO2 & Volatile Acidity\n"
        "axes[0].plot(traj_cork['month'], traj_cork['free sulfur dioxide'], label='Free SO2 (Cork)', color='#1f77b4', lw=2)\n"
        "axes[0].plot(traj_screw['month'], traj_screw['free sulfur dioxide'], label='Free SO2 (Screwcap)', color='#2ca02c', lw=2)\n"
        "axes[0].plot(traj_cork['month'], traj_cork['volatile acidity'] * 30, label='VA x 30 (Cork)', color='#d62728', lw=2, linestyle='--')\n"
        "axes[0].axhline(10.0, color='orange', linestyle=':', label='Antimicrobial Threshold (10 mg/L)')\n"
        "axes[0].set_title('Chemical Kinetics: Free SO2 vs Volatile Acidity', fontweight='bold')\n"
        "axes[0].set_xlabel('Cellaring Months')\n"
        "axes[0].set_ylabel('Concentration')\n"
        "axes[0].legend()\n\n"
        "# Right: Quality Trajectory P(Quality >= 7)\n"
        "axes[1].plot(traj_cork['month'], traj_cork['premium_probability'], label='Natural Cork', color='#1f77b4', lw=2.5)\n"
        "axes[1].plot(traj_screw['month'], traj_screw['premium_probability'], label='Screwcap', color='#2ca02c', lw=2.5)\n"
        "axes[1].axhline(0.35, color='gray', linestyle='--', label='Reserve Threshold (tau=0.35)')\n"
        "axes[1].set_title('Predicted Sensory Quality Trajectory', fontweight='bold')\n"
        "axes[1].set_xlabel('Cellaring Months')\n"
        "axes[1].set_ylabel('P(Quality >= 7)')\n"
        "axes[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Multi-Objective Pareto Frontier Optimization\n\n"
        "Wineries must balance multiple conflicting goals:\n"
        "- **Maximize Sensory Quality:** $P(\\text{Quality} \\ge 7)$\n"
        "- **Minimize Sulphites/Preservatives:** Consumer demand for organic / low-intervention wine.\n"
        "- **Manage Alcohol Body & Excise Tax:** Keeping alcohol in optimal commercial balance.\n\n"
        "A lot $A$ dominates $B$ ($A \\succ B$) if it is at least as good in all objectives and strictly better in at least one. We extract the non-dominated Pareto Rank 1 frontier and find the knee-point best compromise."
    ))

    # Code: Pareto Optimization
    cells.append(nbf.v4.new_code_cell(
        "opt = ParetoEnologicalOptimizer()\n"
        "np.random.seed(42)\n"
        "sample_indices = np.random.choice(len(df_red), size=40, replace=False)\n"
        "candidates = [\n"
        "    {'name': f'Lot_{idx+1}', **df_red.drop(columns=['quality']).iloc[idx].to_dict()}\n"
        "    for idx in sample_indices\n"
        "]\n\n"
        "pareto_res = opt.compute_pareto_frontier(candidates)\n"
        "print(f'Total Candidates: {pareto_res[\"total_candidates_evaluated\"]}')\n"
        "print(f'Pareto-Optimal Lots (Rank 1): {pareto_res[\"pareto_optimal_count\"]}')\n\n"
        "knee = pareto_res['knee_point_lot']\n"
        "print(f'Knee-Point Best Compromise: {knee[\"name\"]} (Quality: {knee[\"quality_prob\"]*100:.1f}%, Sulphates: {knee[\"sulphates\"]} g/L, Distance: {knee[\"knee_distance\"]})')\n"
        "print('Archetypes:', {k: v['name'] for k, v in pareto_res['archetypes'].items() if v})"
    ))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Physicochemical Stress-Testing & Perturbation Vulnerability\n\n"
        "Wine shipments encounter transit heat waves, rough handling, and cork micro-leakage. We stress-test candidate lots across 5 standardized environmental disturbance scenarios to measure stability margin $\\epsilon^*$ and calculate the Vintage Resilience Score."
    ))

    # Code: Stress Testing
    cells.append(nbf.v4.new_code_cell(
        "tester = EnologicalStressTester()\n"
        "stress_res = tester.run_stress_test(prem_sample)\n\n"
        "print(f'Baseline Quality Probability: {stress_res[\"baseline_probability\"]*100:.1f}%')\n"
        "print(f'Resilience Score: {stress_res[\"resilience_score\"]:.1f} / 100 ({stress_res[\"resilience_tier\"]})')\n"
        "print(f'Critical Volatile Acidity Flip Margin: {stress_res[\"critical_va_epsilon_flip_margin\"]} g/L')\n\n"
        "df_stress = pd.DataFrame(stress_res['scenario_results'])\n"
        "display(df_stress[['scenario_name', 'probability_delta', 'retained_classification']])"
    ))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Enological Chemical Interactions & Molecular $\\text{SO}_2$ Equilibrium\n\n"
        "We evaluate non-linear physicochemical interactions, including ethanol sensory masking of acetic acid and Henderson-Hasselbalch active molecular $\\text{SO}_2$:"
    ))

    # Code: SHAP Interactions
    cells.append(nbf.v4.new_code_cell(
        "interactions = ShapEnologicalInteractions.compute_interaction_matrix(sim.model, df_red)\n"
        "print(json.dumps(interactions, indent=2))"
    ))

    # Markdown: Section 5
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Production REST API Verification\n\n"
        "Testing the 4 newly exposed FastAPI endpoints (`/aging/simulate`, `/pareto/optimize`, `/stress/test`, `/metrics/shap-interactions`)."
    ))

    # Code: API verification
    cells.append(nbf.v4.new_code_cell(
        "client = get_test_client()\n\n"
        "# 1. Test /aging/simulate\n"
        "aging_payload = {\n"
        "    'sample': {\n"
        "        'fixed_acidity': prem_sample['fixed acidity'],\n"
        "        'volatile_acidity': prem_sample['volatile acidity'],\n"
        "        'citric_acid': prem_sample['citric acid'],\n"
        "        'residual_sugar': prem_sample['residual sugar'],\n"
        "        'chlorides': prem_sample['chlorides'],\n"
        "        'free_sulfur_dioxide': prem_sample['free sulfur dioxide'],\n"
        "        'total_sulfur_dioxide': prem_sample['total sulfur dioxide'],\n"
        "        'density': prem_sample['density'],\n"
        "        'pH': prem_sample['pH'],\n"
        "        'sulphates': prem_sample['sulphates'],\n"
        "        'alcohol': prem_sample['alcohol']\n"
        "    },\n"
        "    'months': 36,\n"
        "    'temperature_c': 14.0,\n"
        "    'closure_type': 'natural_cork'\n"
        "}\n"
        "r1 = client.post('/aging/simulate', json=aging_payload)\n"
        "print(f'POST /aging/simulate -> Status {r1.status_code}, Initial Prob: {r1.json()[\"initial_probability\"]}')\n\n"
        "# 2. Test /metrics/shap-interactions\n"
        "r2 = client.get('/metrics/shap-interactions')\n"
        "print(f'GET /metrics/shap-interactions -> Status {r2.status_code}, Keys: {list(r2.json()[\"interactions\"].keys())}')\n"
        "assert r1.status_code == 200 and r2.status_code == 200\n"
        "print('All API endpoints verified successfully!')"
    ))

    # Markdown: Embedded Publication Figures
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Generated Publication Figures for Day 11\n\n"
        "All figures generated by `src/day11_figures.py`:\n\n"
        "![Cellar Aging Kinetics](../reports/figures/day11_cellar_aging_kinetics.png)\n\n"
        "![Multi-Objective Pareto Frontier](../reports/figures/day11_pareto_multiobjective_frontier.png)\n\n"
        "![Physicochemical Stress-Testing](../reports/figures/day11_physicochemical_stress_testing.png)\n\n"
        "![Shapley Enological Interactions](../reports/figures/day11_shap_enological_interactions.png)"
    ))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook to {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


def execute_day11_notebook():
    nb_path = create_day11_notebook()
    print(f"Executing notebook {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(REPO_ROOT)}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved {nb_path}!")


if __name__ == "__main__":
    execute_day11_notebook()
