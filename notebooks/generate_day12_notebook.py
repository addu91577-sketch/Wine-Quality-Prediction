"""
Day 12 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day12_Fermentation_Kinetics_Stuck_Detection_and_Sensory_Lexicon.ipynb
and executes all cells to populate verified outputs: enological bioprocess kinetics,
stuck fermentation early-warning triage, 6-axis organoleptic radar profiling,
sommelier tasting notes, food pairing affinities, and doubly robust causal treatment effects.
"""

import os
import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(".").resolve()
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Day12_Fermentation_Kinetics_Stuck_Detection_and_Sensory_Lexicon.ipynb"


def create_day12_notebook():
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
        "# Day 12: Fermentation Bioprocess Kinetics, Stuck Fermentation Early-Warning & Sensory Organoleptic Lexicon Engine\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Author:** ML Engineering & Production Deployment Team\n"
        "**Date:** November 4, 2026\n"
        "**Tech Stack:** Python 3.14, Scikit-learn, Scipy, FastAPI, Pydantic v2, Streamlit, Matplotlib, Seaborn, Joblib\n\n"
        "## Executive Overview & Objectives for Day 12\n"
        "Day 12 completes the bioprocess lifecycle and consumer organoleptic intelligence of the Wine Quality platform:\n\n"
        "1. **Primary Alcoholic Fermentation Kinetics Simulator (`FermentationKineticsSimulator`):**\n"
        "   - Coupled dynamic differential rate equations of *Saccharomyces cerevisiae* biomass proliferation, Monod hexose consumption, ethanol self-inhibition, nitrogen (YAN) depletion, metabolic heat liberation, and stress volatile acidity formation.\n"
        "   - Integrates finished wine chemistry with the champion ML pipeline to project final bottled reserve quality $P(\\text{Quality} \\ge 7)$.\n"
        "2. **Stuck & Sluggish Fermentation Early-Warning Detector (`StuckFermentationDetector`):**\n"
        "   - Real-time time-series kinetic monitor analyzing consumption velocity ($-\\Delta S / \\Delta t$) and deceleration.\n"
        "   - Triage classification (`HEALTHY_ACTIVE`, `SLUGGISH`, `STUCK_FERMENTATION`, `THERMAL_OVERHEATING_RISK`) and prescriptive enological cellar interventions (DAP addition, cooling jacket adjustment, aeration, re-inoculation).\n"
        "3. **Sensory Organoleptic Lexicon & Spider Radar Engine (`SensoryOrganolepticEngine`):**\n"
        "   - Translates laboratory physicochemical parameters into 6 standardized organoleptic dimensions (Body & Viscosity, Acidity & Crispness, Tannic Grip, Fruit Freshness, Sweetness-Acid Harmony, Ethanol Warmth).\n"
        "   - Automated sommelier tasting note synthesis (Eye, Nose, Palate, Finish) and gastronomic food pairing affinity scoring.\n"
        "4. **Causal Enological Counterfactual Inference (`CausalEnologicalEngine`):**\n"
        "   - Observational causal inference using Doubly Robust Inverse Propensity Weighting (IPW) to quantify the Average Treatment Effect (ATE) of cellar interventions while eliminating confounding bias.\n"
        "5. **Production REST API Endpoints (`src/production_service.py`):**\n"
        "   - `/fermentation/simulate`, `/fermentation/diagnose-stuck`, `/sensory/profile`, and `/metrics/causal-effects`."
    ))

    # Code: Environment Setup & Imports
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
        "from src.fermentation_and_sensory import (\n"
        "    FermentationKineticsSimulator,\n"
        "    StuckFermentationDetector,\n"
        "    SensoryOrganolepticEngine,\n"
        "    CausalEnologicalEngine\n"
        ")\n"
        "from src.production_service import get_test_client\n\n"
        "print('Day 12 modules successfully imported!')"
    ))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Alcoholic Fermentation Bioprocess Kinetics Simulation\n\n"
        "Wine quality originates during alcoholic fermentation. We model coupled differential equations:\n"
        "$$\\frac{dX}{dt} = (\\mu - k_d) X$$\n"
        "$$\\frac{dS}{dt} = -\\left( \\frac{\\mu}{Y_{X/S}} + q_{\\text{ferm}} + m_s \\right) X$$\n"
        "$$\\frac{dE}{dt} = \\frac{1}{16.83} \\left(-\\frac{dS}{dt}\\right)$$\n"
        "where specific growth rate $\\mu$ follows Monod saturation with ethanol inhibition and temperature scaling."
    ))

    # Code: Run Fermentation Simulation
    cells.append(nbf.v4.new_code_cell(
        "sim = FermentationKineticsSimulator()\n"
        "res_std = sim.simulate_fermentation(initial_sugar_g_l=225.0, initial_yan_mg_l=220.0, jacket_temp_c=22.0, total_days=14)\n"
        "res_warm = sim.simulate_fermentation(initial_sugar_g_l=225.0, initial_yan_mg_l=220.0, jacket_temp_c=30.0, total_days=14)\n\n"
        "print('Standard Fermentation (22°C):')\n"
        "print(f'  Status: {res_std[\"fermentation_status\"]}')\n"
        "print(f'  Final Sugar: {res_std[\"final_residual_sugar_g_l\"]} g/L | Final Ethanol: {res_std[\"final_ethanol_pct\"]}% vol')\n"
        "print(f'  Volatile Acidity: {res_std[\"final_volatile_acidity_g_l\"]} g/L | Peak Rate: {res_std[\"peak_fermentation_rate_g_l_day\"]} g/L/day')\n"
        "print(f'  Predicted Reserve Probability: {res_std[\"predicted_quality_prob\"]*100:.1f}%')\n\n"
        "print('Warm Fermentation (30°C):')\n"
        "print(f'  Status: {res_warm[\"fermentation_status\"]}')\n"
        "print(f'  Volatile Acidity: {res_warm[\"final_volatile_acidity_g_l\"]} g/L (Elevated thermal stress)')"
    ))

    # Code: Plot Fermentation Curves
    cells.append(nbf.v4.new_code_cell(
        "df_std = pd.DataFrame(res_std['trajectory'])\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n\n"
        "# Left: Sugar & Ethanol\n"
        "axes[0].plot(df_std['day'], df_std['sugar_g_l'], color='#1f77b4', lw=2.5, label='Hexose Sugar (g/L)')\n"
        "axes[0].set_xlabel('Fermentation Days')\n"
        "axes[0].set_ylabel('Sugar (g/L)', color='#1f77b4')\n"
        "axes[0].set_title('Sugar Depletion & Ethanol Synthesis', fontweight='bold')\n"
        "ax0_twin = axes[0].twinx()\n"
        "ax0_twin.plot(df_std['day'], df_std['ethanol_pct'], color='#2ca02c', lw=2.5, label='Ethanol (% vol)')\n"
        "ax0_twin.set_ylabel('Ethanol (% vol)', color='#2ca02c')\n"
        "axes[0].grid(True, alpha=0.3)\n\n"
        "# Right: Biomass & YAN\n"
        "axes[1].plot(df_std['day'], df_std['biomass_g_l'], color='#9467bd', lw=2.5, label='Yeast Biomass (g/L)')\n"
        "axes[1].set_xlabel('Fermentation Days')\n"
        "axes[1].set_ylabel('Biomass X (g/L)', color='#9467bd')\n"
        "axes[1].set_title('Yeast Biomass Proliferation & Nitrogen Depletion', fontweight='bold')\n"
        "ax1_twin = axes[1].twinx()\n"
        "ax1_twin.plot(df_std['day'], df_std['yan_mg_l'], color='#d62728', lw=2.0, linestyle='--', label='YAN (mg/L)')\n"
        "ax1_twin.set_ylabel('YAN (mg/L)', color='#d62728')\n"
        "axes[1].grid(True, alpha=0.3)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Stuck & Sluggish Fermentation Early-Warning Diagnostic\n\n"
        "Premature fermentation arrest creates high risk of microbial spoilage. `StuckFermentationDetector` calculates sugar decay velocity and issues prescriptive remediation protocols before spoilage occurs."
    ))

    # Code: Stuck Fermentation Detector
    cells.append(nbf.v4.new_code_cell(
        "detector = StuckFermentationDetector()\n\n"
        "# 1. Active normal tank\n"
        "readings_active = [\n"
        "    {'day': 0, 'sugar': 225.0, 'temp': 20.0, 'yan': 220.0},\n"
        "    {'day': 3, 'sugar': 160.0, 'temp': 22.0, 'yan': 150.0},\n"
        "    {'day': 6, 'sugar': 85.0, 'temp': 22.0, 'yan': 80.0}\n"
        "]\n"
        "diag_act = detector.diagnose_trajectory(readings_active)\n"
        "print(f'Active Tank Diagnosis: {diag_act[\"diagnosis\"]} (Risk: {diag_act[\"risk_score_pct\"]}%)')\n"
        "print(f'Message: {diag_act[\"message\"]}\\n')\n\n"
        "# 2. Arrested stuck tank\n"
        "readings_stuck = [\n"
        "    {'day': 0, 'sugar': 225.0, 'temp': 20.0},\n"
        "    {'day': 4, 'sugar': 72.0, 'temp': 22.0},\n"
        "    {'day': 7, 'sugar': 71.2, 'temp': 20.0}  # stalled\n"
        "]\n"
        "diag_stk = detector.diagnose_trajectory(readings_stuck)\n"
        "print(f'Arrested Tank Diagnosis: {diag_stk[\"diagnosis\"]} (Risk: {diag_stk[\"risk_score_pct\"]}%)')\n"
        "print(f'Message: {diag_stk[\"message\"]}')\n"
        "print('Prescriptive Remediations:')\n"
        "for act in diag_stk['prescriptive_actions']:\n"
        "    print(f'  * [{act[\"action\"]}] {act[\"protocol\"]}')"
    ))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Sensory Organoleptic 6-Axis Radar & Sommelier Flavor Lexicon\n\n"
        "`SensoryOrganolepticEngine` maps laboratory chemistry into standardized human sensory dimensions: Body, Acidity, Tannic Grip, Fruit Freshness, Sweetness Harmony, and Ethanol Warmth, generating structured sommelier notes and food pairing affinities."
    ))

    # Code: Sensory Profile
    cells.append(nbf.v4.new_code_cell(
        "df_red = load_wine_data('red')\n"
        "reserve_lot = df_red[df_red['quality'] >= 7].drop(columns=['quality']).iloc[0].to_dict()\n"
        "table_lot = df_red[df_red['quality'] == 5].drop(columns=['quality']).iloc[0].to_dict()\n\n"
        "sens_res = SensoryOrganolepticEngine.compute_sensory_profile(reserve_lot)\n"
        "print('Reserve Specimen 6-Axis Sensory Radar:')\n"
        "for k, v in sens_res['radar_scores'].items():\n"
        "    print(f'  * {k}: {v} / 10.0')\n\n"
        "print(f'\\nOverall Intensity: {sens_res[\"overall_sensory_intensity\"]} / 10.0')\n"
        "print(f'Estimated Dry Extract: {sens_res[\"dry_extract_est_g_l\"]} g/L\\n')\n"
        "print('Sommelier Tasting Notes:')\n"
        "for section, text in sens_res['tasting_note'].items():\n"
        "    print(f'  [{section.capitalize()}]: {text}')\n\n"
        "print('\\nGastronomic Culinary Pairings:')\n"
        "for p in sens_res['food_pairings'][:3]:\n"
        "    print(f'  * {p[\"dish\"]} ({p[\"category\"]}): {p[\"affinity\"]}% affinity - {p[\"pairing_rationale\"]}')"
    ))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Causal Enological Counterfactual Inference (Doubly Robust IPW)\n\n"
        "Observational associations between chemistry and wine quality can be confounded by vineyard vintage and harvest ripeness. `CausalEnologicalEngine` implements Inverse Propensity Weighting (IPW) to estimate the true unconfounded Average Treatment Effect (ATE)."
    ))

    # Code: Causal Inference
    cells.append(nbf.v4.new_code_cell(
        "causal_bench = {}\n"
        "for key in CausalEnologicalEngine.INTERVENTIONS:\n"
        "    causal_bench[key] = CausalEnologicalEngine.estimate_causal_effect(df_red, intervention_key=key, n_bootstraps=50)\n"
        "    res = causal_bench[key]\n"
        "    print(f'Intervention: {res[\"intervention_name\"]}')\n"
        "    print(f'  Causal ATE: +{res[\"causal_ate_pct\"]:.1f}% | 95% CI: [{res[\"ci_95_lower\"]*100:.1f}%, {res[\"ci_95_upper\"]*100:.1f}%]')\n"
        "    print(f'  Naive Delta: {res[\"naive_observational_delta\"]*100:.1f}% | Confounding Bias: {res[\"confounding_bias\"]*100:+.1f}%\\n')"
    ))

    # Markdown: Section 5
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Production FastAPI REST Microservice Verification\n\n"
        "Smoke test the 4 new enterprise REST endpoints integrated into `src/production_service.py`."
    ))

    # Code: REST API Endpoints
    cells.append(nbf.v4.new_code_cell(
        "client = get_test_client()\n\n"
        "# 1. POST /fermentation/simulate\n"
        "r1 = client.post('/fermentation/simulate', json={'initial_sugar_g_l': 225.0, 'jacket_temp_c': 20.0, 'total_days': 10})\n"
        "assert r1.status_code == 200\n"
        "print('POST /fermentation/simulate:', r1.json()['fermentation_status'], f'Sugar: {r1.json()[\"final_residual_sugar_g_l\"]} g/L')\n\n"
        "# 2. POST /fermentation/diagnose-stuck\n"
        "r2 = client.post('/fermentation/diagnose-stuck', json={'readings': [{'day': 0, 'sugar': 220, 'temp': 20}, {'day': 3, 'sugar': 180, 'temp': 22}, {'day': 6, 'sugar': 178, 'temp': 22}]})\n"
        "assert r2.status_code == 200\n"
        "print('POST /fermentation/diagnose-stuck:', r2.json()['diagnosis'], f'Risk: {r2.json()[\"risk_score_pct\"]}%')\n\n"
        "# 3. POST /sensory/profile\n"
        "r3 = client.post('/sensory/profile', json={'sample': reserve_lot})\n"
        "assert r3.status_code == 200\n"
        "print('POST /sensory/profile: Overall Intensity:', r3.json()['overall_sensory_intensity'])\n\n"
        "# 4. GET /metrics/causal-effects\n"
        "r4 = client.get('/metrics/causal-effects')\n"
        "assert r4.status_code == 200\n"
        "print('GET /metrics/causal-effects: Verified interventions count =', len(r4.json()['interventions']))"
    ))

    # Markdown: Section 6
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Publication Figures Gallery\n\n"
        "Display the four generated diagnostic figures published in `reports/figures/`."
    ))

    # Code: Display Figures
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename='reports/figures/day12_fermentation_bioprocess_kinetics.png'))\n"
        "display(Image(filename='reports/figures/day12_stuck_fermentation_early_warning.png'))\n"
        "display(Image(filename='reports/figures/day12_sensory_organoleptic_spider_radar.png'))\n"
        "display(Image(filename='reports/figures/day12_causal_treatment_effects.png'))"
    ))

    nb.cells = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created notebook structure at: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


def execute_day12_notebook():
    notebook_file = create_day12_notebook()
    with open(notebook_file, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print(f"Executing notebook: {notebook_file} ...")
    ep.preprocess(nb, {"metadata": {"path": str(REPO_ROOT)}})

    with open(notebook_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully executed and saved: {notebook_file}")


if __name__ == "__main__":
    execute_day12_notebook()
