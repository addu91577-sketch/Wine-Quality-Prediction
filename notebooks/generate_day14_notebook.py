"""
Day 14 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day14_Tartrate_Protein_Filtration_and_Packaging.ipynb
and executes all cells to populate verified outputs: potassium bitartrate thermodynamic
saturation and cold contact crystallization, Langmuir bentonite protein haze fining,
0.45 um crossflow membrane filtration Hermia kinetics, dissolved gas N2 sparging & TPO,
multi-closure 48-month shelf-life forecasting, and bottling line SPC metrology audits.
"""

import os
import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(".").resolve()
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Day14_Tartrate_Protein_Filtration_and_Packaging.ipynb"


def create_day14_notebook():
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
        "# Day 14: Tartrate & Protein Stabilization, Crossflow Microfiltration, Dissolved Gas Packaging & Closure Shelf-Life Forecaster\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Author:** ML Engineering & Production Deployment Team\n"
        "**Milestone:** Day 14 Enterprise Packaging & Finishing Platform Expansion\n\n"
        "### Key Architectural Capabilities:\n"
        "1. **Thermodynamic Tartrate Saturation & Cold Crystallization:** Models potassium bitartrate ($KHT$) Concentration Product ($CP = [K^+][HT^-]$), saturation temperature ($T_{sat}$), minicontact electrical conductivity drop ($\Delta \\chi$), and dynamic contact crystallization kinetics ($-4^\\circ\\text{C}$ with $4\\text{ g/L}$ seeding).\n"
        "2. **Thermolabile Protein Haze & Langmuir Bentonite Optimization:** Simulates heat test ($80^\\circ\\text{C} / 2\\text{h}$) nephelometric turbidity ($\Delta NTU$) and optimizes sodium bentonite dosage via Langmuir adsorption isotherms while quantifying volatile aroma stripping penalties.\n"
        "3. **Crossflow Membrane Microfiltration (0.45 $\\mu$m PES):** Solves Hermia's filtration blocking rate laws, transmembrane pressure (TMP), permeate flux decline $J(t)$, turbidity clarification ($NTU < 0.6$), and Filterability Index ($FI$).\n"
        "4. **Dissolved Gas Packaging Dynamics & Sparging:** Computes Henry's Law solubility equilibria for $DO$ and $CO_2$, micro-bubble counter-current nitrogen sparging deoxygenation, Total Packaging Oxygen ($TPO$), and antioxidant Free $SO_2$ stoichiometric depletion.\n"
        "5. **Multi-Closure Permeation & Predictive Shelf-Life Forecaster:** Solves 0-48 month coupled ODEs for Oxygen Transmission Rate (OTR), Free $SO_2$ decay, spectrophotometric browning ($A_{420}$), organoleptic freshness, and PremOx breach boundaries across Natural Cork, Diam, Nomacorc, and Screwcaps.\n"
        "6. **Bottling Line Statistical Process Control (SPC) Auditor:** Evaluates net fill volume ($750\\text{ mL} \\pm 1.5\\%$ MAV legal metrology), closure insertion depth, and filter membrane bubble point integrity ($> 3.2\\text{ bar}$)."
    ))

    # Code: Setup & Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('..').resolve()) if Path('..').resolve().name != 'Wine Quality Prediction' else str(Path('.').resolve()))\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from src.stabilization_and_packaging import (\n"
        "    TartrateStabilizationSimulator,\n"
        "    ProteinHazeAndBentoniteOptimizer,\n"
        "    CrossflowFiltrationSimulator,\n"
        "    DissolvedGasPackagingEngine,\n"
        "    ClosureShelfLifeForecaster,\n"
        "    BottlingLineQCAuditor,\n"
        "    CLOSURE_PROFILES,\n"
        "    STABILIZATION_ADDITIVES\n"
        ")\n"
        "print('Successfully loaded Day 14 Stabilization, Membrane Filtration & Packaging Modules!')"
    ))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Thermodynamic Tartrate Saturation & Cold Contact Kinetics\n\n"
        "Evaluates the thermodynamic equilibrium of potassium bitartrate ($KHT$) and simulates chilled contact seeding."
    ))

    # Code: Section 1
    cells.append(nbf.v4.new_code_cell(
        "tartrate_sim = TartrateStabilizationSimulator()\n\n"
        "# Evaluate specimen under cellar storage (14°C)\n"
        "eval_res = tartrate_sim.evaluate_tartrate_stability(\n"
        "    tartaric_acid_g_l=7.2,\n"
        "    potassium_mg_l=1150.0,\n"
        "    ph=3.55,\n"
        "    alcohol_pct=13.5,\n"
        "    storage_temp_c=14.0\n"
        ")\n"
        "print('Tartrate Saturation Assessment:')\n"
        "print(f'  Concentration Product (CP): {eval_res[\"concentration_product_cp\"]:.2e} (mol/L)^2')\n"
        "print(f'  Solubility Product (K_sp):  {eval_res[\"solubility_product_ksp\"]:.2e} (mol/L)^2')\n"
        "print(f'  Saturation Ratio:           {eval_res[\"saturation_ratio\"]:.2f}')\n"
        "print(f'  Saturation Temp (T_sat):    {eval_res[\"saturation_temperature_c\"]} °C')\n"
        "print(f'  Minicontact Drop (Delta chi): {eval_res[\"minicontact_delta_chi_us_cm\"]} uS/cm')\n"
        "print(f'  Stability Classification:   {eval_res[\"stability_status\"]}')\n\n"
        "# Dynamic contact crystallization simulation at -4°C with 4 g/L seed crystals\n"
        "sim_res = tartrate_sim.simulate_cold_stabilization_kinetics(\n"
        "    tartaric_acid_g_l=7.2,\n"
        "    potassium_mg_l=1150.0,\n"
        "    ph=3.55,\n"
        "    alcohol_pct=13.5,\n"
        "    chilling_temp_c=-4.0,\n"
        "    seed_crystal_g_l=4.0,\n"
        "    duration_hours=12\n"
        ")\n"
        "traj_df = pd.DataFrame(sim_res['trajectory'])\n"
        "print(f'Precipitated KHT: {sim_res[\"final_state\"][\"total_kht_precipitated_g_l\"]} g/L')\n"
        "print(f'Post-Stabilization Drop: {sim_res[\"final_state\"][\"minicontact_delta_chi_us_cm\"]} uS/cm -> {sim_res[\"final_state\"][\"stability_status\"]}')\n"
        "traj_df.head()"
    ))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Thermolabile Protein Haze & Langmuir Bentonite Optimization\n\n"
        "Quantifies heat-induced haze risk and optimizes sodium bentonite dosage to preserve volatile aromas."
    ))

    # Code: Section 2
    cells.append(nbf.v4.new_code_cell(
        "bentonite_opt = ProteinHazeAndBentoniteOptimizer()\n\n"
        "haze_eval = bentonite_opt.evaluate_protein_haze_risk(protein_content_mg_l=130.0, ph=3.25, cultivar='white')\n"
        "print('Heat Test Haze Evaluation (80°C / 2h):')\n"
        "print(f'  Estimated Delta NTU: {haze_eval[\"estimated_heat_delta_ntu\"]} NTU')\n"
        "print(f'  Risk Category:      {haze_eval[\"risk_category\"]}')\n\n"
        "dose_opt = bentonite_opt.optimize_bentonite_dosage(\n"
        "    protein_content_mg_l=130.0,\n"
        "    ph=3.25,\n"
        "    cultivar='white',\n"
        "    bentonite_type='sodium_bentonite'\n"
        ")\n"
        "print('Langmuir Fining Optimization Result:')\n"
        "print(f'  Recommended Dosage:   {dose_opt[\"recommended_bentonite_g_hl\"]} g/hL')\n"
        "print(f'  Post-Fining Haze:     {dose_opt[\"heat_delta_ntu_final\"]} Delta NTU (Target <= 1.8)')\n"
        "print(f'  Aroma Stripping Loss: {dose_opt[\"aroma_stripping_penalty_pct\"]} %')"
    ))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Crossflow Membrane Microfiltration (0.45 um PES)\n\n"
        "Simulates permeate flux decline $J(t)$, periodic hydrodynamic backwash restoration, and brilliant clarification."
    ))

    # Code: Section 3
    cells.append(nbf.v4.new_code_cell(
        "filt_sim = CrossflowFiltrationSimulator(membrane_surface_area_m2=40.0)\n\n"
        "run_res = filt_sim.simulate_crossflow_run(\n"
        "    initial_turbidity_ntu=38.0,\n"
        "    transmembrane_pressure_bar=1.6,\n"
        "    target_volume_liters=12000.0,\n"
        "    colloidal_index=1.2,\n"
        "    backwash_interval_hours=2.0,\n"
        "    run_duration_hours=6.0\n"
        ")\n"
        "print('Filtration Performance Summary:')\n"
        "for k, v in run_res['summary'].items():\n"
        "    print(f'  {k}: {v}')"
    ))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Dissolved Gas Management, N2 Sparging & Total Packaging Oxygen (TPO)\n\n"
        "Evaluates micro-bubble deoxygenation efficiency and calculates Total Packaging Oxygen (TPO)."
    ))

    # Code: Section 4
    cells.append(nbf.v4.new_code_cell(
        "gas_eng = DissolvedGasPackagingEngine()\n\n"
        "# Micro-bubble N2 sparging\n"
        "sparge_res = gas_eng.simulate_nitrogen_sparging(\n"
        "    initial_do_mg_l=1.40,\n"
        "    target_do_mg_l=0.40,\n"
        "    nitrogen_flow_ratio_nl_l=0.6,\n"
        "    temperature_c=18.0\n"
        ")\n"
        "print('Nitrogen Sparging Results:')\n"
        "print(f'  Initial DO: {sparge_res[\"initial_do_mg_l\"]} mg/L -> Final DO: {sparge_res[\"final_do_mg_l\"]} mg/L')\n"
        "print(f'  Efficiency: {sparge_res[\"deoxygenation_efficiency_pct\"]} % | Target Reached: {sparge_res[\"target_reached\"]}')\n\n"
        "# TPO calculation with vacuum corker inert gas flush (1.5% O2 in headspace)\n"
        "tpo_res = gas_eng.calculate_total_packaging_oxygen(\n"
        "    dissolved_oxygen_wine_mg_l=sparge_res[\"final_do_mg_l\"],\n"
        "    headspace_volume_ml=6.0,\n"
        "    bottle_volume_ml=750.0,\n"
        "    headspace_o2_pct=1.5\n"
        ")\n"
        "print('TPO Packaging Assessment:')\n"
        "print(f'  Total Packaging Oxygen: {tpo_res[\"total_packaging_oxygen_tpo_mg_l\"]} mg/L')\n"
        "print(f'  Grade:                  {tpo_res[\"tpo_grade\"]}')\n"
        "print(f'  Stoichiometric SO2 Hit: {tpo_res[\"stoichiometric_free_so2_depletion_mg_l\"]} mg/L')"
    ))

    # Markdown: Section 5
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Packaging Closure Permeation & 48-Month Shelf-Life Forecaster\n\n"
        "Forecasts Free $SO_2$ depletion, browning index ($A_{420}$), and PremOx boundaries across closures."
    ))

    # Code: Section 5
    cells.append(nbf.v4.new_code_cell(
        "shelf_forecaster = ClosureShelfLifeForecaster()\n\n"
        "c_res = shelf_forecaster.simulate_bottle_shelf_life(\n"
        "    closure_type='diam_5_microagglomerated',\n"
        "    initial_free_so2=32.0,\n"
        "    initial_ph=3.45,\n"
        "    initial_tpo_mg_l=tpo_res[\"total_packaging_oxygen_tpo_mg_l\"],\n"
        "    storage_temperature_c=14.0,\n"
        "    duration_months=48\n"
        ")\n"
        "print('Closure Aging Milestones (Diam 5 @ 14°C):')\n"
        "for k, v in c_res['milestones'].items():\n"
        "    print(f'  {k}: {v}')\n"
        "print('Final 48-Month State:')\n"
        "for k, v in c_res['final_state'].items():\n"
        "    print(f'  {k}: {v}')"
    ))

    # Markdown: Section 6
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Bottling Line Statistical Process Control (SPC) Audit\n\n"
        "Audits commercial batch fill volume metrology ($750\\text{ mL} \\pm 1.5\\%$) and membrane integrity."
    ))

    # Code: Section 6
    cells.append(nbf.v4.new_code_cell(
        "qc_auditor = BottlingLineQCAuditor()\n\n"
        "np.random.seed(42)\n"
        "sample_vols = [float(round(v, 2)) for v in np.random.normal(750.3, 1.6, 25)]\n"
        "sample_depths = [float(round(d, 2)) for d in np.random.normal(1.05, 0.12, 25)]\n"
        "sample_vacs = [float(round(vac, 3)) for vac in np.random.normal(-0.27, 0.03, 25)]\n\n"
        "spc_report = qc_auditor.audit_bottling_batch(\n"
        "    lot_id='LOT_ESTATE_CAB_B2026_01',\n"
        "    fill_volumes_ml=sample_vols,\n"
        "    closure_depths_mm=sample_depths,\n"
        "    headspace_vacuums_bar=sample_vacs,\n"
        "    filter_bubble_point_bar=3.45\n"
        ")\n"
        "print('Bottling Line SPC Audit Report:')\n"
        "print(f'  Audit Tier:       {spc_report[\"audit_tier\"]}')\n"
        "print(f'  Approved:         {spc_report[\"is_batch_approved\"]}')\n"
        "print(f'  Mean Volume:      {spc_report[\"fill_volume_spc\"][\"mean_ml\"]} mL (Nominal 750 mL)')\n"
        "print(f'  Process Cpk:      {spc_report[\"fill_volume_spc\"][\"cpk\"]}')\n"
        "print(f'  Filter Integrity: Bubble Point {spc_report[\"membrane_integrity\"][\"bubble_point_bar\"]} bar -> Passed={spc_report[\"membrane_integrity\"][\"passed\"]}')"
    ))

    # Markdown: Section 7
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Publication Diagnostic Figures Gallery\n\n"
        "Display the four publication-grade diagnostic figures published in `reports/figures/`."
    ))

    # Code: Section 7
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename='reports/figures/day14_tartrate_and_protein_stabilization.png'))\n"
        "display(Image(filename='reports/figures/day14_crossflow_membrane_filtration.png'))\n"
        "display(Image(filename='reports/figures/day14_dissolved_gas_and_sparging.png'))\n"
        "display(Image(filename='reports/figures/day14_closure_permeation_and_shelf_life.png'))"
    ))

    nb.cells = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created notebook structure at: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


def execute_day14_notebook():
    notebook_file = create_day14_notebook()
    with open(notebook_file, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print(f"Executing notebook: {notebook_file} ...")
    ep.preprocess(nb, {"metadata": {"path": str(REPO_ROOT)}})

    with open(notebook_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully executed and saved: {notebook_file}")


if __name__ == "__main__":
    execute_day14_notebook()
