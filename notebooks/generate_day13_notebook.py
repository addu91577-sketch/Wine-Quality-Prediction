"""
Day 13 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day13_Malolactic_Oak_Maturation_and_Cellar_Dispatch.ipynb
and executes all cells to populate verified outputs: secondary malolactic fermentation
bioprocess kinetics, precision oak barrel wood extraction dynamics, digital cellar master
risk dispatch triage, and HACCP enological food safety audits.
"""

import os
import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(".").resolve()
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Day13_Malolactic_Oak_Maturation_and_Cellar_Dispatch.ipynb"


def create_day13_notebook():
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
        "# Day 13: Secondary Malolactic Fermentation, Precision Oak Barrel Maturation, Digital Cellar Dispatch & HACCP Auditor\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Author:** ML Engineering & Production Deployment Team\n"
        "**Milestone:** Day 13 Enterprise Enology Platform Expansion\n\n"
        "### Key Architectural Capabilities:\n"
        "1. **Secondary Malolactic Fermentation (MLF) Biotransformation:** Decarboxylation of dicarboxylic L-malic acid into monocarboxylic L-lactic acid and CO2 by *Oenococcus oeni*, tracking pH elevation, fixed acidity drop, and diacetyl buttery synthesis.\n"
        "2. **Precision Oak Barrel Maturation Dynamics:** First-order Fickian dissolution kinetics of wood ellagitannins, oak lactones, vanillin, and guaiacol smoke across French, American, and Hungarian oak, with micro-oxygenation polymeric pigment color stabilization.\n"
        "3. **Digital Cellar Master Work-Order Dispatch Engine:** Multi-vessel risk triage calculating the Enological Risk Priority Index (ERPI) and generating prescriptive chemical additions (Potassium Metabisulfite PMS doses, lysozyme, topping).\n"
        "4. **HACCP Food Safety & Biogenic Amine Risk Auditor:** Statutory verification of Critical Control Points (CCPs), molecular SO2 antimicrobial envelopes, and biogenic amine (histamine/tyramine) hazard scoring."
    ))

    # Code: Setup & Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('.').resolve()))\n\n"
        "import json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "from src.malolactic_and_oak import (\n"
        "    MalolacticFermentationSimulator,\n"
        "    OakBarrelMaturationSimulator,\n"
        "    CellarMasterDispatchEngine,\n"
        "    EnologicalHaccpAuditor\n"
        ")\n"
        "from src.production_service import get_test_client\n"
        "print('Day 13 modules imported successfully!')"
    ))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Secondary Malolactic Fermentation (MLF) Simulation\n\n"
        "We simulate *Oenococcus oeni* bioconversion of L-malic acid under optimal conditions (20°C, 5 mg/L Free SO2) "
        "and compare against cold cellar (13°C) and high-sulfur (16 mg/L Free SO2) inhibition conditions."
    ))

    # Code: Section 1
    cells.append(nbf.v4.new_code_cell(
        "sim = MalolacticFermentationSimulator()\n"
        "res_opt = sim.simulate(initial_malic_acid=2.8, initial_ph=3.35, alcohol=13.0, free_so2=5.0, temperature=20.0, duration_days=35)\n"
        "res_cold = sim.simulate(initial_malic_acid=2.8, initial_ph=3.35, alcohol=13.0, free_so2=5.0, temperature=13.0, duration_days=35)\n"
        "res_so2 = sim.simulate(initial_malic_acid=2.8, initial_ph=3.35, alcohol=13.0, free_so2=16.0, temperature=20.0, duration_days=35)\n\n"
        "print('=== Optimal MLF (20°C, 5 mg/L Free SO2) ===')\n"
        "print('Final Malic Acid (g/L):', res_opt['final_state']['malic_acid_g_l'])\n"
        "print('Final Lactic Acid (g/L):', res_opt['final_state']['lactic_acid_g_l'])\n"
        "print('Final pH:', res_opt['final_state']['ph'])\n"
        "print('Fixed Acidity (g/L):', res_opt['final_state']['fixed_acidity_g_l'])\n"
        "print('Peak Diacetyl (mg/L):', res_opt['final_state']['peak_diacetyl_mg_l'])\n"
        "print('MLF Complete:', res_opt['diagnostics']['is_complete'])\n"
        "print('Sensory Description:', res_opt['diagnostics']['sensory_descriptor'])\n\n"
        "print('\\n=== Environmental Stress Comparison ===')\n"
        "print('Cold Cellar Conversion (%):', res_cold['final_state']['conversion_rate_pct'])\n"
        "print('High SO2 Conversion (%):', res_so2['final_state']['conversion_rate_pct'])"
    ))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Precision Oak Barrel Maturation & Wood Phenolics Extraction\n\n"
        "Dynamic wood extractives simulation across French (*Quercus petraea*), American (*Quercus alba*), "
        "and Hungarian (*Quercus frainetto*) oak species, tracking ellagitannins, oak lactones, vanillin, "
        "and polymeric pigment color stabilization."
    ))

    # Code: Section 2
    cells.append(nbf.v4.new_code_cell(
        "oak_sim = OakBarrelMaturationSimulator()\n"
        "res_french = oak_sim.simulate(oak_species='french', toast_level='medium_plus', barrel_usage='new', maturation_months=24)\n"
        "res_american = oak_sim.simulate(oak_species='american', toast_level='medium', barrel_usage='new', maturation_months=24)\n"
        "res_neutral = oak_sim.simulate(oak_species='french', toast_level='medium', barrel_usage='neutral', maturation_months=24)\n\n"
        "df_oak = pd.DataFrame([\n"
        "    {\n"
        "        'Profile': 'French New (Med+)',\n"
        "        'Ellagitannins (mg/L)': res_french['final_concentrations']['ellagitannins_mg_l'],\n"
        "        'Lactones (mg/L)': res_french['final_concentrations']['whiskey_lactones_mg_l'],\n"
        "        'Vanillin (mg/L)': res_french['final_concentrations']['vanillin_mg_l'],\n"
        "        'Guaiacol Smoke (mg/L)': res_french['final_concentrations']['guaiacol_mg_l'],\n"
        "        'Polymer Pigment': res_french['final_concentrations']['polymeric_pigment_index'],\n"
        "        'Oak Intensity (0-100)': res_french['final_concentrations']['oak_intensity_score'],\n"
        "    },\n"
        "    {\n"
        "        'Profile': 'American New (Med)',\n"
        "        'Ellagitannins (mg/L)': res_american['final_concentrations']['ellagitannins_mg_l'],\n"
        "        'Lactones (mg/L)': res_american['final_concentrations']['whiskey_lactones_mg_l'],\n"
        "        'Vanillin (mg/L)': res_american['final_concentrations']['vanillin_mg_l'],\n"
        "        'Guaiacol Smoke (mg/L)': res_american['final_concentrations']['guaiacol_mg_l'],\n"
        "        'Polymer Pigment': res_american['final_concentrations']['polymeric_pigment_index'],\n"
        "        'Oak Intensity (0-100)': res_american['final_concentrations']['oak_intensity_score'],\n"
        "    },\n"
        "    {\n"
        "        'Profile': 'French Neutral (4th Fill)',\n"
        "        'Ellagitannins (mg/L)': res_neutral['final_concentrations']['ellagitannins_mg_l'],\n"
        "        'Lactones (mg/L)': res_neutral['final_concentrations']['whiskey_lactones_mg_l'],\n"
        "        'Vanillin (mg/L)': res_neutral['final_concentrations']['vanillin_mg_l'],\n"
        "        'Guaiacol Smoke (mg/L)': res_neutral['final_concentrations']['guaiacol_mg_l'],\n"
        "        'Polymer Pigment': res_neutral['final_concentrations']['polymeric_pigment_index'],\n"
        "        'Oak Intensity (0-100)': res_neutral['final_concentrations']['oak_intensity_score'],\n"
        "    }\n"
        "])\n"
        "display(df_oak)"
    ))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Digital Cellar Master Dispatch Engine\n\n"
        "Simulates multi-vessel fleet triage across winery cellar vessels, computing the Enological Risk "
        "Priority Index (ERPI) and generating prescriptive chemical additions (Potassium Metabisulfite PMS)."
    ))

    # Code: Section 3
    cells.append(nbf.v4.new_code_cell(
        "dispatch = CellarMasterDispatchEngine()\n"
        "fleet = [\n"
        "    {'vessel_id': 'BARREL-FR-01', 'vessel_type': 'BARREL', 'cultivar': 'red', 'volume_liters': 225.0, 'ph': 3.42, 'free_so2': 14.0, 'total_so2': 65.0, 'volatile_acidity': 0.42, 'alcohol': 13.5, 'temperature': 17.5, 'residual_sugar': 1.8, 'malic_acid': 0.04, 'maturation_days': 120, 'status': 'aging'},\n"
        "    {'vessel_id': 'TANK-FERM-03', 'vessel_type': 'TANK', 'cultivar': 'red', 'volume_liters': 5000.0, 'ph': 3.65, 'free_so2': 8.0, 'total_so2': 45.0, 'volatile_acidity': 0.78, 'alcohol': 12.8, 'temperature': 23.5, 'residual_sugar': 6.5, 'malic_acid': 0.85, 'maturation_days': 21, 'status': 'aging'},\n"
        "    {'vessel_id': 'BARREL-US-08', 'vessel_type': 'BARREL', 'cultivar': 'red', 'volume_liters': 225.0, 'ph': 3.50, 'free_so2': 11.0, 'total_so2': 55.0, 'volatile_acidity': 0.52, 'alcohol': 13.8, 'temperature': 18.0, 'residual_sugar': 1.9, 'malic_acid': 0.05, 'maturation_days': 210, 'status': 'aging'},\n"
        "    {'vessel_id': 'TANK-WHITE-02', 'vessel_type': 'TANK', 'cultivar': 'white', 'volume_liters': 10000.0, 'ph': 3.20, 'free_so2': 28.0, 'total_so2': 110.0, 'volatile_acidity': 0.32, 'alcohol': 12.2, 'temperature': 13.0, 'residual_sugar': 4.5, 'malic_acid': 1.80, 'maturation_days': 60, 'status': 'resting'},\n"
        "    {'vessel_id': 'BARREL-VINTAGE-99', 'vessel_type': 'BARREL', 'cultivar': 'red', 'volume_liters': 225.0, 'ph': 3.82, 'free_so2': 6.0, 'total_so2': 90.0, 'volatile_acidity': 1.05, 'alcohol': 14.2, 'temperature': 21.0, 'residual_sugar': 2.0, 'malic_acid': 0.45, 'maturation_days': 340, 'status': 'aging'}\n"
        "]\n"
        "dispatch_summary = dispatch.dispatch_cellar_inventory(fleet)\n"
        "print('=== Cellar Master Fleet Triage Summary ===')\n"
        "print('Total Vessels Evaluated:', dispatch_summary['total_vessels'])\n"
        "print('Tier Distribution:', dispatch_summary['tier_distribution'])\n"
        "print('Cellar Health Score (0-100):', dispatch_summary['cellar_health_score'])\n"
        "print('Aggregate PMS Reagents Required (kg):', dispatch_summary['aggregate_pms_reagents_required_kg'])\n\n"
        "# Print Top Urgent Vessel Work Order\n"
        "top_vessel = dispatch_summary['dispatched_vessels'][0]\n"
        "print(f'\\nTop Priority Vessel: {top_vessel[\"vessel_id\"]} (ERPI: {top_vessel[\"metrics\"][\"erpi_score\"]})')\n"
        "print('Action Tier:', top_vessel['triage']['action_tier'])\n"
        "for idx, wo in enumerate(top_vessel['work_orders'], 1):\n"
        "    print(f'  Work Order {idx}: {wo[\"action\"]} | Reagent: {wo[\"chemical\"]} | Dose: {wo[\"dosage_grams\"]} g')\n"
        "    print(f'    Rationale: {wo[\"rationale\"]}')"
    ))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. HACCP Food Safety & Biogenic Amine Risk Auditor\n\n"
        "Performs statutory Critical Control Point (CCP) audits and calculates biogenic amine "
        "(histamine, tyramine) decarboxylation hazard scores for production lots."
    ))

    # Code: Section 4
    cells.append(nbf.v4.new_code_cell(
        "auditor = EnologicalHaccpAuditor()\n"
        "audit_pass = auditor.audit_wine_lot(\n"
        "    lot_id='LOT-RESERVE-2026',\n"
        "    cultivar='red',\n"
        "    ph=3.42,\n"
        "    free_so2=24.0,\n"
        "    total_so2=85.0,\n"
        "    volatile_acidity=0.45,\n"
        "    alcohol=13.5,\n"
        "    malic_acid=0.03,\n"
        "    temperature=16.0\n"
        ")\n\n"
        "audit_hazard = auditor.audit_wine_lot(\n"
        "    lot_id='LOT-STRESSED-02',\n"
        "    cultivar='red',\n"
        "    ph=3.85,\n"
        "    free_so2=5.0,\n"
        "    total_so2=40.0,\n"
        "    volatile_acidity=0.65,\n"
        "    alcohol=12.2,\n"
        "    malic_acid=0.75,\n"
        "    temperature=22.0\n"
        ")\n\n"
        "print('=== Safe Lot Audit ===')\n"
        "print('Status:', audit_pass['audit_status'])\n"
        "print('All CCPs Passed:', audit_pass['all_ccp_passed'])\n"
        "print('Biogenic Amine Risk:', audit_pass['biogenic_amine_risk']['risk_level'])\n"
        "print('Predicted Histamine:', audit_pass['biogenic_amine_risk']['predicted_histamine_content'])\n\n"
        "print('=== Stressed Lot Audit ===')\n"
        "print('Status:', audit_hazard['audit_status'])\n"
        "print('All CCPs Passed:', audit_hazard['all_ccp_passed'])\n"
        "print('Biogenic Amine Risk:', audit_hazard['biogenic_amine_risk']['risk_level'], f'({audit_hazard[\"biogenic_amine_risk\"][\"risk_index_score\"]})')\n"
        "print('Predicted Histamine:', audit_hazard['biogenic_amine_risk']['predicted_histamine_content'])\n"
        "print('Findings:', audit_hazard['audit_findings'])"
    ))

    # Markdown: Section 5
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Production REST API Endpoints Verification\n\n"
        "We test all four Day 13 REST microservice endpoints using the in-memory FastAPI `TestClient`."
    ))

    # Code: Section 5
    cells.append(nbf.v4.new_code_cell(
        "client = get_test_client()\n\n"
        "# 1. POST /cellar/malolactic/simulate\n"
        "r1 = client.post('/cellar/malolactic/simulate', json={\n"
        "    'initial_malic_acid': 2.8,\n"
        "    'initial_ph': 3.35,\n"
        "    'alcohol': 13.0,\n"
        "    'free_so2': 6.0,\n"
        "    'temperature': 20.0,\n"
        "    'duration_days': 35\n"
        "})\n"
        "assert r1.status_code == 200\n"
        "print('POST /cellar/malolactic/simulate: 200 OK | Final Conversion =', r1.json()['final_state']['conversion_rate_pct'], '%')\n\n"
        "# 2. POST /cellar/oak-maturation/simulate\n"
        "r2 = client.post('/cellar/oak-maturation/simulate', json={\n"
        "    'oak_species': 'french',\n"
        "    'toast_level': 'medium_plus',\n"
        "    'barrel_usage': 'new',\n"
        "    'maturation_months': 18\n"
        "})\n"
        "assert r2.status_code == 200\n"
        "print('POST /cellar/oak-maturation/simulate: 200 OK | Ellagitannins =', r2.json()['final_concentrations']['ellagitannins_mg_l'], 'mg/L')\n\n"
        "# 3. POST /cellar/dispatch/work-orders\n"
        "r3 = client.post('/cellar/dispatch/work-orders', json={'inventory': fleet})\n"
        "assert r3.status_code == 200\n"
        "print('POST /cellar/dispatch/work-orders: 200 OK | Total Vessels =', r3.json()['total_vessels'])\n\n"
        "# 4. POST /compliance/haccp-audit\n"
        "r4 = client.post('/compliance/haccp-audit', json={\n"
        "    'lot_id': 'LOT-TEST-REST',\n"
        "    'cultivar': 'red',\n"
        "    'ph': 3.4,\n"
        "    'free_so2': 22.0,\n"
        "    'total_so2': 85.0,\n"
        "    'volatile_acidity': 0.45,\n"
        "    'alcohol': 13.5\n"
        "})\n"
        "assert r4.status_code == 200\n"
        "print('POST /compliance/haccp-audit: 200 OK | Status =', r4.json()['audit_status'])"
    ))

    # Markdown: Section 6
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Publication Figures Gallery\n\n"
        "Display the four generated diagnostic figures published in `reports/figures/`."
    ))

    # Code: Section 6
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename='reports/figures/day13_malolactic_fermentation_kinetics.png'))\n"
        "display(Image(filename='reports/figures/day13_oak_maturation_wood_extractives.png'))\n"
        "display(Image(filename='reports/figures/day13_cellar_master_dispatch_matrix.png'))\n"
        "display(Image(filename='reports/figures/day13_haccp_compliance_and_biogenic_risk.png'))"
    ))

    nb.cells = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created notebook structure at: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


def execute_day13_notebook():
    notebook_file = create_day13_notebook()
    with open(notebook_file, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print(f"Executing notebook: {notebook_file} ...")
    ep.preprocess(nb, {"metadata": {"path": str(REPO_ROOT)}})

    with open(notebook_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully executed and saved: {notebook_file}")


if __name__ == "__main__":
    execute_day13_notebook()
