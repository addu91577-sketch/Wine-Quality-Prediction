"""
Day 7 Jupyter Notebook Generator & Preprocessor.
Generates notebooks/Day7_Production_API_and_Deployment.ipynb
and executes all cells to populate verified outputs, latency profiles,
test suite results, and embedded publication figures.
"""

import os
import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(".").resolve()
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Day7_Production_API_and_Deployment.ipynb"


def create_day7_notebook():
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
        "# Day 7: Production Inference Engine, Real-Time REST API, Interactive Web Dashboard, Docker Containerization & CI/CD Automated Testing Framework\n"
        "**Project:** Wine Quality Prediction (Cortez et al. UCI Machine Learning Repository)\n"
        "**Author:** ML Engineering & Production Deployment Team\n"
        "**Date:** October 21, 2026\n"
        "**Tech Stack:** Python 3.14, FastAPI, Uvicorn, Pydantic v2, Streamlit, Pytest, Docker, Docker-Compose, GitHub Actions, Scikit-learn, Joblib, Plotly, Seaborn, Matplotlib\n\n"
        "## Objectives for Day 7\n"
        "1. **Production Inference Service & Domain Feature Synthesis (`src/production_service.py`):**\n"
        "   - Encapsulate the serialized champion Random Forest ensemble into a hardened serving layer.\n"
        "   - Automatically derive 5 biochemical interaction features (`total_acidity`, `bound_sulfur_dioxide`, `free_to_total_so2_ratio`, `alcohol_to_density_ratio`, `sulphate_to_chloride_ratio`).\n"
        "   - Synthesize enological cellar directives and 4 commercial sensory tiers (*Grand Reserve*, *Reserve Grade*, *Table Standard*, *Commercial Bulk*).\n"
        "2. **Pydantic v2 Data Contract & Boundary Enforcement:**\n"
        "   - Implement rigorous schemas with strict biological/physical bounds and cross-attribute consistency checks (e.g. Total SO₂ $\\ge$ Free SO₂).\n"
        "3. **High-Throughput Vectorized Batch Inference:**\n"
        "   - Provide batch inference endpoints capable of evaluating thousands of winery specimens concurrently.\n"
        "4. **Empirical Latency & Scaling Benchmarks:**\n"
        "   - Profile single-sample and batch latency percentiles (P50, P90, P95, P99) and throughput scaling curves.\n"
        "5. **Interactive Web Dashboard (`src/interactive_dashboard.py`):**\n"
        "   - Build a Streamlit application featuring live parameter tuning, radar profiling against reserve benchmarks, batch CSV uploads, and downloadable lab reports.\n"
        "6. **Automated Comprehensive Pytest Suite (`tests/`):**\n"
        "   - Implement and execute 22 unit and integration tests covering data integrity, schema validation, feature parity, API contracts, threshold overrides, and latency budgets.\n"
        "7. **Containerization & CI/CD Pipeline:**\n"
        "   - Deliver multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow (`.github/workflows/mlops_pipeline.yml`)."
    ))

    # Code: Setup and Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "import json\n"
        "import time\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display, Markdown\n\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# Ensure repository root is on sys.path\n"
        "REPO_ROOT = Path('.').resolve()\n"
        "if str(REPO_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_ROOT))\n\n"
        "from src.production_service import (\n"
        "    WineQualityInferenceService,\n"
        "    WinePhysicochemicalInput,\n"
        "    derive_interaction_features,\n"
        "    get_test_client,\n"
        "    DEFAULT_MODEL_PATH,\n"
        "    DEFAULT_METADATA_PATH,\n"
        "    RECOMMENDED_SCREENING_THRESHOLD\n"
        ")\n"
        "from src.data_loader import load_wine_data\n\n"
        "print('Day 7 Environment and Serving Modules Initialized Successfully!')"
    ))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Champion Model Artifact & Metadata Provenance Inspection\n"
        "Before exposing endpoints to production traffic, we verify the integrity of serialized artifacts on disk (`models/champion_rf_pipeline.joblib` and `models/champion_rf_pipeline.json`)."
    ))

    # Code: Model Verification
    cells.append(nbf.v4.new_code_cell(
        "service = WineQualityInferenceService()\n"
        "print(f'Model Ready: {service.is_ready}')\n"
        "print(f'Artifact Path: {service.model_path}')\n\n"
        "with open(DEFAULT_METADATA_PATH, 'r') as f:\n"
        "    meta = json.load(f)\n\n"
        "meta_df = pd.DataFrame([\n"
        "    {'Parameter': 'Model Name', 'Value': meta.get('model_name')},\n"
        "    {'Parameter': 'Date Finalized', 'Value': meta.get('date_finalized')},\n"
        "    {'Parameter': 'Number of Trees', 'Value': meta.get('n_estimators')},\n"
        "    {'Parameter': 'Max Depth', 'Value': meta.get('max_depth')},\n"
        "    {'Parameter': 'Class Weighting', 'Value': meta.get('class_weight')},\n"
        "    {'Parameter': 'Features Expected', 'Value': meta.get('num_features')},\n"
        "    {'Parameter': 'Holdout Test ROC-AUC', 'Value': f\"{meta.get('test_roc_auc'):.4f}\"},\n"
        "    {'Parameter': 'Holdout Test PR-AUC', 'Value': f\"{meta.get('test_pr_auc'):.4f}\"},\n"
        "    {'Parameter': 'Screening Threshold (tau)', 'Value': meta.get('operating_screening_threshold')}\n"
        "])\n"
        "display(meta_df)"
    ))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Pydantic v2 Data Contract Validation & Exception Handling\n"
        "Data validation is the first line of defense in production ML systems. We verify that:\n"
        "- Valid physicochemical inputs parse correctly.\n"
        "- Biologically impossible or out-of-bounds parameters raise explicit validation errors.\n"
        "- Contradictory inputs (e.g. Free SO₂ > Total SO₂) are intercepted before feature synthesis."
    ))

    # Code: Pydantic Validation Demo
    cells.append(nbf.v4.new_code_cell(
        "from pydantic import ValidationError\n\n"
        "# 1. Valid Input Sample\n"
        "valid_payload = {\n"
        "    'sample_id': 'LOT-2026-VAL-001',\n"
        "    'fixed acidity': 8.8,\n"
        "    'volatile acidity': 0.34,\n"
        "    'citric acid': 0.42,\n"
        "    'residual sugar': 2.1,\n"
        "    'chlorides': 0.071,\n"
        "    'free sulfur dioxide': 19.0,\n"
        "    'total sulfur dioxide': 38.0,\n"
        "    'density': 0.9954,\n"
        "    'pH': 3.31,\n"
        "    'sulphates': 0.84,\n"
        "    'alcohol': 12.3\n"
        "}\n"
        "parsed_specimen = WinePhysicochemicalInput(**valid_payload)\n"
        "print('Valid Sample Parsed Successfully:', parsed_specimen.sample_id)\n\n"
        "# 2. Intercepting Invalid Range (e.g. Negative pH)\n"
        "try:\n"
        "    invalid_payload = dict(valid_payload, pH=1.2)\n"
        "    WinePhysicochemicalInput(**invalid_payload)\n"
        "except ValidationError as e:\n"
        "    print('Successfully intercepted invalid pH:', e.errors()[0]['msg'])\n\n"
        "# 3. Intercepting SO2 Cross-Attribute Contradiction\n"
        "try:\n"
        "    contradictory_payload = dict(valid_payload, **{'free sulfur dioxide': 50.0, 'total sulfur dioxide': 20.0})\n"
        "    WinePhysicochemicalInput(**contradictory_payload)\n"
        "except ValidationError as e:\n"
        "    print('Successfully intercepted SO2 contradiction:', e.errors()[0]['msg'])"
    ))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Real-Time Single-Sample Prediction & Cellar Guidance\n"
        "We test single-sample inference comparing an everyday table wine against a candidate reserve wine."
    ))

    # Code: Single Sample Inference
    cells.append(nbf.v4.new_code_cell(
        "table_wine = WinePhysicochemicalInput(\n"
        "    sample_id='SAMPLE-TABLE-01',\n"
        "    fixed_acidity=7.2, volatile_acidity=0.62, citric_acid=0.08, residual_sugar=1.8,\n"
        "    chlorides=0.084, free_sulfur_dioxide=10.0, total_sulfur_dioxide=55.0, density=0.9972,\n"
        "    pH=3.45, sulphates=0.52, alcohol=9.4\n"
        ")\n\n"
        "reserve_wine = WinePhysicochemicalInput(\n"
        "    sample_id='SAMPLE-RESERVE-01',\n"
        "    fixed_acidity=9.5, volatile_acidity=0.28, citric_acid=0.52, residual_sugar=2.2,\n"
        "    chlorides=0.062, free_sulfur_dioxide=22.0, total_sulfur_dioxide=36.0, density=0.9948,\n"
        "    pH=3.28, sulphates=0.86, alcohol=12.8\n"
        ")\n\n"
        "res_table = service.predict_single(table_wine)\n"
        "res_reserve = service.predict_single(reserve_wine)\n\n"
        "comparison_df = pd.DataFrame([\n"
        "    {\n"
        "        'Sample ID': res_table.sample_id,\n"
        "        'Verdict': res_table.prediction_label,\n"
        "        'Sensory Tier': res_table.sommelier_grade,\n"
        "        'Probability (>=7)': f'{res_table.probability_good * 100:.1f}%',\n"
        "        'Total Acidity': res_table.engineered_features['total_acidity'],\n"
        "        'Alc/Density': res_table.engineered_features['alcohol_to_density_ratio'],\n"
        "        'Cellar Recommendation': res_table.decision_guidance\n"
        "    },\n"
        "    {\n"
        "        'Sample ID': res_reserve.sample_id,\n"
        "        'Verdict': res_reserve.prediction_label,\n"
        "        'Sensory Tier': res_reserve.sommelier_grade,\n"
        "        'Probability (>=7)': f'{res_reserve.probability_good * 100:.1f}%',\n"
        "        'Total Acidity': res_reserve.engineered_features['total_acidity'],\n"
        "        'Alc/Density': res_reserve.engineered_features['alcohol_to_density_ratio'],\n"
        "        'Cellar Recommendation': res_reserve.decision_guidance\n"
        "    }\n"
        "])\n"
        "display(comparison_df)"
    ))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. High-Throughput Vectorized Laboratory Batch Scoring\n"
        "We simulate a commercial laboratory run by scoring 50 wine vats sampled from the dataset."
    ))

    # Code: Batch Scoring Demo
    cells.append(nbf.v4.new_code_cell(
        "df_raw = load_wine_data(wine_type='red').sample(50, random_state=42).reset_index(drop=True)\n"
        "batch_inputs = [\n"
        "    WinePhysicochemicalInput(\n"
        "        sample_id=f'VAT-2026-{idx+1:03d}',\n"
        "        fixed_acidity=row['fixed acidity'],\n"
        "        volatile_acidity=row['volatile acidity'],\n"
        "        citric_acid=row['citric acid'],\n"
        "        residual_sugar=row['residual sugar'],\n"
        "        chlorides=row['chlorides'],\n"
        "        free_sulfur_dioxide=row['free sulfur dioxide'],\n"
        "        total_sulfur_dioxide=row['total sulfur dioxide'],\n"
        "        density=row['density'],\n"
        "        pH=row['pH'],\n"
        "        sulphates=row['sulphates'],\n"
        "        alcohol=row['alcohol']\n"
        "    )\n"
        "    for idx, row in df_raw.iterrows()\n"
        "]\n\n"
        "batch_resp = service.predict_batch(batch_inputs, threshold=RECOMMENDED_SCREENING_THRESHOLD)\n\n"
        "batch_kpi = pd.DataFrame([\n"
        "    {'Metric': 'Total Specimens Scored', 'Value': batch_resp.total_samples},\n"
        "    {'Metric': 'Reserve Grade Count (Score >= 7)', 'Value': batch_resp.premium_samples},\n"
        "    {'Metric': 'Standard Table Wine Count', 'Value': batch_resp.standard_samples},\n"
        "    {'Metric': 'Reserve Yield Rate', 'Value': f'{batch_resp.premium_rate_pct:.1f}%'},\n"
        "    {'Metric': 'Mean Good Wine Probability', 'Value': f'{batch_resp.mean_probability_good * 100:.1f}%'},\n"
        "    {'Metric': 'Batch Processing Latency', 'Value': f'{batch_resp.batch_latency_ms:.2f} ms'},\n"
        "    {'Metric': 'Throughput', 'Value': f'{batch_resp.total_samples / (batch_resp.batch_latency_ms / 1000.0):.0f} samples/sec'}\n"
        "])\n"
        "display(batch_kpi)"
    ))

    # Markdown: Section 5
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. REST API Latency Profile & Empirical Percentiles\n"
        "We evaluate response latencies under ASGI serving to ensure compliance with production Service Level Objectives (SLOs)."
    ))

    # Code: Latency Benchmark & Display Figure
    cells.append(nbf.v4.new_code_cell(
        "client = get_test_client()\n\n"
        "# Measure 25 API requests\n"
        "api_latencies = []\n"
        "for _ in range(5):\n"
        "    client.post('/predict', json=valid_payload)\n\n"
        "for _ in range(25):\n"
        "    t0 = time.perf_counter()\n"
        "    resp = client.post('/predict', json=valid_payload)\n"
        "    api_latencies.append((time.perf_counter() - t0) * 1000.0)\n\n"
        "lat_summary = pd.DataFrame([\n"
        "    {'Percentile': 'P50 (Median)', 'Latency (ms)': f'{np.percentile(api_latencies, 50):.2f} ms'},\n"
        "    {'Percentile': 'P90', 'Latency (ms)': f'{np.percentile(api_latencies, 90):.2f} ms'},\n"
        "    {'Percentile': 'P95 (Budget)', 'Latency (ms)': f'{np.percentile(api_latencies, 95):.2f} ms'},\n"
        "    {'Percentile': 'P99', 'Latency (ms)': f'{np.percentile(api_latencies, 99):.2f} ms'}\n"
        "])\n"
        "display(lat_summary)\n\n"
        "display(Image(filename='reports/figures/day7_api_latency_distribution.png'))"
    ))

    # Markdown: Section 6
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Batch Inference Throughput Scaling Analysis\n"
        "We profile how batch inference scales across increasing specimen lot sizes from 1 to 500 samples."
    ))

    # Code: Batch Scaling Display
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='reports/figures/day7_batch_inference_scaling.png'))"
    ))

    # Markdown: Section 7
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Interactive Streamlit Web Platform & Enological Radar Profiling\n"
        "The web interface (`src/interactive_dashboard.py`) enables winemakers and sommeliers to visualize multidimensional chemical profiles against empirical quality targets."
    ))

    # Code: Radar Chart Display
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='reports/figures/day7_sommelier_profile_radar.png'))"
    ))

    # Markdown: Section 8
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Automated Test Suite Execution (Pytest Verification)\n"
        "We execute the 22-test automated suite across `tests/test_pipeline.py` and `tests/test_production_service.py` to ensure complete system verification."
    ))

    # Code: Pytest Run
    cells.append(nbf.v4.new_code_cell(
        "import pytest\n"
        "result_code = pytest.main(['-q', '--tb=short', 'tests'])\n"
        "print(f'Pytest Exit Code: {result_code} (0 = All Passed)')\n"
        "assert result_code == 0, 'Test suite failed!'"
    ))

    # Markdown: Section 9
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Production Architecture, Docker Containerization & CI/CD Pipeline\n"
        "Below is the complete MLOps serving architecture and deployment blueprint."
    ))

    # Code: Architecture Display
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='reports/figures/day7_production_architecture.png'))"
    ))

    # Markdown: Final Synthesis
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Day 7 Technical Deliverables & Production Deployment Sign-Off\n\n"
        "### Key Production Deliverables Completed on Day 7:\n"
        "1. **`src/production_service.py`:** Production FastAPI serving engine with Pydantic v2 validation contracts, chemical ratio derivation, sommelier grading, and vectorized batch inference.\n"
        "2. **`src/interactive_dashboard.py`:** Interactive Streamlit web application with single-specimen slider controls, enological alerts, batch CSV scoring, and radar alignment.\n"
        "3. **`tests/test_production_service.py` & `tests/test_pipeline.py`:** 22 passing automated unit and integration tests covering data integrity, schema validation, feature engineering parity, and API contracts.\n"
        "4. **`Dockerfile` & `docker-compose.yml`:** Multi-stage production container setup with non-root security compliance, healthchecks, and multi-service orchestration.\n"
        "5. **`.github/workflows/mlops_pipeline.yml`:** Automated CI/CD workflow running multi-version Python testing, model artifact verification, and API smoke tests on every push.\n"
        "6. **`reports/figures/day7_*`:** 4 publication figures documenting latency distributions, batch throughput scaling, sommelier radar profiling, and production architecture."
    ))

    # Assign cells to notebook
    nb["cells"] = cells

    # Write notebook file
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated notebook template at {NOTEBOOK_PATH}")

    # Execute notebook
    print("Executing notebook cells to pre-populate outputs...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_to_run = nbf.read(f, as_version=4)

    ep.preprocess(nb_to_run, {"metadata": {"path": str(REPO_ROOT)}})

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb_to_run, f)
    print(f"Successfully executed and saved: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_day7_notebook()
