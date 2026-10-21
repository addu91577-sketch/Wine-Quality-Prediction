"""
Interactive Streamlit Web Application: Wine Quality Intelligence Platform
Provides winery laboratory personnel, winemakers, and sommeliers with real-time
sensory grading, enological decision support, chemical radar profiling, and bulk batch scoring.
"""

import sys
from pathlib import Path
import json
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.production_service import (
    WineQualityInferenceService,
    WinePhysicochemicalInput,
    derive_interaction_features,
    RECOMMENDED_SCREENING_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_METADATA_PATH,
    FULL_FEATURE_ORDER
)


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Wine Quality Intelligence Platform",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache Inference Service Initialization
@st.cache_resource
def get_inference_service():
    return WineQualityInferenceService(
        model_path=DEFAULT_MODEL_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
        default_threshold=RECOMMENDED_SCREENING_THRESHOLD
    )


# Empirical Benchmarks for Radar Chart Comparison
BENCHMARKS = {
    "alcohol": {"Normal": 10.2, "Good": 11.5, "Unit": "% vol"},
    "volatile acidity": {"Normal": 0.54, "Good": 0.37, "Unit": "g/dm³"},
    "sulphates": {"Normal": 0.62, "Good": 0.74, "Unit": "g/dm³"},
    "citric acid": {"Normal": 0.24, "Good": 0.40, "Unit": "g/dm³"},
    "chlorides": {"Normal": 0.082, "Good": 0.073, "Unit": "g/dm³"},
    "total_acidity": {"Normal": 8.8, "Good": 9.2, "Unit": "g/dm³"}
}

# Pre-configured Wine Specimen Presets
PRESETS = {
    "Custom Manual Sliders": None,
    "🏆 Top-Tier Reserve Candidate": {
        "fixed acidity": 9.5,
        "volatile acidity": 0.30,
        "citric acid": 0.50,
        "residual sugar": 2.2,
        "chlorides": 0.065,
        "free sulfur dioxide": 20.0,
        "total sulfur dioxide": 40.0,
        "density": 0.9950,
        "pH": 3.30,
        "sulphates": 0.82,
        "alcohol": 12.5
    },
    "🍷 Everyday Table Wine": {
        "fixed acidity": 7.8,
        "volatile acidity": 0.58,
        "citric acid": 0.18,
        "residual sugar": 2.0,
        "chlorides": 0.085,
        "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 65.0,
        "density": 0.9972,
        "pH": 3.38,
        "sulphates": 0.56,
        "alcohol": 9.8
    },
    "⚠️ Acetic Spoilage Fault (High VA)": {
        "fixed acidity": 7.2,
        "volatile acidity": 0.85,
        "citric acid": 0.05,
        "residual sugar": 1.8,
        "chlorides": 0.095,
        "free sulfur dioxide": 8.0,
        "total sulfur dioxide": 45.0,
        "density": 0.9968,
        "pH": 3.52,
        "sulphates": 0.48,
        "alcohol": 9.4
    },
    "🛡️ High Preservative / SO2 Lot": {
        "fixed acidity": 8.2,
        "volatile acidity": 0.42,
        "citric acid": 0.32,
        "residual sugar": 3.5,
        "chlorides": 0.072,
        "free sulfur dioxide": 45.0,
        "total sulfur dioxide": 160.0,
        "density": 0.9978,
        "pH": 3.25,
        "sulphates": 0.72,
        "alcohol": 10.8
    }
}


def build_radar_chart(current_sample: dict):
    """Generate an enological radar comparison against empirical Normal vs Good medians."""
    categories = ["Alcohol", "Citric Acid", "Sulphates", "Low Volatile Acid (Inverted)", "Low Chlorides (Inverted)", "Total Acidity"]
    
    # Normalize features to 0-1 scale for balanced radar comparison
    # For volatile acidity and chlorides, lower is better, so invert
    val_alc = min(1.0, max(0.0, (current_sample["alcohol"] - 8.0) / 6.0))
    val_cit = min(1.0, max(0.0, current_sample["citric acid"] / 0.8))
    val_sul = min(1.0, max(0.0, (current_sample["sulphates"] - 0.3) / 1.0))
    val_va = min(1.0, max(0.0, 1.0 - (current_sample["volatile acidity"] / 1.0)))
    val_cl = min(1.0, max(0.0, 1.0 - (current_sample["chlorides"] / 0.2)))
    tot_acid = current_sample.get("total_acidity", current_sample["fixed acidity"] + current_sample["volatile acidity"])
    val_ta = min(1.0, max(0.0, (tot_acid - 6.0) / 8.0))

    sample_vals = [val_alc, val_cit, val_sul, val_va, val_cl, val_ta]
    good_vals = [0.65, 0.55, 0.60, 0.65, 0.65, 0.45]
    normal_vals = [0.35, 0.30, 0.40, 0.40, 0.50, 0.35]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=sample_vals + [sample_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(139, 0, 0, 0.3)',
        line=dict(color='#8B0000', width=3),
        name='Current Specimen'
    ))

    fig.add_trace(go.Scatterpolar(
        r=good_vals + [good_vals[0]],
        theta=categories + [categories[0]],
        line=dict(color='#2E7D32', width=2, dash='dash'),
        name='Reserve Benchmark (>=7)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=normal_vals + [normal_vals[0]],
        theta=categories + [categories[0]],
        line=dict(color='#757575', width=1.5, dash='dot'),
        name='Standard Benchmark (<7)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        ),
        showlegend=True,
        title=dict(text="Physicochemical Radar Alignment", font=dict(size=15)),
        margin=dict(l=40, r=40, t=40, b=30),
        height=380
    )
    return fig


def main():
    service = get_inference_service()

    # Sidebar Header and System Status
    st.sidebar.title("🍷 Decision Control")
    st.sidebar.markdown("**Wine Quality Prediction Engine**  \n*Powered by Cortez et al. UCI Red Wine Dataset*")
    st.sidebar.divider()

    # Model Status Card
    st.sidebar.subheader("⚙️ Active Champion Model")
    st.sidebar.info(
        f"**Algorithm:** Tuned Random Forest  \n"
        f"**ROC-AUC:** {service.metadata.get('test_roc_auc', 0.9368):.4f}  \n"
        f"**PR-AUC:** {service.metadata.get('test_pr_auc', 0.8095):.4f}  \n"
        f"**Recall (Good Wines):** 79.1%  \n"
        f"**Status:** {'🟢 Ready' if service.is_ready else '🔴 Offline'}"
    )

    # Threshold Adjustment
    st.sidebar.subheader("🎯 Decision Threshold (τ)")
    operating_threshold = st.sidebar.slider(
        "Operating Probability Cutoff",
        min_value=0.10,
        max_value=0.80,
        value=float(RECOMMENDED_SCREENING_THRESHOLD),
        step=0.05,
        help="Standard threshold is 0.50. Optimized screening threshold is 0.35, which yields +102% minority recall for cellar batch screening."
    )
    if operating_threshold == 0.35:
        st.sidebar.caption("✅ Recommended Industrial Screening Standard (τ=0.35)")
    elif operating_threshold == 0.50:
        st.sidebar.caption("⚖️ Strict Balanced Probability Standard (τ=0.50)")

    st.sidebar.divider()

    # Preset Selector
    selected_preset = st.sidebar.selectbox("📋 Load Specimen Preset", list(PRESETS.keys()))
    active_preset = PRESETS[selected_preset]

    # Main Application Header
    st.title("🍷 Wine Quality Intelligence Platform")
    st.markdown(
        "Enological quality classification, real-time chemical diagnostic screening, "
        "and automated sommelier grading for premium wine production."
    )

    # Tabs for Application Modules
    tab_single, tab_batch, tab_diagnostics = st.tabs([
        "🔬 Single Specimen Profiler",
        "📦 Commercial Winery Batch Scorer",
        "📊 Architecture & Production Governance"
    ])

    # ------------------------------------------------------------------------
    # TAB 1: Single Wine Specimen Profiler
    # ------------------------------------------------------------------------
    with tab_single:
        st.subheader("Laboratory Physicochemical Input")
        st.caption("Adjust laboratory parameters below or select a specimen preset from the left sidebar.")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("##### 🍋 Organic Acids & pH")
            fixed_acidity = st.slider(
                "Fixed Acidity (g/dm³ tartaric)",
                3.0, 16.0,
                float(active_preset["fixed acidity"]) if active_preset else 8.5,
                0.1,
                help="Predominant tartaric acid. Normal median: 7.9 g/dm³; Good median: 8.8 g/dm³."
            )
            volatile_acidity = st.slider(
                "Volatile Acidity (g/dm³ acetic)",
                0.10, 1.60,
                float(active_preset["volatile acidity"]) if active_preset else 0.35,
                0.01,
                help="Acetic acid causing vinegar notes. Normal median: 0.54 g/dm³; Good median: 0.37 g/dm³."
            )
            citric_acid = st.slider(
                "Citric Acid (g/dm³)",
                0.0, 1.0,
                float(active_preset["citric acid"]) if active_preset else 0.42,
                0.01,
                help="Imparts freshness and crispness. Normal median: 0.24 g/dm³; Good median: 0.40 g/dm³."
            )
            ph = st.slider(
                "pH (Acidity level)",
                2.8, 4.0,
                float(active_preset["pH"]) if active_preset else 3.32,
                0.01,
                help="Scale of acidity. Typical range: 3.1 - 3.6."
            )

        with col_b:
            st.markdown("##### 🍇 Fermentation Dynamics")
            alcohol = st.slider(
                "Alcohol (% vol)",
                8.0, 15.0,
                float(active_preset["alcohol"]) if active_preset else 12.2,
                0.1,
                help="Ethanol concentration. Primary driver of wine quality. Normal median: 10.2%; Good median: 11.5%."
            )
            residual_sugar = st.slider(
                "Residual Sugar (g/dm³)",
                0.5, 15.0,
                float(active_preset["residual sugar"]) if active_preset else 2.1,
                0.1,
                help="Sugar remaining after fermentation stops. Normal median: 2.2 g/dm³."
            )
            density = st.slider(
                "Density (g/cm³)",
                0.9880, 1.0040,
                float(active_preset["density"]) if active_preset else 0.9958,
                0.0002,
                format="%.4f",
                help="Water/alcohol balance. Lower density correlates with higher alcohol."
            )

        with col_c:
            st.markdown("##### 🛡️ Preservatives & Mineral Salinity")
            free_so2 = st.slider(
                "Free Sulfur Dioxide (mg/dm³)",
                2.0, 70.0,
                float(active_preset["free sulfur dioxide"]) if active_preset else 18.0,
                1.0,
                help="Active antimicrobial gas. Protects against oxidation and spoilage."
            )
            total_so2 = st.slider(
                "Total Sulfur Dioxide (mg/dm³)",
                max(float(free_so2), 6.0), 250.0,
                max(float(free_so2) + 10.0, float(active_preset["total sulfur dioxide"]) if active_preset else 38.0),
                1.0,
                help="Combined free and bound SO2. Mask fruit aromas if excessively high."
            )
            sulphates = st.slider(
                "Sulphates (g/dm³ potassium sulphate)",
                0.20, 2.0,
                float(active_preset["sulphates"]) if active_preset else 0.80,
                0.02,
                help="Wine additive contributing to SO2 levels. Good median: 0.74 g/dm³."
            )
            chlorides = st.slider(
                "Chlorides (g/dm³ sodium chloride)",
                0.010, 0.400,
                float(active_preset["chlorides"]) if active_preset else 0.068,
                0.002,
                format="%.3f",
                help="Salts contributing to mineral brackishness. Lower is generally preferred."
            )

        # Assemble Input Model
        input_data = WinePhysicochemicalInput(
            sample_id="UI-LAB-INSPECTION",
            fixed_acidity=fixed_acidity,
            volatile_acidity=volatile_acidity,
            citric_acid=citric_acid,
            residual_sugar=residual_sugar,
            chlorides=chlorides,
            free_sulfur_dioxide=free_so2,
            total_sulfur_dioxide=total_so2,
            density=density,
            pH=ph,
            sulphates=sulphates,
            alcohol=alcohol
        )

        st.divider()

        # Execute Prediction
        t_start = time.perf_counter()
        result = service.predict_single(input_data, threshold=operating_threshold)
        infer_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Display Prediction Cards
        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            st.markdown("#### Sensory Verdict & Sommelier Grading")
            if result.is_premium_wine:
                st.success(
                    f"### 🏆 {result.sommelier_grade.upper()}\n"
                    f"**Classification:** {result.prediction_label}  \n"
                    f"**Posterior Probability:** `{result.probability_good * 100.0:.1f}%`  \n"
                    f"**Decision Threshold:** `τ = {operating_threshold:.2f}`  \n"
                    f"**Cellar Recommendation:** {result.decision_guidance}"
                )
            else:
                st.warning(
                    f"### 🍷 {result.sommelier_grade.upper()}\n"
                    f"**Classification:** {result.prediction_label}  \n"
                    f"**Posterior Probability:** `{result.probability_good * 100.0:.1f}%`  \n"
                    f"**Decision Threshold:** `τ = {operating_threshold:.2f}`  \n"
                    f"**Cellar Recommendation:** {result.decision_guidance}"
                )

            # Probability Gauge Progress Bar
            st.markdown(f"**Quality Probability Meter (Threshold: {operating_threshold:.2f})**")
            st.progress(float(result.probability_good))
            st.caption(
                f"Confidence delta: `{result.decision_confidence:.3f}` | "
                f"Inference latency: `{infer_latency_ms:.2f} ms`"
            )

            # Enological Alerts
            st.markdown("##### 🧪 Enological Diagnostic Observations")
            for alert in result.enological_alerts:
                if "[WARNING]" in alert:
                    st.warning(alert)
                elif "[OPTIMAL]" in alert:
                    st.success(alert)
                else:
                    st.info(alert)

        with res_col2:
            st.markdown("#### Physicochemical Alignment vs Benchmarks")
            # Build Radar Chart
            radar_dict = input_data.to_raw_dict()
            radar_dict["total_acidity"] = result.engineered_features["total_acidity"]
            fig_radar = build_radar_chart(radar_dict)
            st.plotly_chart(fig_radar, use_container_width=True)

            # Engineered Domain Ratios
            st.markdown("##### 🔬 Synthesized Chemical Interaction Ratios")
            eng_c1, eng_c2, eng_c3, eng_c4 = st.columns(4)
            eng_c1.metric("Total Acidity", f"{result.engineered_features['total_acidity']:.2f}", "g/dm³")
            eng_c2.metric("Alc/Density", f"{result.engineered_features['alcohol_to_density_ratio']:.2f}")
            eng_c3.metric("Free/Total SO₂", f"{result.engineered_features['free_to_total_so2_ratio']:.2f}")
            eng_c4.metric("SO₄/Cl Ratio", f"{result.engineered_features['sulphate_to_chloride_ratio']:.1f}")

    # ------------------------------------------------------------------------
    # TAB 2: Commercial Winery Batch Scorer
    # ------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Commercial Winery Batch Evaluation")
        st.markdown(
            "Upload a laboratory CSV export to evaluate hundreds or thousands of wine vats or "
            "bottles simultaneously. The system derives chemical interaction features and computes "
            "sensory tiers with sub-millisecond per-sample latency."
        )

        batch_col1, batch_col2 = st.columns([1, 2])

        with batch_col1:
            uploaded_file = st.file_uploader("Upload Laboratory CSV", type=["csv"])
            use_sample_btn = st.button("📥 Load Sample Winery Batch (50 Vats)")

        df_to_score = None
        if uploaded_file is not None:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded: {len(df_to_score)} samples.")
        elif use_sample_btn:
            # Load sample test data from Data/winequality-red.csv
            data_file = PROJECT_ROOT / "Data" / "winequality-red.csv"
            if data_file.exists():
                raw_df = pd.read_csv(data_file)
                # Sample 50 rows
                df_to_score = raw_df.sample(50, random_state=42).reset_index(drop=True)
                df_to_score["sample_id"] = [f"VAT-2026-{i+101:03d}" for i in range(len(df_to_score))]
                st.info("Loaded 50 representative winery vats from repository dataset.")

        if df_to_score is not None:
            st.divider()
            # Convert DataFrame to Pydantic objects
            batch_inputs = []
            for idx, row in df_to_score.iterrows():
                try:
                    s_id = str(row.get("sample_id", f"LOT-{idx+1:04d}"))
                    batch_inputs.append(
                        WinePhysicochemicalInput(
                            sample_id=s_id,
                            fixed_acidity=float(row["fixed acidity"]),
                            volatile_acidity=float(row["volatile acidity"]),
                            citric_acid=float(row["citric acid"]),
                            residual_sugar=float(row["residual sugar"]),
                            chlorides=float(row["chlorides"]),
                            free_sulfur_dioxide=float(row["free sulfur dioxide"]),
                            total_sulfur_dioxide=float(row["total sulfur dioxide"]),
                            density=float(row["density"]),
                            pH=float(row["pH"]),
                            sulphates=float(row["sulphates"]),
                            alcohol=float(row["alcohol"])
                        )
                    )
                except Exception as ex:
                    st.error(f"Row {idx} parsing failed: {ex}")

            if batch_inputs:
                batch_resp = service.predict_batch(batch_inputs, threshold=operating_threshold)

                # Summary Scorecard
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                kpi1.metric("Total Specimens", batch_resp.total_samples)
                kpi2.metric("🏆 Reserve Grade", batch_resp.premium_samples)
                kpi3.metric("🍷 Table Grade", batch_resp.standard_samples)
                kpi4.metric("Reserve Yield Rate", f"{batch_resp.premium_rate_pct:.1f}%")
                kpi5.metric("Batch Compute Time", f"{batch_resp.batch_latency_ms:.1f} ms")

                # Build Results Table
                results_data = []
                for p in batch_resp.predictions:
                    results_data.append({
                        "Sample ID": p.sample_id,
                        "Verdict": "🏆 RESERVE" if p.is_premium_wine else "🍷 STANDARD",
                        "Sensory Tier": p.sommelier_grade,
                        "Quality Prob (%)": round(p.probability_good * 100, 1),
                        "Total Acidity": p.engineered_features["total_acidity"],
                        "Free/Total SO2": p.engineered_features["free_to_total_so2_ratio"],
                        "Alc/Density": p.engineered_features["alcohol_to_density_ratio"],
                        "Cellar Guidance": p.decision_guidance
                    })
                df_results = pd.DataFrame(results_data)

                st.markdown("#### Scored Batch Roster")
                st.dataframe(df_results, use_container_width=True)

                # CSV Download Button
                csv_bytes = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Enological Batch Report (CSV)",
                    data=csv_bytes,
                    file_name="winery_batch_quality_predictions.csv",
                    mime="text/csv"
                )

    # ------------------------------------------------------------------------
    # TAB 3: Architecture & Production Governance
    # ------------------------------------------------------------------------
    with tab_diagnostics:
        st.subheader("Production System Architecture & Operational Governance")
        st.markdown(
            "Overview of the full 7-day end-to-end Machine Learning pipeline, model card, "
            "REST API specification, and vintage drift monitoring protocols."
        )

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown("#### 🏗️ Serving Pipeline Architecture")
            st.code(
                """
  +-------------------------------------------------------+
  |                   Client Requests                     |
  |  (Web Dashboard / Sommelier App / Winery LIMS / REST)  |
  +---------------------------+---------------------------+
                              | HTTP POST /predict
                              v
  +-------------------------------------------------------+
  |              FastAPI Microservice Gateway             |
  |         (Pydantic v2 Schema & Range Validation)        |
  +---------------------------+---------------------------+
                              | Validated Attributes
                              v
  +-------------------------------------------------------+
  |           Domain Feature Synthesis Engine             |
  |  (5 Biochemical Ratios: SO2, Alc/Density, Acidity)   |
  +---------------------------+---------------------------+
                              | 16-Feature Vector
                              v
  +-------------------------------------------------------+
  |         Champion Random Forest Ensemble Model         |
  |     (200 Trees, Depth 18, Class-Weight Balanced)      |
  +---------------------------+---------------------------+
                              | Posterior Probability
                              v
  +-------------------------------------------------------+
  |           Sensory Grading & Guidance Engine           |
  |     (Optimal Threshold tau=0.35 + Enological Rules)   |
  +---------------------------+---------------------------+
                              |
                              v
          Sensory Grade (Reserve vs Table Standard)
                + Cellar Operational Directives
                """,
                language="text"
            )

        with arch_col2:
            st.markdown("#### 📋 Model Card & Provenance")
            meta = service.metadata
            card_df = pd.DataFrame([
                {"Specification": "Algorithm", "Value": meta.get("model_name", "Random Forest Classifier")},
                {"Specification": "Tree Count (n_estimators)", "Value": str(meta.get("n_estimators", 200))},
                {"Specification": "Tree Depth (max_depth)", "Value": str(meta.get("max_depth", 18))},
                {"Specification": "Class Weight Strategy", "Value": str(meta.get("class_weight", "balanced"))},
                {"Specification": "Input Feature Dimensionality", "Value": "16 (11 Physicochemical + 5 Engineered)"},
                {"Specification": "Holdout Test ROC-AUC", "Value": f"{meta.get('test_roc_auc', 0.9368):.4f}"},
                {"Specification": "Holdout Test PR-AUC", "Value": f"{meta.get('test_pr_auc', 0.8095):.4f}"},
                {"Specification": "Minority Recall (tau=0.35)", "Value": "79.07% (vs 39.0% Baseline)"},
                {"Specification": "Median Single-Sample Latency", "Value": "1.2 ms (CPU)"},
                {"Specification": "Batch Throughput", "Value": "14,000+ samples / second"}
            ])
            st.table(card_df)

        st.divider()

        # Monitoring Protocol
        st.markdown("#### 📡 Vintage Drift & Retraining Standard Operating Procedure (SOP)")
        st.markdown(
            "1. **Input Data Drift Detection:** Every vintage harvest season, incoming physicochemical "
            "distributions are tested using Two-Sample Kolmogorov-Smirnov (K-S) tests against the training baseline. "
            "Significant shifts ($p < 0.01$) flag climate anomalies (e.g. drought yielding abnormal alcohol).\n"
            "2. **Cellar Performance Auditing:** Quarterly blind sommelier panels benchmark blind human tasting "
            "scores against automated classifications. If precision drops below 65% at $\\tau=0.35$, model retraining is triggered.\n"
            "3. **Continuous Integration Testing:** Automated GitHub Actions workflows re-verify end-to-end model "
            "serialization, schema bounds, and latency budgets on every code change."
        )


if __name__ == "__main__":
    main()
