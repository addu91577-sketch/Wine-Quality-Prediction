"""
Production Serving Engine and REST API for Wine Quality Prediction.
Exposes high-throughput inference endpoints, strict Pydantic v2 data contracts,
automatic chemical interaction feature engineering, enological risk alerts,
and batch scoring.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field, field_validator, ConfigDict
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware


# Default model paths
DEFAULT_MODEL_PATH = Path("models/champion_rf_pipeline.joblib")
DEFAULT_METADATA_PATH = Path("models/champion_rf_pipeline.json")

# Operational decision threshold established during Day 4 optimization
RECOMMENDED_SCREENING_THRESHOLD = 0.35

# Feature ordering expected by champion Random Forest pipeline
REQUIRED_RAW_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

FULL_FEATURE_ORDER = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "total_acidity", "bound_sulfur_dioxide",
    "free_to_total_so2_ratio", "alcohol_to_density_ratio", "sulphate_to_chloride_ratio"
]


# ============================================================================
# Pydantic Schemas / Data Contracts
# ============================================================================

class WinePhysicochemicalInput(BaseModel):
    """
    Validated physicochemical attributes of a wine laboratory specimen.
    Supports both standard field names and original dataset column aliases.
    Strict bounds prevent aberrant or biologically impossible inputs.
    """
    sample_id: Optional[str] = Field(
        default=None,
        description="Optional unique laboratory identifier or batch lot code"
    )
    fixed_acidity: float = Field(
        ...,
        alias="fixed acidity",
        ge=3.0,
        le=20.0,
        description="Tartaric acid concentration (g / dm^3)",
        examples=[8.5]
    )
    volatile_acidity: float = Field(
        ...,
        alias="volatile acidity",
        ge=0.05,
        le=2.5,
        description="Acetic acid concentration (g / dm^3)",
        examples=[0.32]
    )
    citric_acid: float = Field(
        ...,
        alias="citric acid",
        ge=0.0,
        le=1.5,
        description="Citric acid concentration (g / dm^3)",
        examples=[0.45]
    )
    residual_sugar: float = Field(
        ...,
        alias="residual sugar",
        ge=0.4,
        le=30.0,
        description="Residual sugar after fermentation (g / dm^3)",
        examples=[2.2]
    )
    chlorides: float = Field(
        ...,
        ge=0.005,
        le=1.0,
        description="Sodium chloride concentration (g / dm^3)",
        examples=[0.068]
    )
    free_sulfur_dioxide: float = Field(
        ...,
        alias="free sulfur dioxide",
        ge=1.0,
        le=120.0,
        description="Free gaseous SO2 anti-microbial agent (mg / dm^3)",
        examples=[18.0]
    )
    total_sulfur_dioxide: float = Field(
        ...,
        alias="total sulfur dioxide",
        ge=5.0,
        le=350.0,
        description="Total bound and free SO2 (mg / dm^3)",
        examples=[36.0]
    )
    density: float = Field(
        ...,
        ge=0.970,
        le=1.050,
        description="Density of wine relative to water at 20 deg C (g / cm^3)",
        examples=[0.9958]
    )
    pH: float = Field(
        ...,
        ge=2.6,
        le=4.6,
        description="Hydrogen ion concentration on logarithmic scale",
        examples=[3.32]
    )
    sulphates: float = Field(
        ...,
        ge=0.20,
        le=2.50,
        description="Potassium sulphate additive concentration (g / dm^3)",
        examples=[0.82]
    )
    alcohol: float = Field(
        ...,
        ge=7.0,
        le=17.5,
        description="Ethanol percentage by volume (% vol)",
        examples=[12.4]
    )
    cultivar: Optional[str] = Field(
        default="auto",
        description="Wine cultivar ('red', 'white', or 'auto' for automatic chemical classification)",
        examples=["auto"]
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "sample_id": "BATCH-2026-VINTAGE-001",
                "fixed acidity": 8.5,
                "volatile acidity": 0.32,
                "citric acid": 0.45,
                "residual sugar": 2.2,
                "chlorides": 0.068,
                "free sulfur dioxide": 18.0,
                "total sulfur dioxide": 36.0,
                "density": 0.9958,
                "pH": 3.32,
                "sulphates": 0.82,
                "alcohol": 12.4
            }
        }
    )

    @field_validator("total_sulfur_dioxide")
    @classmethod
    def validate_so2_consistency(cls, v: float, info) -> float:
        """Verify that total SO2 is strictly greater than or equal to free SO2."""
        free_so2 = info.data.get("free_sulfur_dioxide")
        if free_so2 is not None and v < free_so2:
            raise ValueError(
                f"total_sulfur_dioxide ({v}) cannot be less than free_sulfur_dioxide ({free_so2})"
            )
        return v

    def to_raw_dict(self) -> Dict[str, float]:
        """Convert validated model to dictionary with canonical dataset feature names."""
        return {
            "fixed acidity": float(self.fixed_acidity),
            "volatile acidity": float(self.volatile_acidity),
            "citric acid": float(self.citric_acid),
            "residual sugar": float(self.residual_sugar),
            "chlorides": float(self.chlorides),
            "free sulfur dioxide": float(self.free_sulfur_dioxide),
            "total sulfur dioxide": float(self.total_sulfur_dioxide),
            "density": float(self.density),
            "pH": float(self.pH),
            "sulphates": float(self.sulphates),
            "alcohol": float(self.alcohol),
        }


class PredictionResult(BaseModel):
    """Structured response for single sample inference."""
    sample_id: Optional[str] = None
    cultivar: Optional[str] = Field(default="red", description="Evaluated wine cultivar: 'red' or 'white'")
    cultivar_confidence: Optional[float] = Field(default=1.0, description="Confidence of cultivar detection")
    is_premium_wine: bool = Field(description="True if probability >= threshold (Score >= 7)")
    prediction_label: str = Field(description="Classification label")
    sommelier_grade: str = Field(description="Commercial sensory tier: Grand Reserve, Reserve, Table Standard, Commercial Bulk")
    probability_good: float = Field(description="Posterior probability of quality >= 7")
    probability_normal: float = Field(description="Posterior probability of quality < 7")
    operating_threshold: float = Field(description="Active decision probability cutoff")
    decision_confidence: float = Field(description="Distance of posterior probability from decision threshold")
    decision_guidance: str = Field(description="Winery production and cellar action recommendation")
    enological_alerts: List[str] = Field(default_factory=list, description="Chemical diagnostics and sensory risk warnings")
    engineered_features: Dict[str, float] = Field(description="Synthesized domain chemical interaction features")
    inference_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BatchPredictionRequest(BaseModel):
    """Payload for multi-sample laboratory batch scoring."""
    samples: List[WinePhysicochemicalInput] = Field(
        ...,
        description="Collection of wine specimen inputs (up to 5,000 samples per batch)"
    )
    threshold: Optional[float] = Field(
        default=RECOMMENDED_SCREENING_THRESHOLD,
        ge=0.05,
        le=0.95,
        description="Custom screening threshold override (default: 0.35)"
    )


class BatchPredictionResponse(BaseModel):
    """Aggregated batch inference response with summary analytics."""
    total_samples: int
    premium_samples: int
    standard_samples: int
    premium_rate_pct: float
    operating_threshold: float
    mean_probability_good: float
    batch_latency_ms: float
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    """System health and model readiness response."""
    status: str
    model_loaded: bool
    model_name: str
    model_path: str
    uptime_seconds: float
    server_time: str
    api_version: str = "1.0.0"


class MetadataResponse(BaseModel):
    """Comprehensive champion model specifications and training provenance."""
    model_name: str
    project: str
    date_finalized: str
    n_estimators: int
    max_depth: Optional[int] = None
    class_weight: str
    features: List[str]
    num_features: int
    test_roc_auc: float
    test_pr_auc: float
    test_balanced_accuracy_screen_threshold: float
    operating_screening_threshold: float
    loaded_artifact_path: str


class FeatureDriftItem(BaseModel):
    """Statistical drift evaluation for an individual feature."""
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    wasserstein_distance: float
    drift_status: str
    alert_level: str


class DriftAuditResponse(BaseModel):
    """Consolidated drift audit response across an incoming laboratory batch."""
    total_features_evaluated: int
    features_red_alert: int
    features_amber_alert: int
    features_green: int
    overall_drift_status: str
    retraining_recommended: bool
    triggered_reasons: List[str]
    feature_drift_scorecard: List[FeatureDriftItem]
    timestamp: str


class DriftMetricsResponse(BaseModel):
    """System drift monitoring parameters and PSI alerting thresholds."""
    status: str
    baseline_dataset_size: int
    baseline_features: List[str]
    psi_thresholds: Dict[str, float]
    monitoring_mode: str
    server_time: str


class CultivarDetectionResponse(BaseModel):
    """Detection result identifying whether sample is Red or White wine."""
    cultivar: str = Field(description="Predicted cultivar: 'red' or 'white'")
    confidence: float = Field(description="Detection probability confidence")
    prob_red: float = Field(description="Probability of red cultivar")
    prob_white: float = Field(description="Probability of white cultivar")


class RecourseInterventionItem(BaseModel):
    """Single recommended cellar chemical adjustment."""
    feature: str
    original_value: float
    target_value: float
    absolute_change: float
    percentage_change: float
    recommendation: str


class CellarRecourseResponse(BaseModel):
    """Actionable prescriptive cellar recourse plan to achieve premium quality."""
    status: str
    cultivar: str
    initial_prob: float
    target_prob: float
    projected_prob: float
    probability_gain: float
    operating_threshold: float
    upgraded_to_premium: bool
    feasibility: str
    interventions: List[RecourseInterventionItem]
    adjusted_sample: Dict[str, float]


class LotItemInput(WinePhysicochemicalInput):
    """Input specification for a candidate wine lot in coupage optimization."""
    lot_name: Optional[str] = Field(default="Candidate Lot", alias="lot_name")
    volume_liters: Optional[float] = Field(default=1000.0, alias="volume_liters")
    cost_per_liter: Optional[float] = Field(default=3.50, alias="cost_per_liter")


class EnologicalConstraintsInput(BaseModel):
    """Configurable boundaries for enological coupage optimization."""
    max_volatile_acidity: Optional[float] = 0.60
    max_total_so2: Optional[float] = 160.0
    min_free_so2: Optional[float] = 20.0
    min_alcohol: Optional[float] = 11.5
    min_ph: Optional[float] = 3.10
    max_ph: Optional[float] = 3.65


class AssemblageOptimizationRequest(BaseModel):
    """Request payload for multi-lot wine coupage optimization."""
    lots: List[LotItemInput] = Field(..., min_length=2, description="At least two candidate wine lots")
    constraints: Optional[EnologicalConstraintsInput] = None
    cost_penalty_weight: float = Field(default=0.0, ge=0.0, description="Cost weight lambda in optimization")


class AssemblageOptimizationResponse(BaseModel):
    """Response payload for multi-lot coupage optimization."""
    status: str
    optimal_weights: List[float]
    weights_breakdown: List[Dict[str, Any]]
    blended_chemistry: Dict[str, float]
    predicted_premium_probability: float
    operating_threshold: float
    is_premium: bool
    sommelier_grade: str
    prob_lift_vs_mean_lot: float
    prob_lift_vs_best_lot: float
    individual_lot_evaluations: List[Dict[str, Any]]
    constraints_compliant: bool
    enological_compliance_report: Dict[str, Any]


class ConformalPredictionResponse(BaseModel):
    """Finite-sample distribution-free conformal prediction set and uncertainty report."""
    prediction_set: List[int]
    prediction_labels: List[str]
    set_size: int
    confidence_pct: str
    prob_premium: float
    prob_standard: float
    conformal_quantile_threshold: float
    probability_inclusion_cutoff: float
    is_ambiguous: bool
    is_empty: bool
    verdict: str
    sommelier_action: str


class EconomicEvaluationRequest(BaseModel):
    """Commercial winery financial payoff parameters."""
    lot_size_bottles: int = Field(default=25000, ge=100)
    tp_margin: float = Field(default=28.00)
    fp_penalty: float = Field(default=-45.00)
    fn_penalty: float = Field(default=-12.00)
    tn_margin: float = Field(default=4.50)


class EconomicEvaluationResponse(BaseModel):
    """Financial optimization report identifying optimal commercial release threshold."""
    optimal_financial_threshold: float
    max_harvest_profit: float
    max_profit_per_bottle: float
    default_threshold_profit: float
    f1_threshold_profit: float
    dollar_uplift_vs_default: float
    percent_uplift_vs_default: float
    best_evaluation: Dict[str, Any]
    default_threshold_evaluation: Dict[str, Any]


class StackingBenchmarkResponse(BaseModel):
    """Benchmarking metrics for multi-model stacking ensemble vs base learners."""
    model_name: str
    stacking_roc_auc: float
    stacking_pr_auc: float
    stacking_brier_score: float
    stacking_balanced_accuracy: float
    operating_threshold: float
    meta_learner_coefficients: Dict[str, float]
    base_models_benchmarks: Dict[str, Dict[str, float]]


class AgingSimulationRequest(BaseModel):
    """Request parameters for bottle cellar aging simulation."""
    sample: WinePhysicochemicalInput = Field(..., description="Laboratory chemistry of bottled vintage")
    months: int = Field(default=60, ge=6, le=120, description="Simulation duration horizon in months")
    temperature_c: float = Field(default=14.0, ge=5.0, le=30.0, description="Cellar ambient temperature in Celsius")
    closure_type: str = Field(default="natural_cork", description="Bottle closure: natural_cork, screwcap, synthetic, technical_cork")
    step_months: int = Field(default=1, ge=1, le=12)


class AgingSimulationResponse(BaseModel):
    """Dynamic chemical and sensory quality trajectory across cellar aging."""
    cellar_temperature_c: float
    closure_type: str
    so2_decay_rate_k: float
    simulation_months: int
    initial_probability: float
    peak_probability: float
    peak_month: int
    drinking_window_start_month: Optional[int]
    drinking_window_end_month: Optional[int]
    spoilage_risk_month: Optional[int]
    sommelier_cellar_advice: str
    trajectory_preview: List[Dict[str, Any]]


class ParetoCandidateItem(WinePhysicochemicalInput):
    """Candidate lot for multi-objective optimization."""
    name: Optional[str] = Field(default="Candidate Lot")


class ParetoOptimizationRequest(BaseModel):
    """Request for non-dominated multi-objective Pareto analysis."""
    candidates: List[ParetoCandidateItem] = Field(..., min_length=2)


class ParetoOptimizationResponse(BaseModel):
    """Multi-objective Pareto frontier and compromise archetypes."""
    total_candidates_evaluated: int
    pareto_optimal_count: int
    pareto_frontier: List[Dict[str, Any]]
    knee_point_lot: Optional[Dict[str, Any]]
    archetypes: Dict[str, Any]


class StressTestRequest(BaseModel):
    """Request for physicochemical environmental perturbation stress test."""
    sample: WinePhysicochemicalInput = Field(..., description="Physicochemical wine lot to stress-test")


class StressScenarioResult(BaseModel):
    """Stress response under individual disturbance scenario."""
    scenario_id: str
    scenario_name: str
    description: str
    perturbed_probability: float
    probability_delta: float
    retained_classification: bool
    perturbed_is_premium: bool


class StressTestResponse(BaseModel):
    """Consolidated physicochemical perturbation and stability audit."""
    baseline_probability: float
    baseline_is_premium: bool
    decision_threshold: float
    scenarios_tested: int
    classification_flips: int
    resilience_score: float
    resilience_tier: str
    critical_va_epsilon_flip_margin: float
    scenario_results: List[StressScenarioResult]


class ShapInteractionsResponse(BaseModel):
    """Pairwise chemical synergy and molecular equilibrium interactions."""
    interactions: Dict[str, Any]


# ============================================================================
# Day 12: Fermentation Bioprocess, Stuck Early-Warning & Sensory Schemas
# ============================================================================

class FermentationSimulationRequest(BaseModel):
    """Parameters for numerical bioprocess simulation of primary alcoholic fermentation."""
    initial_sugar_g_l: float = Field(default=225.0, ge=100.0, le=350.0, description="Initial hexose sugar concentration in g/L")
    initial_yan_mg_l: float = Field(default=220.0, ge=20.0, le=500.0, description="Initial Yeast Assimilable Nitrogen in mg/L")
    initial_temp_c: float = Field(default=20.0, ge=10.0, le=35.0, description="Initial must fermentation temperature in Celsius")
    jacket_temp_c: float = Field(default=20.0, ge=10.0, le=35.0, description="Cooling jacket setpoint temperature in Celsius")
    initial_biomass_g_l: float = Field(default=0.5, ge=0.05, le=5.0, description="Inoculated yeast biomass in g/L")
    total_days: int = Field(default=14, ge=3, le=30, description="Fermentation simulation duration in days")
    yeast_strain: str = Field(default="EC1118_champagne", description="Yeast strain profile (EC1118_champagne, RC212_pinot_red, D254_rhone_bold, QA23_aromatic_white)")


class FermentationSimulationResponse(BaseModel):
    """Results from bioprocess kinetic simulation."""
    yeast_strain: str
    fermentation_status: str
    duration_days: int
    final_residual_sugar_g_l: float
    final_ethanol_pct: float
    final_volatile_acidity_g_l: float
    peak_fermentation_rate_g_l_day: float
    peak_rate_day: float
    predicted_quality_prob: float
    is_premium: bool
    trajectory: List[Dict[str, Any]]


class StuckDiagnosisReading(BaseModel):
    """Single timestamp observation in a fermentation log."""
    day: float = Field(..., ge=0.0, description="Elapsed day of fermentation")
    sugar: float = Field(..., ge=0.0, description="Residual hexose sugar in g/L")
    temp: Optional[float] = Field(default=20.0, description="Tank temperature in Celsius")
    yan: Optional[float] = Field(default=150.0, description="YAN in mg/L")
    alcohol: Optional[float] = Field(default=0.0, description="Current ethanol in % v/v")


class StuckDiagnosisRequest(BaseModel):
    """Sequential time-series log for kinetic stuck/sluggish triage."""
    readings: List[StuckDiagnosisReading] = Field(..., min_length=2, description="At least 2 chronological fermentation measurements")


class StuckDiagnosisResponse(BaseModel):
    """Early warning status, kinetic derivatives, and prescriptive intervention."""
    diagnosis: str
    risk_score_pct: float
    is_stuck_or_sluggish: bool
    current_sugar_g_l: float
    current_rate_g_l_day: float
    acceleration_g_l_day2: float
    current_temperature_c: float
    current_ethanol_pct: float
    message: str
    prescriptive_actions: List[Dict[str, str]]


class SensoryProfileRequest(BaseModel):
    """Request for 6-axis organoleptic radar, sommelier tasting note, and food pairings."""
    sample: WinePhysicochemicalInput = Field(..., description="Laboratory chemistry of wine sample")


class SensoryProfileResponse(BaseModel):
    """Calculated sensory organoleptic profile."""
    radar_scores: Dict[str, float]
    overall_sensory_intensity: float
    dry_extract_est_g_l: float
    tasting_note: Dict[str, str]
    food_pairings: List[Dict[str, Any]]


class CausalMetricsResponse(BaseModel):
    """Causal Average Treatment Effects and confounding bias mitigation metrics."""
    interventions: Dict[str, Any]


# ============================================================================
# Day 13 Schemas: Malolactic, Oak Aging, Cellar Dispatch & HACCP Compliance
# ============================================================================

class MalolacticSimulationRequest(BaseModel):
    """Parameters for secondary malolactic fermentation biotransformation simulation."""
    initial_malic_acid: float = Field(default=2.8, ge=0.2, le=8.0, description="Initial L-malic acid in g/L")
    initial_ph: float = Field(default=3.35, ge=2.8, le=4.2, description="Initial must/wine pH")
    initial_fixed_acidity: float = Field(default=7.2, ge=3.5, le=15.0, description="Initial fixed acidity in g/L")
    alcohol: float = Field(default=13.0, ge=8.0, le=17.0, description="Ethanol concentration in % v/v")
    free_so2: float = Field(default=6.0, ge=0.0, le=40.0, description="Free SO2 in mg/L")
    temperature: float = Field(default=20.0, ge=10.0, le=32.0, description="Cellar temperature in Celsius")
    inoculated_biomass: float = Field(default=0.05, ge=0.005, le=1.0, description="Initial Oenococcus oeni biomass in g/L")
    duration_days: int = Field(default=35, ge=5, le=90, description="Simulation duration in days")


class MalolacticSimulationResponse(BaseModel):
    """Dynamic results of secondary malolactic fermentation."""
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    diagnostics: Dict[str, Any]
    trajectory: Dict[str, List[float]]


class OakMaturationRequest(BaseModel):
    """Parameters for oak barrel wood extractive and color stabilization simulation."""
    oak_species: str = Field(default="french", description="Oak species (french, american, hungarian)")
    toast_level: str = Field(default="medium_plus", description="Toast intensity (light, medium, medium_plus, heavy)")
    barrel_usage: str = Field(default="new", description="Barrel generation (new, first_fill, second_fill, neutral)")
    maturation_months: int = Field(default=18, ge=1, le=48, description="Barrel maturation duration in months")
    barrel_volume_liters: float = Field(default=225.0, ge=50.0, le=10000.0, description="Vessel volume in Liters")


class OakMaturationResponse(BaseModel):
    """Results of wood phenolic extraction and micro-oxygenation tannin polymerization."""
    parameters: Dict[str, Any]
    final_concentrations: Dict[str, Any]
    trajectory: Dict[str, List[float]]


class CellarVesselItem(BaseModel):
    """Single winery cellar vessel reading and status."""
    vessel_id: str = Field(default="TANK-01", description="Unique vessel identifier")
    vessel_type: str = Field(default="BARREL", description="Vessel category (BARREL, TANK, PUNCHEON, FOUDRE)")
    cultivar: str = Field(default="red", description="Wine cultivar (red or white)")
    volume_liters: float = Field(default=225.0, ge=10.0, description="Volume in Liters")
    ph: float = Field(default=3.4, ge=2.8, le=4.3, description="Measured pH")
    free_so2: float = Field(default=15.0, ge=0.0, le=100.0, description="Free SO2 in mg/L")
    total_so2: float = Field(default=60.0, ge=0.0, le=300.0, description="Total SO2 in mg/L")
    volatile_acidity: float = Field(default=0.45, ge=0.0, le=2.5, description="Volatile acidity in g/L")
    alcohol: float = Field(default=13.0, ge=7.0, le=18.0, description="Ethanol % v/v")
    temperature: float = Field(default=18.0, ge=5.0, le=35.0, description="Cellar temperature in Celsius")
    residual_sugar: float = Field(default=2.0, ge=0.0, le=150.0, description="Residual sugar in g/L")
    malic_acid: float = Field(default=0.05, ge=0.0, le=8.0, description="Residual malic acid in g/L")
    maturation_days: int = Field(default=120, ge=0, description="Days in current vessel")
    status: str = Field(default="aging", description="Vessel state (fermentation, mlf, aging, resting)")


class CellarDispatchRequest(BaseModel):
    """Fleet inventory of winery vessels for triage and work order generation."""
    inventory: List[CellarVesselItem] = Field(..., min_length=1, description="List of cellar vessels")


class CellarDispatchResponse(BaseModel):
    """Aggregated triage metrics and prioritized cellar work orders."""
    total_vessels: int
    tier_distribution: Dict[str, int]
    cellar_health_score: float
    aggregate_pms_reagents_required_kg: float
    dispatched_vessels: List[Dict[str, Any]]


class HaccpAuditRequest(BaseModel):
    """Wine lot parameters for statutory HACCP food safety and biogenic amine risk audit."""
    lot_id: str = Field(default="LOT-2026-CHAMPION", description="Wine batch identifier")
    cultivar: str = Field(default="red", description="Cultivar (red or white)")
    ph: float = Field(default=3.45, ge=2.8, le=4.3, description="Measured pH")
    free_so2: float = Field(default=22.0, ge=0.0, le=100.0, description="Free SO2 in mg/L")
    total_so2: float = Field(default=85.0, ge=0.0, le=300.0, description="Total SO2 in mg/L")
    volatile_acidity: float = Field(default=0.48, ge=0.0, le=2.5, description="Volatile acidity in g/L")
    alcohol: float = Field(default=13.5, ge=7.0, le=18.0, description="Ethanol % v/v")
    residual_sugar: float = Field(default=2.1, ge=0.0, le=150.0, description="Residual sugar in g/L")
    malic_acid: float = Field(default=0.04, ge=0.0, le=8.0, description="Residual malic acid in g/L")
    temperature: float = Field(default=16.0, ge=5.0, le=35.0, description="Storage temperature in Celsius")


class HaccpAuditResponse(BaseModel):
    """HACCP CCP verification report and biogenic amine hazard evaluation."""
    lot_id: str
    cultivar: str
    audit_status: str
    all_ccp_passed: bool
    summary: str
    ccp_verifications: Dict[str, Any]
    biogenic_amine_risk: Dict[str, Any]
    audit_findings: List[str]


# ============================================================================
# Day 14 Schemas: Tartrate, Bentonite, Membrane Filtration & Packaging
# ============================================================================

class TartrateEvaluationRequest(BaseModel):
    """Parameters for evaluating potassium bitartrate thermodynamic stability."""
    tartaric_acid_g_l: float = Field(default=6.8, ge=2.0, le=16.0, description="Tartaric acid in g/L")
    potassium_mg_l: float = Field(default=1050.0, ge=200.0, le=2500.0, description="Potassium ion in mg/L")
    ph: float = Field(default=3.45, ge=2.8, le=4.2, description="Wine pH")
    alcohol_pct: float = Field(default=13.0, ge=8.0, le=18.0, description="Ethanol % v/v")
    storage_temp_c: float = Field(default=14.0, ge=-5.0, le=35.0, description="Cellar storage temperature in Celsius")


class TartrateEvaluationResponse(BaseModel):
    """Results of thermodynamic tartrate stability and conductivity drop evaluation."""
    potassium_molar: float
    bitartrate_molar: float
    bitartrate_fraction: float
    concentration_product_cp: float
    solubility_product_ksp: float
    saturation_ratio: float
    saturation_temperature_c: float
    storage_temp_c: float
    minicontact_delta_chi_us_cm: float
    stability_status: str
    haze_crystal_risk: str
    action_required: bool


class ColdStabilizationSimulationRequest(BaseModel):
    """Parameters for simulating dynamic chilled contact crystallization."""
    tartaric_acid_g_l: float = Field(default=7.0, ge=2.0, le=16.0, description="Initial tartaric acid in g/L")
    potassium_mg_l: float = Field(default=1100.0, ge=300.0, le=2500.0, description="Initial potassium in mg/L")
    ph: float = Field(default=3.50, ge=2.8, le=4.2, description="Wine pH")
    alcohol_pct: float = Field(default=13.5, ge=8.0, le=18.0, description="Ethanol % v/v")
    chilling_temp_c: float = Field(default=-4.0, ge=-8.0, le=10.0, description="Chilling temperature in Celsius")
    seed_crystal_g_l: float = Field(default=4.0, ge=0.0, le=10.0, description="KHT seed crystals in g/L")
    duration_hours: int = Field(default=12, ge=2, le=72, description="Contact process duration in hours")


class ColdStabilizationSimulationResponse(BaseModel):
    """Dynamic results of cold contact tartrate stabilization."""
    parameters: Dict[str, Any]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    trajectory: Dict[str, List[Any]]


class BentoniteOptimizationRequest(BaseModel):
    """Parameters for heat test protein haze risk and Langmuir bentonite dosage optimization."""
    protein_content_mg_l: float = Field(default=110.0, ge=5.0, le=400.0, description="Thermolabile PR protein content in mg/L")
    ph: float = Field(default=3.25, ge=2.8, le=4.2, description="Wine pH")
    cultivar: str = Field(default="white", description="Wine cultivar ('red' or 'white')")
    bentonite_type: str = Field(default="sodium_bentonite", description="Bentonite type: 'sodium_bentonite' or 'calcium_bentonite'")


class BentoniteOptimizationResponse(BaseModel):
    """Optimal bentonite fining dosage and aroma preservation evaluation."""
    protein_initial_mg_l: float
    heat_delta_ntu_initial: float
    bentonite_type: Optional[str] = None
    recommended_bentonite_g_hl: float
    residual_protein_mg_l: float
    heat_delta_ntu_final: float
    aroma_stripping_penalty_pct: float
    status: str


class CrossflowFiltrationRequest(BaseModel):
    """Parameters for crossflow membrane microfiltration simulation (0.45 um PES)."""
    initial_turbidity_ntu: float = Field(default=35.0, ge=1.0, le=300.0, description="Feed wine turbidity in NTU")
    transmembrane_pressure_bar: float = Field(default=1.6, ge=0.5, le=3.5, description="Transmembrane pressure (TMP) in bar")
    target_volume_liters: float = Field(default=10000.0, ge=500.0, le=100000.0, description="Batch volume to filter in liters")
    colloidal_index: float = Field(default=1.2, ge=0.2, le=3.5, description="Colloidal fouling index")
    backwash_interval_hours: float = Field(default=2.0, ge=0.5, le=6.0, description="Hydraulic backwash interval in hours")
    run_duration_hours: float = Field(default=6.0, ge=1.0, le=24.0, description="Simulation duration in hours")


class CrossflowFiltrationResponse(BaseModel):
    """Dynamic permeate flux, turbidity reduction and filterability diagnostics."""
    parameters: Dict[str, Any]
    summary: Dict[str, Any]
    trajectory: Dict[str, List[Any]]


class GasSpargingRequest(BaseModel):
    """Parameters for micro-bubble nitrogen sparging deoxygenation."""
    initial_do_mg_l: float = Field(default=1.4, ge=0.2, le=12.0, description="Initial dissolved oxygen in mg/L")
    target_do_mg_l: float = Field(default=0.40, ge=0.1, le=2.0, description="Target packaging dissolved oxygen in mg/L")
    nitrogen_flow_ratio_nl_l: float = Field(default=0.5, ge=0.05, le=3.0, description="Norm-liters N2 per liter wine")
    temperature_c: float = Field(default=18.0, ge=5.0, le=35.0, description="Wine temperature in Celsius")


class GasSpargingResponse(BaseModel):
    """Deoxygenation efficiency and residual DO results."""
    initial_do_mg_l: float
    final_do_mg_l: float
    target_do_mg_l: float
    nitrogen_ratio_nl_l: float
    oxygen_removed_mg_l: float
    deoxygenation_efficiency_pct: float
    target_reached: bool


class TotalPackagingOxygenRequest(BaseModel):
    """Parameters for calculating Total Packaging Oxygen (TPO)."""
    dissolved_oxygen_wine_mg_l: float = Field(default=0.45, ge=0.05, le=10.0, description="Dissolved oxygen in wine (mg/L)")
    headspace_volume_ml: float = Field(default=6.0, ge=1.0, le=25.0, description="Bottle headspace volume in mL")
    bottle_volume_ml: float = Field(default=750.0, ge=187.0, le=1500.0, description="Nominal bottle volume in mL")
    headspace_o2_pct: float = Field(default=1.8, ge=0.1, le=22.0, description="Oxygen percentage in bottle headspace")
    wine_temperature_c: float = Field(default=18.0, ge=5.0, le=30.0, description="Wine temperature in Celsius")


class TotalPackagingOxygenResponse(BaseModel):
    """TPO calculation, compartmental breakdown, and SO2 depletion estimation."""
    dissolved_oxygen_mg_l: float
    headspace_oxygen_mg_l_wine: float
    headspace_o2_pct: float
    total_packaging_oxygen_tpo_mg_l: float
    stoichiometric_free_so2_depletion_mg_l: float
    tpo_grade: str
    is_within_premium_spec: bool


class ClosureShelfLifeRequest(BaseModel):
    """Parameters for predicting bottle shelf-life and oxidation kinetics."""
    closure_type: str = Field(default="diam_5_microagglomerated", description="Closure type key from CLOSURE_PROFILES")
    initial_free_so2: float = Field(default=32.0, ge=5.0, le=80.0, description="Free SO2 at bottling in mg/L")
    initial_ph: float = Field(default=3.45, ge=2.8, le=4.2, description="Wine pH")
    initial_tpo_mg_l: float = Field(default=0.85, ge=0.1, le=6.0, description="Total Packaging Oxygen at bottling (mg/L)")
    storage_temperature_c: float = Field(default=14.0, ge=5.0, le=35.0, description="Storage temperature in Celsius")
    duration_months: int = Field(default=48, ge=6, le=120, description="Simulation duration in months")


class ClosureShelfLifeResponse(BaseModel):
    """Predictive shelf-life milestones, browning kinetics and PremOx boundary."""
    closure_info: Dict[str, Any]
    parameters: Dict[str, Any]
    milestones: Dict[str, Any]
    final_state: Dict[str, Any]
    trajectory: Dict[str, List[Any]]


class BottlingQCAuditRequest(BaseModel):
    """Batch inspection samples for statistical process control (SPC) bottling line audit."""
    lot_id: str = Field(default="LOT_ESTATE_B2026_01", description="Commercial bottling run lot identifier")
    fill_volumes_ml: List[float] = Field(description="Net fill volume samples in mL (Nominal 750 mL)")
    closure_depths_mm: List[float] = Field(description="Closure insertion depth samples in mm (Nominal 1.0 mm)")
    headspace_vacuums_bar: List[float] = Field(description="Headspace vacuum samples in bar (Nominal -0.25 bar)")
    filter_bubble_point_bar: float = Field(default=3.4, ge=1.0, le=6.0, description="0.45 um filter bubble point in bar")


class BottlingQCAuditResponse(BaseModel):
    """Statistical Process Control results, legal metrology compliance, and capability metrics."""
    lot_id: str
    sample_count: int
    audit_tier: str
    is_batch_approved: bool
    fill_volume_spc: Dict[str, Any]
    closure_insertion_spc: Dict[str, Any]
    headspace_vacuum_spc: Dict[str, Any]
    membrane_integrity: Dict[str, Any]


# ============================================================================
# Day 15 Pydantic Schemas: Cold-Chain, Terroir, Hedonics & Winery Twin
# ============================================================================

class TransitSimulationRequest(BaseModel):
    """Parameters for intermodal transportation stress evaluation."""
    preset_name: str = Field(default="maritime_reefer_rotterdam_singapore", description="Transit preset key")
    closure_type: str = Field(default="natural_cork_fleur", description="Closure type")
    initial_free_so2: float = Field(default=34.0, ge=0.0, le=80.0, description="Initial Free SO2 in mg/L")
    initial_a420_browning: float = Field(default=0.120, ge=0.02, le=0.80, description="Initial browning optical density A420")
    dissolved_co2_mg_l: float = Field(default=850.0, ge=200.0, le=2500.0, description="Dissolved CO2 in mg/L")


class TransitSimulationResponse(BaseModel):
    """Transportation stress results including thermal damage units and closure seal risk."""
    preset_name: str
    route_description: str
    transit_mode: str
    duration_days: float
    closure_type: str
    temperature_summary: Dict[str, Any]
    kinetic_stress_metrics: Dict[str, Any]
    chemical_impact: Dict[str, Any]
    barometric_seal_integrity: Dict[str, Any]


class BarometricSealRequest(BaseModel):
    """Parameters for trans-closure differential pressure assessment."""
    liquid_temp_c: float = Field(default=28.0, ge=0.0, le=55.0, description="Liquid wine temperature in °C")
    ambient_pressure_kpa: float = Field(default=75.2, ge=50.0, le=120.0, description="Ambient cargo pressure in kPa")
    dissolved_co2_mg_l: float = Field(default=900.0, ge=100.0, le=2500.0, description="Dissolved CO2 in mg/L")
    headspace_volume_ml: float = Field(default=4.5, ge=1.0, le=15.0, description="Headspace volume in mL")


class BarometricSealResponse(BaseModel):
    """Headspace pressure differential and seal failure risk."""
    liquid_temp_c: float
    ambient_pressure_kpa: float
    effective_headspace_ml: float
    gas_partial_pressure_kpa: float
    co2_partial_pressure_kpa: float
    total_internal_pressure_kpa: float
    delta_p_differential_kpa: float


class TerroirVerificationRequest(BaseModel):
    """Parameters for verifying appellation terroir and provenance authenticity."""
    claimed_appellation: str = Field(default="bordeaux_left_bank", description="Claimed AOC/DOC/AVA appellation")
    sample_minerals: Dict[str, float] = Field(description="ICP-MS elemental concentrations (Li, Rb, Sr, Ba, Mn, B) in ppm")
    measured_sr_ratio: float = Field(default=0.7088, ge=0.7000, le=0.7300, description="87Sr/86Sr radiogenic strontium ratio")
    measured_delta13c: float = Field(default=-27.1, ge=-32.0, le=-10.0, description="Carbon stable isotope ratio in per mil VPDB")
    measured_delta18o: float = Field(default=3.6, ge=-12.0, le=10.0, description="Oxygen stable isotope ratio in per mil VSMOW")


class TerroirVerificationResponse(BaseModel):
    """Terroir chemometric authenticity audit and fraud detection verdict."""
    claimed_appellation: str
    appellation_name: str
    bedrock_geology: str
    mahalanobis_terroir_distance: float
    terroir_p_value: float
    is_authentic: bool
    verdict: str
    confidence_pct: float
    elemental_z_scores: Dict[str, float]
    isotopic_audits: Dict[str, Any]


class ChaptalizationAuditRequest(BaseModel):
    """Parameters for detecting exogenous C4 sugar chaptalization."""
    measured_delta13c: float = Field(default=-24.2, ge=-32.0, le=-10.0, description="Measured delta13C VPDB in per mil")
    baseline_grape_delta13c: float = Field(default=-27.2, description="Baseline pure C3 grape delta13C in per mil")


class ChaptalizationAuditResponse(BaseModel):
    """Exogenous sugar fraction calculation and legal compliance verdict."""
    measured_delta13c_permil: float
    baseline_grape_delta13c_permil: float
    estimated_c4_sugar_fraction_pct: float
    is_adulterated: bool
    compliance_verdict: str


class HedonicValuationRequest(BaseModel):
    """Parameters for Rosen's hedonic econometric price estimation."""
    predicted_quality: float = Field(default=7.2, ge=3.0, le=10.0, description="Predicted wine quality score")
    cultivar: str = Field(default="red", description="Wine cultivar (red or white)")
    alcohol: float = Field(default=13.8, ge=8.0, le=16.0, description="Alcohol content % vol")
    ph: float = Field(default=3.52, ge=2.8, le=4.2, description="Wine pH")
    residual_sugar: float = Field(default=2.4, ge=0.5, le=50.0, description="Residual sugar in g/L")
    volatile_acidity: float = Field(default=0.42, ge=0.1, le=1.8, description="Volatile acidity in g/L")
    cellar_aging_years: float = Field(default=2.5, ge=0.0, le=30.0, description="Cellar maturation in years")
    appellation_tier: str = Field(default="doc_quality", description="Appellation tier (table, igt, doc_quality, grand_cru)")


class HedonicValuationResponse(BaseModel):
    """Fair market bottle valuation and confidence interval."""
    predicted_quality: float
    appellation_tier: str
    estimated_market_price_usd: float
    confidence_interval_90_usd: List[float]
    hedonic_components: Dict[str, Any]


class RegionalPalateRequest(BaseModel):
    """Parameters for evaluating cross-regional consumer palate alignment."""
    alcohol: float = Field(default=14.0, ge=8.0, le=16.0, description="Alcohol % vol")
    ph: float = Field(default=3.62, ge=2.8, le=4.2, description="Wine pH")
    residual_sugar: float = Field(default=3.0, ge=0.5, le=30.0, description="Residual sugar in g/L")
    volatile_acidity: float = Field(default=0.44, ge=0.1, le=1.5, description="Volatile acidity in g/L")
    sulphates: float = Field(default=0.68, ge=0.2, le=2.0, description="Sulphates in g/L")
    oak_index: float = Field(default=0.80, ge=0.0, le=1.0, description="Oak extract index (0 - 1)")


class RegionalPalateResponse(BaseModel):
    """Consumer alignment scores across global markets."""
    regions: Dict[str, Any]


class DigitalWineryTwinRequest(BaseModel):
    """Parameters for end-to-end multi-stage digital twin simulation."""
    cultivar: str = Field(default="red", description="Wine cultivar")
    harvest_sugar_brix: float = Field(default=24.5, ge=18.0, le=32.0, description="Harvest must sugar in °Brix")
    must_tartaric_acid: float = Field(default=6.8, ge=4.0, le=12.0, description="Must tartaric acid in g/L")
    must_ph: float = Field(default=3.55, ge=2.8, le=4.0, description="Must pH")
    oak_regime: str = Field(default="french_medium_plus", description="Oak barrel regime")
    oak_duration_months: int = Field(default=12, ge=0, le=36, description="Oak maturation months")
    stabilization_method: str = Field(default="contact_chilling_seeding", description="Stabilization technique")
    closure_choice: str = Field(default="technical_diam", description="Closure type")
    shipping_route: str = Field(default="maritime_reefer_rotterdam_singapore", description="Intermodal shipping route")
    appellation_claimed: str = Field(default="bordeaux_left_bank", description="Claimed appellation")
    production_lot_bottles: int = Field(default=20000, ge=500, le=500000, description="Production lot volume in bottles")


class DigitalWineryTwinResponse(BaseModel):
    """Holistic winery twin lifecycle simulation report."""
    simulation_id: str
    cultivar: str
    lot_size_bottles: int
    stage_1_fermentation: Dict[str, Any]
    stage_2_maturation: Dict[str, Any]
    stage_3_stabilization: Dict[str, Any]
    stage_4_quality_scoring: Dict[str, Any]
    stage_5_cold_chain_logistics: Dict[str, Any]
    stage_6_terroir_authenticity: Dict[str, Any]
    stage_7_hedonic_and_pricing: Dict[str, Any]
    stage_8_esg_lca_sustainability: Dict[str, Any]


class SustainabilityLCARequest(BaseModel):
    """Parameters for Scope 1, 2, 3 carbon footprint and water LCA."""
    bottle_glass_weight_g: float = Field(default=420.0, ge=320.0, le=1000.0, description="Empty glass bottle weight in grams")
    is_organic_certified: bool = Field(default=True, description="Organic vineyard certification")
    fermentation_co2_captured: bool = Field(default=True, description="Fermentation biogenic CO2 captured")
    transit_distance_km: float = Field(default=6500.0, ge=50.0, le=25000.0, description="Freight shipping distance in km")
    freight_mode: str = Field(default="maritime", description="Freight mode: maritime, truck, air_cargo")


class SustainabilityLCAResponse(BaseModel):
    """GHG emissions per bottle and ESG eco-rating."""
    scope_1_direct_g_co2e: float
    scope_2_electricity_g_co2e: float
    scope_3_supply_chain_g_co2e: float
    total_footprint_g_co2e_per_bottle: float
    glass_bottle_contribution_pct: float
    freight_transport_contribution_pct: float
    water_footprint_liters_per_liter: float
    sustainability_rating: str
    carbon_neutrality_offset_cost_usd: float



# ============================================================================
# Domain Feature Engineering & Chemical Diagnostics
# ============================================================================

def derive_interaction_features(raw_features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute 5 biochemical interaction features identically to training pipeline.

    Formulas:
    1. total_acidity = fixed acidity + volatile acidity
    2. bound_sulfur_dioxide = total sulfur dioxide - free sulfur dioxide
    3. free_to_total_so2_ratio = free sulfur dioxide / (total sulfur dioxide + 1e-5)
    4. alcohol_to_density_ratio = alcohol / density
    5. sulphate_to_chloride_ratio = sulphates / (chlorides + 1e-5)
    """
    d = dict(raw_features)
    d["total_acidity"] = float(d["fixed acidity"] + d["volatile acidity"])
    d["bound_sulfur_dioxide"] = float(d["total sulfur dioxide"] - d["free sulfur dioxide"])
    d["free_to_total_so2_ratio"] = float(d["free sulfur dioxide"] / (d["total sulfur dioxide"] + 1e-5))
    d["alcohol_to_density_ratio"] = float(d["alcohol"] / d["density"])
    d["sulphate_to_chloride_ratio"] = float(d["sulphates"] / (d["chlorides"] + 1e-5))
    return d


def evaluate_enological_rules(raw: Dict[str, float], prob_good: float) -> tuple[str, str, List[str]]:
    """
    Generate commercial sensory grade, production cellar guidance, and chemical alerts.
    Uses ASCII-safe markers to prevent terminal encoding failures.
    """
    alerts: List[str] = []

    # Check for volatile acidity (acetic acid / vinegar spoilage)
    va = raw.get("volatile acidity", 0.0)
    if va >= 0.70:
        alerts.append(f"[WARNING] High Volatile Acidity ({va:.2f} g/dm3): Risk of acetic spoilage, pungent vinegar notes.")
    elif va <= 0.40:
        alerts.append(f"[OPTIMAL] Clean Volatile Acidity ({va:.2f} g/dm3): Pure fermentation aroma, minimal volatile acid.")

    # Check for free SO2 protection vs oxidation
    free_so2 = raw.get("free sulfur dioxide", 0.0)
    if free_so2 < 12.0:
        alerts.append(f"[WARNING] Low Free SO2 ({free_so2:.1f} mg/dm3): Insufficient antioxidant protection; oxidation risk.")
    elif free_so2 > 40.0:
        alerts.append(f"[NOTICE] Elevated Free SO2 ({free_so2:.1f} mg/dm3): Potential masking of delicate fruit bouquet.")

    # Check alcohol body and warmth
    alc = raw.get("alcohol", 0.0)
    if alc >= 11.5:
        alerts.append(f"[OPTIMAL] Generous Ethanol ({alc:.1f}% vol): Desirable phenolic extraction, rich body, warm finish.")
    elif alc < 9.5:
        alerts.append(f"[NOTICE] Light Body ({alc:.1f}% vol): Palate may lack warmth; suitable for volume blending.")

    # Check sulphates
    sulphates = raw.get("sulphates", 0.0)
    if sulphates >= 0.70:
        alerts.append(f"[OPTIMAL] Robust Sulphates ({sulphates:.2f} g/dm3): High antioxidant and antimicrobial protection.")
    elif sulphates < 0.50:
        alerts.append(f"[WARNING] Low Sulphates ({sulphates:.2f} g/dm3): Reduced preservative longevity during cellaring.")

    # Chlorides / Salinity
    chlorides = raw.get("chlorides", 0.0)
    if chlorides > 0.12:
        alerts.append(f"[WARNING] Elevated Chlorides ({chlorides:.3f} g/dm3): Potential mineral brackishness or salinity.")

    # Determine sensory tier
    if prob_good >= 0.75:
        tier = "Grand Reserve"
        guidance = "Exceptional enological balance. Top candidate for premium oak barrel maturation and reserve labeling."
    elif prob_good >= RECOMMENDED_SCREENING_THRESHOLD:
        tier = "Reserve Grade"
        guidance = "Solid candidate for reserve or vintage bottling. Passed industrial screening threshold (tau=0.35)."
    elif prob_good >= 0.20:
        tier = "Table Standard"
        guidance = "Standard commercial quality. Suitable for immediate table wine bottling, early release, or volume blending."
    else:
        tier = "Commercial Bulk"
        guidance = "Low reserve probability. Recommended for secondary blending, dealcoholization, or value commercial lines."

    return tier, guidance, alerts


# ============================================================================
# WineQualityInferenceService
# ============================================================================

class WineQualityInferenceService:
    """
    Production-grade inference service encapsulating the champion Random Forest pipeline,
    metadata provenance, feature engineering, and high-performance prediction methods.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
        default_threshold: float = RECOMMENDED_SCREENING_THRESHOLD
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.default_threshold = float(default_threshold)
        self.pipeline: Optional[Pipeline] = None
        self.white_pipeline: Optional[Any] = None
        self.cultivar_detector: Optional[Any] = None
        self.recourse_optimizer: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.startup_time = time.time()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model and JSON metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path.resolve()}")

        self.pipeline = joblib.load(self.model_path)

        white_model_path = Path("models/white_champion_rf_pipeline.joblib")
        if white_model_path.exists():
            try:
                self.white_pipeline = joblib.load(white_model_path)
            except Exception:
                self.white_pipeline = None

        cultivar_model_path = Path("models/cultivar_classifier.joblib")
        if cultivar_model_path.exists():
            try:
                from src.cross_cultivar_and_recourse import CultivarDetector, CellarRecourseOptimizer
                self.cultivar_detector = CultivarDetector(cultivar_model_path)
                self.recourse_optimizer = CellarRecourseOptimizer()
            except Exception:
                self.cultivar_detector = None
                self.recourse_optimizer = None

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "model_name": "Tuned Random Forest Classifier",
                "project": "Wine Quality Prediction",
                "operating_screening_threshold": self.default_threshold
            }

    @property
    def is_ready(self) -> bool:
        """Check whether the model pipeline is loaded and operational."""
        return self.pipeline is not None

    def predict_single(
        self,
        wine_input: WinePhysicochemicalInput,
        threshold: Optional[float] = None
    ) -> PredictionResult:
        """
        Execute real-time prediction for a single wine sample.
        """
        if not self.is_ready:
            raise RuntimeError("Model pipeline is not loaded.")

        active_thresh = float(threshold if threshold is not None else self.default_threshold)
        raw_dict = wine_input.to_raw_dict()
        full_dict = derive_interaction_features(raw_dict)

        # Cultivar resolution
        target_cult = (wine_input.cultivar or "auto").lower()
        if target_cult == "auto":
            if self.cultivar_detector is not None:
                c_res = self.cultivar_detector.predict_cultivar(raw_dict)
                resolved_cultivar = c_res["cultivar"]
                c_conf = c_res["confidence"]
            else:
                resolved_cultivar = "red"
                c_conf = 1.0
        else:
            resolved_cultivar = target_cult
            c_conf = 1.0

        # Select model pipeline
        active_pipe = self.white_pipeline if (resolved_cultivar == "white" and self.white_pipeline is not None) else self.pipeline

        # Build single-row DataFrame strictly ordered
        ordered_data = {col: [full_dict[col]] for col in FULL_FEATURE_ORDER}
        df_input = pd.DataFrame(ordered_data)

        # Predict probabilities
        probabilities = active_pipe.predict_proba(df_input)[0]
        prob_normal = float(probabilities[0])
        prob_good = float(probabilities[1])

        is_premium = prob_good >= active_thresh
        tier, guidance, alerts = evaluate_enological_rules(raw_dict, prob_good)

        # Extract only the 5 engineered interaction features for response
        eng_features = {
            "total_acidity": full_dict["total_acidity"],
            "bound_sulfur_dioxide": full_dict["bound_sulfur_dioxide"],
            "free_to_total_so2_ratio": full_dict["free_to_total_so2_ratio"],
            "alcohol_to_density_ratio": full_dict["alcohol_to_density_ratio"],
            "sulphate_to_chloride_ratio": full_dict["sulphate_to_chloride_ratio"]
        }

        confidence = abs(prob_good - active_thresh)

        return PredictionResult(
            sample_id=wine_input.sample_id,
            cultivar=resolved_cultivar,
            cultivar_confidence=round(c_conf, 4),
            is_premium_wine=is_premium,
            prediction_label="Good / Premium Wine (Score >= 7)" if is_premium else "Normal Wine (Score < 7)",
            sommelier_grade=tier,
            probability_good=round(prob_good, 4),
            probability_normal=round(prob_normal, 4),
            operating_threshold=active_thresh,
            decision_confidence=round(confidence, 4),
            decision_guidance=guidance,
            enological_alerts=alerts,
            engineered_features={k: round(v, 4) for k, v in eng_features.items()}
        )

    def predict_batch(
        self,
        samples: List[WinePhysicochemicalInput],
        threshold: Optional[float] = None
    ) -> BatchPredictionResponse:
        """
        Execute vectorized batch inference across multiple wine specimens.
        """
        if not self.is_ready:
            raise RuntimeError("Model pipeline is not loaded.")

        t0 = time.perf_counter()
        active_thresh = float(threshold if threshold is not None else self.default_threshold)

        if not samples:
            return BatchPredictionResponse(
                total_samples=0,
                premium_samples=0,
                standard_samples=0,
                premium_rate_pct=0.0,
                operating_threshold=active_thresh,
                mean_probability_good=0.0,
                batch_latency_ms=0.0,
                predictions=[]
            )

        # Vectorized dataframe construction
        raw_dicts = [s.to_raw_dict() for s in samples]
        full_dicts = [derive_interaction_features(d) for d in raw_dicts]

        df_batch = pd.DataFrame(full_dicts)[FULL_FEATURE_ORDER]

        # Vectorized batch prediction
        probs = self.pipeline.predict_proba(df_batch)
        probs_good = probs[:, 1]
        probs_normal = probs[:, 0]

        predictions: List[PredictionResult] = []
        premium_count = 0

        for i, sample in enumerate(samples):
            prob_g = float(probs_good[i])
            prob_n = float(probs_normal[i])
            is_prem = prob_g >= active_thresh
            if is_prem:
                premium_count += 1

            tier, guidance, alerts = evaluate_enological_rules(raw_dicts[i], prob_g)
            confidence = abs(prob_g - active_thresh)

            eng_features = {
                "total_acidity": full_dicts[i]["total_acidity"],
                "bound_sulfur_dioxide": full_dicts[i]["bound_sulfur_dioxide"],
                "free_to_total_so2_ratio": full_dicts[i]["free_to_total_so2_ratio"],
                "alcohol_to_density_ratio": full_dicts[i]["alcohol_to_density_ratio"],
                "sulphate_to_chloride_ratio": full_dicts[i]["sulphate_to_chloride_ratio"]
            }

            predictions.append(
                PredictionResult(
                    sample_id=sample.sample_id,
                    is_premium_wine=is_prem,
                    prediction_label="Good / Premium Wine (Score >= 7)" if is_prem else "Normal Wine (Score < 7)",
                    sommelier_grade=tier,
                    probability_good=round(prob_g, 4),
                    probability_normal=round(prob_n, 4),
                    operating_threshold=active_thresh,
                    decision_confidence=round(confidence, 4),
                    decision_guidance=guidance,
                    enological_alerts=alerts,
                    engineered_features={k: round(v, 4) for k, v in eng_features.items()}
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        total = len(samples)
        std_count = total - premium_count
        premium_rate = (premium_count / total * 100.0) if total > 0 else 0.0
        mean_prob = float(np.mean(probs_good)) if total > 0 else 0.0

        return BatchPredictionResponse(
            total_samples=total,
            premium_samples=premium_count,
            standard_samples=std_count,
            premium_rate_pct=round(premium_rate, 2),
            operating_threshold=active_thresh,
            mean_probability_good=round(mean_prob, 4),
            batch_latency_ms=round(latency_ms, 2),
            predictions=predictions
        )


# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_app(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    default_threshold: float = RECOMMENDED_SCREENING_THRESHOLD
) -> FastAPI:
    """
    Create and configure the production FastAPI application instance.
    """
    app = FastAPI(
        title="Wine Quality Prediction REST API",
        description=(
            "Production-grade Machine Learning inference service serving Cortez et al. "
            "Wine Quality Classification. Powered by a hyperparameter-tuned Random Forest "
            "ensemble with automated chemical interaction feature engineering and sensory grading."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Enable CORS for browser frontends and web dashboards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize inference service
    service = WineQualityInferenceService(
        model_path=model_path,
        metadata_path=metadata_path,
        default_threshold=default_threshold
    )

    @app.get("/", tags=["General"])
    def root():
        """Welcome landing endpoint with API documentation links and service metadata."""
        return {
            "title": "Wine Quality Prediction Production API",
            "status": "online",
            "model_ready": service.is_ready,
            "version": "1.0.0",
            "documentation_urls": {
                "swagger_ui": "/docs",
                "redoc": "/redoc"
            },
            "endpoints": {
                "health": "/health",
                "metadata": "/metadata",
                "rules": "/rules",
                "predict_single": "/predict",
                "predict_batch": "/predict/batch"
            }
        }

    @app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    def health_check():
        """System health and model readiness probe."""
        uptime = time.time() - service.startup_time
        return HealthResponse(
            status="healthy" if service.is_ready else "degraded",
            model_loaded=service.is_ready,
            model_name=service.metadata.get("model_name", "Tuned Random Forest"),
            model_path=str(service.model_path),
            uptime_seconds=round(uptime, 2),
            server_time=datetime.utcnow().isoformat(),
            api_version="1.0.0"
        )

    @app.get("/metadata", response_model=MetadataResponse, tags=["Diagnostics"])
    def get_model_metadata():
        """Expose champion model lineage, hyperparameters, and cross-validation metrics."""
        meta = service.metadata
        return MetadataResponse(
            model_name=meta.get("model_name", "Tuned Random Forest Classifier"),
            project=meta.get("project", "Wine Quality Prediction"),
            date_finalized=meta.get("date_finalized", "2026-10-08"),
            n_estimators=int(meta.get("n_estimators", 200)),
            max_depth=int(meta["max_depth"]) if meta.get("max_depth") is not None else None,
            class_weight=str(meta.get("class_weight", "balanced")),
            features=meta.get("features", FULL_FEATURE_ORDER),
            num_features=int(meta.get("num_features", len(FULL_FEATURE_ORDER))),
            test_roc_auc=float(meta.get("test_roc_auc", 0.9368)),
            test_pr_auc=float(meta.get("test_pr_auc", 0.8095)),
            test_balanced_accuracy_screen_threshold=float(
                meta.get("test_balanced_accuracy_screen_threshold", 0.852)
            ),
            operating_screening_threshold=float(
                meta.get("operating_screening_threshold", service.default_threshold)
            ),
            loaded_artifact_path=str(service.model_path.resolve())
        )

    @app.get("/rules", tags=["Enology"])
    def get_enological_rules():
        """Retrieve sensory chemistry decision thresholds and operational cellar targets."""
        return {
            "description": "Empirical chemical benchmarks contrasting Normal (< 7) vs Good (>= 7) wines",
            "thresholds": {
                "alcohol": {
                    "good_median": 11.5,
                    "target": ">= 11.5 % vol",
                    "impact": "Enhances body, perceived warmth, and bouquet release"
                },
                "volatile_acidity": {
                    "good_median": 0.37,
                    "target": "<= 0.45 g/dm³",
                    "impact": "Suppresses pungent acetic acid vinegar notes"
                },
                "sulphates": {
                    "good_median": 0.74,
                    "target": ">= 0.70 g/dm³",
                    "impact": "Imparts antioxidant capacity and microbiological stability"
                },
                "chlorides": {
                    "good_median": 0.073,
                    "target": "<= 0.075 g/dm³",
                    "impact": "Prevents mineral brackishness and harsh salinity"
                },
                "total_acidity": {
                    "good_median": 9.20,
                    "target": "8.5 - 10.0 g/dm³",
                    "impact": "Provides structural crispness and natural preservation"
                }
            },
            "operating_screening_threshold": service.default_threshold
        }

    @app.post(
        "/predict",
        response_model=PredictionResult,
        status_code=status.HTTP_200_OK,
        tags=["Inference"]
    )
    def predict_single_sample(
        wine: WinePhysicochemicalInput,
        threshold: Optional[float] = Query(
            default=None,
            ge=0.05,
            le=0.95,
            description="Custom probability decision cutoff override"
        )
    ):
        """
        Real-time single-sample prediction.
        Synthesizes 5 domain chemical interaction features and scores the specimen
        against the champion Random Forest model.
        """
        try:
            return service.predict_single(wine, threshold=threshold)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Inference"]
    )
    def predict_sample_batch(batch_request: BatchPredictionRequest):
        """
        High-throughput vectorized inference for laboratory batches.
        Processes up to thousands of wine samples concurrently with sub-millisecond per-sample latency.
        """
        try:
            return service.predict_batch(
                batch_request.samples,
                threshold=batch_request.threshold
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/metrics/drift",
        response_model=DriftMetricsResponse,
        tags=["Monitoring"]
    )
    def get_drift_monitoring_metrics():
        """
        Baseline statistical monitoring parameters and PSI alerting thresholds.
        """
        from src.drift_monitoring import PSI_THRESHOLD_GREEN, PSI_THRESHOLD_AMBER, RAW_PHYSICOCHEMICAL_FEATURES
        return DriftMetricsResponse(
            status="ACTIVE",
            baseline_dataset_size=1599,
            baseline_features=RAW_PHYSICOCHEMICAL_FEATURES,
            psi_thresholds={
                "green_max": PSI_THRESHOLD_GREEN,
                "amber_max": PSI_THRESHOLD_AMBER,
                "red_min": PSI_THRESHOLD_AMBER
            },
            monitoring_mode="QUANTILE_PSI_AND_KS_2SAMP",
            server_time=datetime.utcnow().isoformat()
        )

    @app.post(
        "/audit/evaluate",
        response_model=DriftAuditResponse,
        tags=["Monitoring"]
    )
    def evaluate_incoming_batch_drift(batch_request: BatchPredictionRequest):
        """
        Evaluate covariate shift and statistical drift on an incoming laboratory batch
        relative to the baseline reference dataset.
        """
        from src.data_loader import load_wine_data
        from src.drift_monitoring import compute_dataset_drift_summary, RAW_PHYSICOCHEMICAL_FEATURES

        try:
            samples_dicts = [s.to_raw_dict() for s in batch_request.samples]
            batch_df = pd.DataFrame(samples_dicts)

            ref_df = load_wine_data("red")
            drift_summary = compute_dataset_drift_summary(ref_df, batch_df, feature_cols=RAW_PHYSICOCHEMICAL_FEATURES)

            red_count = int((drift_summary["Alert_Level"] == "RED").sum())
            amber_count = int((drift_summary["Alert_Level"] == "AMBER").sum())
            green_count = int((drift_summary["Alert_Level"] == "GREEN").sum())

            critical_levers = ["alcohol", "volatile acidity", "sulphates"]
            critical_reds = [
                row["Feature"] for _, row in drift_summary.iterrows()
                if row["Alert_Level"] == "RED" and row["Feature"] in critical_levers
            ]

            reasons: List[str] = []
            if red_count >= 2:
                reasons.append(f"{red_count} features exceeded Red PSI threshold (>= 0.25)")
            if critical_reds:
                reasons.append(f"Critical enological quality driver(s) severely drifted: {critical_reds}")

            retraining_rec = (red_count >= 2) or bool(critical_reds)
            overall_status = "RED" if red_count > 0 else ("AMBER" if amber_count > 0 else "GREEN")

            scorecard = [
                FeatureDriftItem(
                    feature=row["Feature"],
                    psi=float(row["PSI"]),
                    ks_statistic=float(row["KS_Statistic"]),
                    ks_p_value=float(row["KS_P_Value"]),
                    wasserstein_distance=float(row["Wasserstein_Dist"]),
                    drift_status=row["Drift_Status"],
                    alert_level=row["Alert_Level"]
                )
                for _, row in drift_summary.iterrows()
            ]

            return DriftAuditResponse(
                total_features_evaluated=len(scorecard),
                features_red_alert=red_count,
                features_amber_alert=amber_count,
                features_green=green_count,
                overall_drift_status=overall_status,
                retraining_recommended=retraining_rec,
                triggered_reasons=reasons,
                feature_drift_scorecard=scorecard,
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/cultivar/detect",
        response_model=CultivarDetectionResponse,
        tags=["Cultivar & Recourse"]
    )
    def detect_wine_cultivar(wine_input: WinePhysicochemicalInput):
        """
        Classify whether an incoming wine specimen is Red or White cultivar
        based on raw physicochemical properties.
        """
        try:
            from src.cross_cultivar_and_recourse import CultivarDetector
            detector = service.cultivar_detector or CultivarDetector()
            res = detector.predict_cultivar(wine_input.to_raw_dict())
            return CultivarDetectionResponse(
                cultivar=res["cultivar"],
                confidence=res["confidence"],
                prob_red=res["prob_red"],
                prob_white=res["prob_white"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/predict/recourse",
        response_model=CellarRecourseResponse,
        tags=["Cultivar & Recourse"]
    )
    def get_prescriptive_cellar_recourse(
        wine_input: WinePhysicochemicalInput,
        target_prob: float = Query(default=0.45, ge=0.20, le=0.90, description="Target probability for reserve upgrade"),
        operating_threshold: float = Query(default=0.35, ge=0.10, le=0.90, description="Screening decision threshold")
    ):
        """
        Compute minimal actionable enological interventions to elevate a sub-premium
        wine lot to Reserve Grade status.
        """
        try:
            from src.cross_cultivar_and_recourse import CellarRecourseOptimizer, CultivarDetector
            raw_dict = wine_input.to_raw_dict()
            cult = (wine_input.cultivar or "auto").lower()
            if cult == "auto":
                detector = service.cultivar_detector or CultivarDetector()
                cult = detector.predict_cultivar(raw_dict)["cultivar"]

            optimizer = service.recourse_optimizer or CellarRecourseOptimizer()
            recourse_dict = optimizer.optimize_recourse(
                sample=raw_dict,
                cultivar=cult,
                target_prob=target_prob,
                operating_threshold=operating_threshold
            )

            interventions = [
                RecourseInterventionItem(**item)
                for item in recourse_dict.get("interventions", [])
            ]

            return CellarRecourseResponse(
                status=recourse_dict["status"],
                cultivar=recourse_dict["cultivar"],
                initial_prob=recourse_dict["initial_prob"],
                target_prob=recourse_dict["target_prob"],
                projected_prob=recourse_dict["projected_prob"],
                probability_gain=recourse_dict["probability_gain"],
                operating_threshold=recourse_dict["operating_threshold"],
                upgraded_to_premium=recourse_dict["upgraded_to_premium"],
                feasibility=recourse_dict["feasibility"],
                interventions=interventions,
                adjusted_sample=recourse_dict.get("adjusted_sample", raw_dict)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/metrics/cultivars",
        tags=["Cultivar & Recourse"]
    )
    def get_cultivar_divergence_metrics():
        """
        Return statistical divergence metrics and covariate shift between Red and White datasets.
        """
        try:
            from src.cross_cultivar_and_recourse import CultivarDomainAnalyzer
            analyzer = CultivarDomainAnalyzer()
            div_df = analyzer.compute_cross_cultivar_divergence()
            frag = analyzer.evaluate_cross_domain_fragility()
            return {
                "cross_cultivar_divergence": div_df.to_dict(orient="records"),
                "zero_shot_domain_gap": frag,
                "server_time": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/blend/optimize",
        response_model=AssemblageOptimizationResponse,
        tags=["Assemblage & Coupage"]
    )
    def optimize_wine_coupage(request: AssemblageOptimizationRequest):
        """
        Optimize blending proportions for multi-tank wine assemblage under enological constraints.
        """
        try:
            from src.assemblage_and_conformal import WineAssemblageOptimizer
            lots_raw = [lot.to_raw_dict() for lot in request.lots]
            names = [lot.lot_name or f"Lot_{i+1}" for i, lot in enumerate(request.lots)]
            costs = [lot.cost_per_liter or 0.0 for lot in request.lots]

            constraints_dict = None
            if request.constraints:
                constraints_dict = {
                    k: v for k, v in request.constraints.model_dump().items() if v is not None
                }

            optimizer = WineAssemblageOptimizer(model=service.pipeline)
            res = optimizer.optimize_blend(
                lots=lots_raw,
                lot_names=names,
                constraints=constraints_dict,
                lot_costs=costs,
                cost_penalty_weight=request.cost_penalty_weight
            )
            return AssemblageOptimizationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/predict/conformal",
        response_model=ConformalPredictionResponse,
        tags=["Conformal & Uncertainty"]
    )
    def predict_with_conformal_uncertainty(
        wine_input: WinePhysicochemicalInput,
        alpha: float = Query(default=0.10, ge=0.01, le=0.50, description="Significance level (error rate)")
    ):
        """
        Predict wine quality with distribution-free conformal prediction sets and sommelier guidance.
        """
        try:
            from src.assemblage_and_conformal import EnologicalConformalPredictor
            conformal_path = Path("models/conformal_calibrator.json")
            conformal = EnologicalConformalPredictor(model=service.pipeline)
            if conformal_path.exists():
                conformal.load(conformal_path)
            else:
                from src.data_loader import load_wine_data
                from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
                df = load_wine_data("red")
                df_tar = create_binned_targets(df)
                df_f = create_engineered_features(df_tar)
                X_tr, X_cal, y_tr, y_cal = get_stratified_train_test_split(df_f, df_f["binary_quality"], test_size=0.25, random_state=42)
                conformal.fit_calibration(X_cal, y_cal)

            raw_dict = wine_input.to_raw_dict()
            res = conformal.predict_conformal_set(raw_dict, alpha=alpha)
            return ConformalPredictionResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/economics/evaluate",
        response_model=EconomicEvaluationResponse,
        tags=["Winery Economics"]
    )
    def evaluate_winery_economics(request: EconomicEvaluationRequest):
        """
        Evaluate commercial winery profit matrix and identify financially optimal operating threshold.
        """
        try:
            from src.assemblage_and_conformal import WineryEconomicEngine
            from src.data_loader import load_wine_data
            from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
            df = load_wine_data("red")
            df_tar = create_binned_targets(df)
            df_f = create_engineered_features(df_tar)
            X = df_f[FULL_FEATURE_ORDER]
            y = df_f["binary_quality"]
            _, X_test, _, y_test = get_stratified_train_test_split(X, y, test_size=0.20, random_state=42)
            probs = service.pipeline.predict_proba(X_test)[:, 1]

            econ = WineryEconomicEngine(
                tp_margin=request.tp_margin,
                fp_penalty=request.fp_penalty,
                fn_penalty=request.fn_penalty,
                tn_margin=request.tn_margin
            )
            res = econ.optimize_financial_threshold(
                y_true=y_test.values,
                y_prob=probs,
                lot_size_bottles=request.lot_size_bottles
            )
            return EconomicEvaluationResponse(
                optimal_financial_threshold=res["optimal_financial_threshold"],
                max_harvest_profit=res["max_harvest_profit"],
                max_profit_per_bottle=res["max_profit_per_bottle"],
                default_threshold_profit=res["default_threshold_evaluation"]["harvest_lot_profit"],
                f1_threshold_profit=res["f1_threshold_evaluation"]["harvest_lot_profit"],
                dollar_uplift_vs_default=res["dollar_uplift_vs_default"],
                percent_uplift_vs_default=res["percent_uplift_vs_default"],
                best_evaluation=res["best_evaluation"],
                default_threshold_evaluation=res["default_threshold_evaluation"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/metrics/stacking",
        response_model=StackingBenchmarkResponse,
        tags=["Stacking Ensemble"]
    )
    def get_stacking_ensemble_metrics():
        """
        Return benchmark comparison metrics for multi-model stacking meta-learner.
        """
        try:
            stack_json = Path("models/stacking_ensemble.json")
            if not stack_json.exists():
                from src.assemblage_and_conformal import WineStackingEnsemble
                from src.data_loader import load_wine_data
                from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
                df = load_wine_data("red")
                df_tar = create_binned_targets(df)
                df_f = create_engineered_features(df_tar)
                X = df_f[FULL_FEATURE_ORDER]
                y = df_f["binary_quality"]
                X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.20, random_state=42)
                X_tr, _, y_tr, _ = get_stratified_train_test_split(X_train, y_train, test_size=0.25, random_state=42)
                stacker = WineStackingEnsemble(random_state=42)
                stacker.fit(X_tr, y_tr)
                stacker.save("models/stacking_ensemble.joblib", stack_json)
                eval_metrics = stacker.evaluate(X_test, y_test)
            else:
                from src.assemblage_and_conformal import WineStackingEnsemble
                from src.data_loader import load_wine_data
                from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
                stacker = WineStackingEnsemble.load("models/stacking_ensemble.joblib")
                df = load_wine_data("red")
                df_tar = create_binned_targets(df)
                df_f = create_engineered_features(df_tar)
                X = df_f[FULL_FEATURE_ORDER]
                y = df_f["binary_quality"]
                _, X_test, _, y_test = get_stratified_train_test_split(X, y, test_size=0.20, random_state=42)
                eval_metrics = stacker.evaluate(X_test, y_test)

            return StackingBenchmarkResponse(
                model_name="Heterogeneous Stacking Meta-Learner (Day 10)",
                stacking_roc_auc=eval_metrics["stacking_roc_auc"],
                stacking_pr_auc=eval_metrics["stacking_pr_auc"],
                stacking_brier_score=eval_metrics["stacking_brier_score"],
                stacking_balanced_accuracy=eval_metrics["stacking_balanced_accuracy"],
                operating_threshold=eval_metrics["operating_threshold"],
                meta_learner_coefficients=eval_metrics["meta_learner_coefficients"],
                base_models_benchmarks=eval_metrics["base_models_benchmarks"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Day 11 Endpoints: Aging Kinetics, Pareto Optimization, Stress Testing, SHAP
    # ========================================================================

    @app.post(
        "/aging/simulate",
        response_model=AgingSimulationResponse,
        summary="Simulate temporal bottle cellar aging and shelf-life kinetics",
        tags=["Enological Dynamics (Day 11)"]
    )
    def simulate_bottle_aging(req: AgingSimulationRequest):
        try:
            from src.aging_pareto_stress import CellarAgingKineticsSimulator
            sim = CellarAgingKineticsSimulator(model_pipeline=service.pipeline)
            raw_dict = req.sample.model_dump(by_alias=False)
            chem = {
                "fixed acidity": raw_dict["fixed_acidity"],
                "volatile acidity": raw_dict["volatile_acidity"],
                "citric acid": raw_dict["citric_acid"],
                "residual sugar": raw_dict["residual_sugar"],
                "chlorides": raw_dict["chlorides"],
                "free sulfur dioxide": raw_dict["free_sulfur_dioxide"],
                "total sulfur dioxide": raw_dict["total_sulfur_dioxide"],
                "density": raw_dict["density"],
                "pH": raw_dict.get("pH", raw_dict.get("ph")),
                "sulphates": raw_dict["sulphates"],
                "alcohol": raw_dict["alcohol"]
            }
            res = sim.simulate_trajectory(
                chem,
                months=req.months,
                temperature_c=req.temperature_c,
                closure=req.closure_type,
                step_months=req.step_months
            )
            return AgingSimulationResponse(
                cellar_temperature_c=res["cellar_temperature_c"],
                closure_type=res["closure_type"],
                so2_decay_rate_k=res["so2_decay_rate_k"],
                simulation_months=res["simulation_months"],
                initial_probability=res["initial_probability"],
                peak_probability=res["peak_probability"],
                peak_month=res["peak_month"],
                drinking_window_start_month=res["drinking_window_start_month"],
                drinking_window_end_month=res["drinking_window_end_month"],
                spoilage_risk_month=res["spoilage_risk_month"],
                sommelier_cellar_advice=res["sommelier_cellar_advice"],
                trajectory_preview=res["trajectory"][:12]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/pareto/optimize",
        response_model=ParetoOptimizationResponse,
        summary="Compute non-dominated Pareto trade-off frontier balancing quality and preservatives",
        tags=["Multi-Objective Optimization (Day 11)"]
    )
    def optimize_pareto_frontier(req: ParetoOptimizationRequest):
        try:
            from src.aging_pareto_stress import ParetoEnologicalOptimizer
            opt = ParetoEnologicalOptimizer(model_pipeline=service.pipeline)
            candidates = []
            for item in req.candidates:
                d = item.model_dump(by_alias=False)
                candidates.append({
                    "name": item.name or "Candidate Lot",
                    "fixed acidity": d["fixed_acidity"],
                    "volatile acidity": d["volatile_acidity"],
                    "citric acid": d["citric_acid"],
                    "residual sugar": d["residual_sugar"],
                    "chlorides": d["chlorides"],
                    "free sulfur dioxide": d["free_sulfur_dioxide"],
                    "total sulfur dioxide": d["total_sulfur_dioxide"],
                    "density": d["density"],
                    "pH": d.get("pH", d.get("ph")),
                    "sulphates": d["sulphates"],
                    "alcohol": d["alcohol"]
                })
            res = opt.compute_pareto_frontier(candidates)
            return ParetoOptimizationResponse(
                total_candidates_evaluated=res["total_candidates_evaluated"],
                pareto_optimal_count=res["pareto_optimal_count"],
                pareto_frontier=res["pareto_frontier"],
                knee_point_lot=res["knee_point_lot"],
                archetypes=res["archetypes"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/stress/test",
        response_model=StressTestResponse,
        summary="Audit physicochemical perturbation stability and adversarial flip margin",
        tags=["Adversarial Robustness (Day 11)"]
    )
    def audit_stress_testing(req: StressTestRequest):
        try:
            from src.aging_pareto_stress import EnologicalStressTester
            tester = EnologicalStressTester(model_pipeline=service.pipeline)
            raw_dict = req.sample.model_dump(by_alias=False)
            chem = {
                "fixed acidity": raw_dict["fixed_acidity"],
                "volatile acidity": raw_dict["volatile_acidity"],
                "citric acid": raw_dict["citric_acid"],
                "residual sugar": raw_dict["residual_sugar"],
                "chlorides": raw_dict["chlorides"],
                "free sulfur dioxide": raw_dict["free_sulfur_dioxide"],
                "total sulfur dioxide": raw_dict["total_sulfur_dioxide"],
                "density": raw_dict["density"],
                "pH": raw_dict.get("pH", raw_dict.get("ph")),
                "sulphates": raw_dict["sulphates"],
                "alcohol": raw_dict["alcohol"]
            }
            res = tester.run_stress_test(chem)
            scenarios = [StressScenarioResult(**s) for s in res["scenario_results"]]
            return StressTestResponse(
                baseline_probability=res["baseline_probability"],
                baseline_is_premium=res["baseline_is_premium"],
                decision_threshold=res["decision_threshold"],
                scenarios_tested=res["scenarios_tested"],
                classification_flips=res["classification_flips"],
                resilience_score=res["resilience_score"],
                resilience_tier=res["resilience_tier"],
                critical_va_epsilon_flip_margin=res["critical_va_epsilon_flip_margin"],
                scenario_results=scenarios
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/metrics/shap-interactions",
        response_model=ShapInteractionsResponse,
        summary="Retrieve chemical interaction synergies and molecular SO2 equilibrium metrics",
        tags=["Explainability & Synergy (Day 11)"]
    )
    def get_shap_interactions():
        try:
            p = Path("models/enological_interactions.json")
            if p.exists():
                with open(p, "r") as f:
                    data = json.load(f)
            else:
                from src.aging_pareto_stress import ShapEnologicalInteractions
                from src.data_loader import load_wine_data
                df = load_wine_data("red")
                data = ShapEnologicalInteractions.compute_interaction_matrix(service.model_pipeline, df)
            return ShapInteractionsResponse(interactions=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Day 12 Endpoints: Fermentation Kinetics, Stuck Early Warning & Sensory
    # ========================================================================

    @app.post(
        "/fermentation/simulate",
        response_model=FermentationSimulationResponse,
        summary="Simulate primary alcoholic fermentation bioprocess kinetics",
        tags=["Fermentation & Bioprocess (Day 12)"]
    )
    def simulate_fermentation_endpoint(req: FermentationSimulationRequest):
        try:
            from src.fermentation_and_sensory import FermentationKineticsSimulator
            sim = FermentationKineticsSimulator(model_pipeline=service.pipeline)
            res = sim.simulate_fermentation(
                initial_sugar_g_l=req.initial_sugar_g_l,
                initial_yan_mg_l=req.initial_yan_mg_l,
                initial_temp_c=req.initial_temp_c,
                jacket_temp_c=req.jacket_temp_c,
                initial_biomass_g_l=req.initial_biomass_g_l,
                total_days=req.total_days,
                yeast_strain=req.yeast_strain
            )
            return FermentationSimulationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/fermentation/diagnose-stuck",
        response_model=StuckDiagnosisResponse,
        summary="Evaluate sequential fermentation readings for sluggish/stuck risks",
        tags=["Fermentation & Bioprocess (Day 12)"]
    )
    def diagnose_stuck_endpoint(req: StuckDiagnosisRequest):
        try:
            from src.fermentation_and_sensory import StuckFermentationDetector
            detector = StuckFermentationDetector()
            readings_raw = [r.model_dump() for r in req.readings]
            res = detector.diagnose_trajectory(readings_raw)
            return StuckDiagnosisResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/sensory/profile",
        response_model=SensoryProfileResponse,
        summary="Translate laboratory chemistry into 6-axis organoleptic radar and sommelier notes",
        tags=["Sensory & Organoleptic (Day 12)"]
    )
    def sensory_profile_endpoint(req: SensoryProfileRequest):
        try:
            from src.fermentation_and_sensory import SensoryOrganolepticEngine
            raw_dict = req.sample.model_dump(by_alias=False)
            chem = {
                "fixed acidity": raw_dict["fixed_acidity"],
                "volatile acidity": raw_dict["volatile_acidity"],
                "citric acid": raw_dict["citric_acid"],
                "residual sugar": raw_dict["residual_sugar"],
                "chlorides": raw_dict["chlorides"],
                "free sulfur dioxide": raw_dict["free_sulfur_dioxide"],
                "total sulfur dioxide": raw_dict["total_sulfur_dioxide"],
                "density": raw_dict["density"],
                "pH": raw_dict.get("pH", raw_dict.get("ph")),
                "sulphates": raw_dict["sulphates"],
                "alcohol": raw_dict["alcohol"]
            }
            res = SensoryOrganolepticEngine.compute_sensory_profile(chem)
            return SensoryProfileResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/metrics/causal-effects",
        response_model=CausalMetricsResponse,
        summary="Retrieve Doubly Robust IPW causal Average Treatment Effects on wine quality",
        tags=["Causal Inference (Day 12)"]
    )
    def get_causal_effects_endpoint():
        try:
            p = Path("models/causal_intervention_benchmarks.json")
            if p.exists():
                with open(p, "r") as f:
                    data = json.load(f)
            else:
                from src.fermentation_and_sensory import CausalEnologicalEngine
                from src.data_loader import load_wine_data
                df = load_wine_data("red")
                data = {}
                for k in CausalEnologicalEngine.INTERVENTIONS:
                    data[k] = CausalEnologicalEngine.estimate_causal_effect(df, intervention_key=k, n_bootstraps=50)
            return CausalMetricsResponse(interventions=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Day 13: Malolactic, Oak Maturation, Cellar Dispatch & HACCP Endpoints
    # ========================================================================

    @app.post(
        "/cellar/malolactic/simulate",
        response_model=MalolacticSimulationResponse,
        summary="Simulate secondary malolactic fermentation biotransformation by Oenococcus oeni",
        tags=["Malolactic & Oak Maturation (Day 13)"]
    )
    def simulate_malolactic_endpoint(req: MalolacticSimulationRequest):
        try:
            from src.malolactic_and_oak import MalolacticFermentationSimulator
            sim = MalolacticFermentationSimulator()
            res = sim.simulate(
                initial_malic_acid=req.initial_malic_acid,
                initial_ph=req.initial_ph,
                initial_fixed_acidity=req.initial_fixed_acidity,
                alcohol=req.alcohol,
                free_so2=req.free_so2,
                temperature=req.temperature,
                inoculated_biomass=req.inoculated_biomass,
                duration_days=req.duration_days,
            )
            return MalolacticSimulationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/cellar/oak-maturation/simulate",
        response_model=OakMaturationResponse,
        summary="Simulate oak wood phenolics extraction and tannin polymerization",
        tags=["Malolactic & Oak Maturation (Day 13)"]
    )
    def simulate_oak_maturation_endpoint(req: OakMaturationRequest):
        try:
            from src.malolactic_and_oak import OakBarrelMaturationSimulator
            sim = OakBarrelMaturationSimulator()
            res = sim.simulate(
                oak_species=req.oak_species,
                toast_level=req.toast_level,
                barrel_usage=req.barrel_usage,
                maturation_months=req.maturation_months,
                barrel_volume_liters=req.barrel_volume_liters,
            )
            return OakMaturationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/cellar/dispatch/work-orders",
        response_model=CellarDispatchResponse,
        summary="Triage cellar vessels by ERPI risk and generate prescriptive chemical work orders",
        tags=["Cellar Master Dispatch & HACCP (Day 13)"]
    )
    def dispatch_cellar_work_orders_endpoint(req: CellarDispatchRequest):
        try:
            from src.malolactic_and_oak import CellarMasterDispatchEngine
            engine = CellarMasterDispatchEngine()
            inventory_raw = [v.model_dump() for v in req.inventory]
            res = engine.dispatch_cellar_inventory(inventory_raw)
            return CellarDispatchResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/compliance/haccp-audit",
        response_model=HaccpAuditResponse,
        summary="Perform statutory HACCP Critical Control Point audit and biogenic amine risk scoring",
        tags=["Cellar Master Dispatch & HACCP (Day 13)"]
    )
    def haccp_audit_endpoint(req: HaccpAuditRequest):
        try:
            from src.malolactic_and_oak import EnologicalHaccpAuditor
            auditor = EnologicalHaccpAuditor()
            res = auditor.audit_wine_lot(
                lot_id=req.lot_id,
                cultivar=req.cultivar,
                ph=req.ph,
                free_so2=req.free_so2,
                total_so2=req.total_so2,
                volatile_acidity=req.volatile_acidity,
                alcohol=req.alcohol,
                residual_sugar=req.residual_sugar,
                malic_acid=req.malic_acid,
                temperature=req.temperature,
            )
            return HaccpAuditResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Day 14: Tartrate, Bentonite, Membrane Filtration & Packaging Endpoints
    # ========================================================================

    @app.post(
        "/stabilization/evaluate-tartrate",
        response_model=TartrateEvaluationResponse,
        summary="Evaluate potassium bitartrate saturation and minicontact conductivity drop",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def evaluate_tartrate_endpoint(req: TartrateEvaluationRequest):
        try:
            from src.stabilization_and_packaging import TartrateStabilizationSimulator
            sim = TartrateStabilizationSimulator()
            res = sim.evaluate_tartrate_stability(
                tartaric_acid_g_l=req.tartaric_acid_g_l,
                potassium_mg_l=req.potassium_mg_l,
                ph=req.ph,
                alcohol_pct=req.alcohol_pct,
                storage_temp_c=req.storage_temp_c
            )
            return TartrateEvaluationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/stabilization/simulate-cold",
        response_model=ColdStabilizationSimulationResponse,
        summary="Simulate chilled contact crystallization kinetics of potassium bitartrate",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def simulate_cold_stabilization_endpoint(req: ColdStabilizationSimulationRequest):
        try:
            from src.stabilization_and_packaging import TartrateStabilizationSimulator
            sim = TartrateStabilizationSimulator()
            res = sim.simulate_cold_stabilization_kinetics(
                tartaric_acid_g_l=req.tartaric_acid_g_l,
                potassium_mg_l=req.potassium_mg_l,
                ph=req.ph,
                alcohol_pct=req.alcohol_pct,
                chilling_temp_c=req.chilling_temp_c,
                seed_crystal_g_l=req.seed_crystal_g_l,
                duration_hours=req.duration_hours
            )
            return ColdStabilizationSimulationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/stabilization/optimize-bentonite",
        response_model=BentoniteOptimizationResponse,
        summary="Optimize bentonite fining dosage via Langmuir isotherm for protein haze prevention",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def optimize_bentonite_endpoint(req: BentoniteOptimizationRequest):
        try:
            from src.stabilization_and_packaging import ProteinHazeAndBentoniteOptimizer
            opt = ProteinHazeAndBentoniteOptimizer()
            res = opt.optimize_bentonite_dosage(
                protein_content_mg_l=req.protein_content_mg_l,
                ph=req.ph,
                cultivar=req.cultivar,
                bentonite_type=req.bentonite_type
            )
            return BentoniteOptimizationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/filtration/crossflow-simulate",
        response_model=CrossflowFiltrationResponse,
        summary="Simulate crossflow microfiltration permeate flux decay and turbidity reduction",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def simulate_crossflow_endpoint(req: CrossflowFiltrationRequest):
        try:
            from src.stabilization_and_packaging import CrossflowFiltrationSimulator
            sim = CrossflowFiltrationSimulator()
            res = sim.simulate_crossflow_run(
                initial_turbidity_ntu=req.initial_turbidity_ntu,
                transmembrane_pressure_bar=req.transmembrane_pressure_bar,
                target_volume_liters=req.target_volume_liters,
                colloidal_index=req.colloidal_index,
                backwash_interval_hours=req.backwash_interval_hours,
                run_duration_hours=req.run_duration_hours
            )
            return CrossflowFiltrationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/packaging/gas-sparging-simulate",
        response_model=GasSpargingResponse,
        summary="Simulate micro-bubble nitrogen sparging deoxygenation efficiency",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def simulate_sparging_endpoint(req: GasSpargingRequest):
        try:
            from src.stabilization_and_packaging import DissolvedGasPackagingEngine
            eng = DissolvedGasPackagingEngine()
            res = eng.simulate_nitrogen_sparging(
                initial_do_mg_l=req.initial_do_mg_l,
                target_do_mg_l=req.target_do_mg_l,
                nitrogen_flow_ratio_nl_l=req.nitrogen_flow_ratio_nl_l,
                temperature_c=req.temperature_c
            )
            return GasSpargingResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/packaging/calculate-tpo",
        response_model=TotalPackagingOxygenResponse,
        summary="Calculate Total Packaging Oxygen (TPO) and Free SO2 consumption hit",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def calculate_tpo_endpoint(req: TotalPackagingOxygenRequest):
        try:
            from src.stabilization_and_packaging import DissolvedGasPackagingEngine
            eng = DissolvedGasPackagingEngine()
            res = eng.calculate_total_packaging_oxygen(
                dissolved_oxygen_wine_mg_l=req.dissolved_oxygen_wine_mg_l,
                headspace_volume_ml=req.headspace_volume_ml,
                bottle_volume_ml=req.bottle_volume_ml,
                headspace_o2_pct=req.headspace_o2_pct,
                wine_temperature_c=req.wine_temperature_c
            )
            return TotalPackagingOxygenResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/packaging/closure-shelf-life",
        response_model=ClosureShelfLifeResponse,
        summary="Forecast multi-year closure oxygen permeation, Free SO2 decay, and PremOx risk",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def closure_shelf_life_endpoint(req: ClosureShelfLifeRequest):
        try:
            from src.stabilization_and_packaging import ClosureShelfLifeForecaster
            forecaster = ClosureShelfLifeForecaster()
            res = forecaster.simulate_bottle_shelf_life(
                closure_type=req.closure_type,
                initial_free_so2=req.initial_free_so2,
                initial_ph=req.initial_ph,
                initial_tpo_mg_l=req.initial_tpo_mg_l,
                storage_temperature_c=req.storage_temperature_c,
                duration_months=req.duration_months
            )
            return ClosureShelfLifeResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/packaging/bottling-qc-audit",
        response_model=BottlingQCAuditResponse,
        summary="Audit bottling run batch against SPC legal metrology and filter membrane integrity",
        tags=["Stabilization & Packaging (Day 14)"]
    )
    def bottling_qc_audit_endpoint(req: BottlingQCAuditRequest):
        try:
            from src.stabilization_and_packaging import BottlingLineQCAuditor
            auditor = BottlingLineQCAuditor()
            res = auditor.audit_bottling_batch(
                lot_id=req.lot_id,
                fill_volumes_ml=req.fill_volumes_ml,
                closure_depths_mm=req.closure_depths_mm,
                headspace_vacuums_bar=req.headspace_vacuums_bar,
                filter_bubble_point_bar=req.filter_bubble_point_bar
            )
            return BottlingQCAuditResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Day 15 Endpoints: Logistics, Terroir, Hedonics & Winery Twin
    # ========================================================================

    @app.post(
        "/logistics/simulate-transit",
        response_model=TransitSimulationResponse,
        summary="Simulate intermodal transportation thermal shock, TDU accumulation, and closure seal integrity",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def simulate_transit_endpoint(req: TransitSimulationRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import ColdChainLogisticsSimulator
            sim = ColdChainLogisticsSimulator()
            res = sim.evaluate_transit_scenario(
                preset_name=req.preset_name,
                closure_type=req.closure_type,
                initial_free_so2=req.initial_free_so2,
                initial_a420_browning=req.initial_a420_browning,
                dissolved_co2_mg_l=req.dissolved_co2_mg_l
            )
            return TransitSimulationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/logistics/assess-barometric-seal",
        response_model=BarometricSealResponse,
        summary="Assess trans-closure differential pressure and extrusion risk under high altitude or air freight",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def assess_barometric_seal_endpoint(req: BarometricSealRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import ColdChainLogisticsSimulator
            sim = ColdChainLogisticsSimulator()
            res = sim.calculate_headspace_pressure_differential(
                liquid_temp_c=req.liquid_temp_c,
                ambient_pressure_kpa=req.ambient_pressure_kpa,
                dissolved_co2_mg_l=req.dissolved_co2_mg_l,
                headspace_volume_ml=req.headspace_volume_ml
            )
            return BarometricSealResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/authenticity/verify-terroir",
        response_model=TerroirVerificationResponse,
        summary="Verify wine geographical provenance authenticity using ICP-MS trace minerals and strontium ratios",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def verify_terroir_endpoint(req: TerroirVerificationRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import IsotopicTerroirAndFraudAuditor
            auditor = IsotopicTerroirAndFraudAuditor()
            res = auditor.verify_appellation_terroir(
                claimed_appellation=req.claimed_appellation,
                sample_minerals=req.sample_minerals,
                measured_sr_ratio=req.measured_sr_ratio,
                measured_delta13c=req.measured_delta13c,
                measured_delta18o=req.measured_delta18o
            )
            return TerroirVerificationResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/authenticity/detect-chaptalization",
        response_model=ChaptalizationAuditResponse,
        summary="Detect exogenous C4 cane or corn sugar addition via delta13C VPDB stable isotope mass balance",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def detect_chaptalization_endpoint(req: ChaptalizationAuditRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import IsotopicTerroirAndFraudAuditor
            auditor = IsotopicTerroirAndFraudAuditor()
            res = auditor.calculate_chaptalization_fraction(
                measured_delta13c=req.measured_delta13c,
                baseline_grape_delta13c=req.baseline_grape_delta13c
            )
            return ChaptalizationAuditResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/market/hedonic-valuation",
        response_model=HedonicValuationResponse,
        summary="Estimate fair market release value ($/bottle) using Rosen hedonic pricing econometrics",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def hedonic_valuation_endpoint(req: HedonicValuationRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import ConsumerHedonicAndMarketPricingOptimizer
            opt = ConsumerHedonicAndMarketPricingOptimizer()
            res = opt.estimate_bottle_market_value(
                predicted_quality=req.predicted_quality,
                cultivar=req.cultivar,
                alcohol=req.alcohol,
                ph=req.ph,
                residual_sugar=req.residual_sugar,
                volatile_acidity=req.volatile_acidity,
                cellar_aging_years=req.cellar_aging_years,
                appellation_tier=req.appellation_tier
            )
            return HedonicValuationResponse(
                predicted_quality=res["predicted_quality"],
                appellation_tier=res["appellation_tier"],
                estimated_market_price_usd=res["estimated_market_price_usd"],
                confidence_interval_90_usd=list(res["confidence_interval_90_usd"]),
                hedonic_components=res["hedonic_components"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/market/regional-palate-alignment",
        response_model=RegionalPalateResponse,
        summary="Calculate consumer palate alignment scores across North America, Europe, East Asia, and Nordics",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def regional_palate_endpoint(req: RegionalPalateRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import ConsumerHedonicAndMarketPricingOptimizer
            opt = ConsumerHedonicAndMarketPricingOptimizer()
            res = opt.calculate_regional_palate_alignment(
                alcohol=req.alcohol,
                ph=req.ph,
                residual_sugar=req.residual_sugar,
                volatile_acidity=req.volatile_acidity,
                sulphates=req.sulphates,
                oak_index=req.oak_index
            )
            return RegionalPalateResponse(regions=res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/winery-twin/simulate-end-to-end",
        response_model=DigitalWineryTwinResponse,
        summary="Run full 15-stage unified digital winery twin simulation from harvest must to retail release",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def winery_twin_endpoint(req: DigitalWineryTwinRequest):
        try:
            from src.transport_terroir_hedonic_and_twin import UnifiedDigitalWineryTwinPlatform
            twin = UnifiedDigitalWineryTwinPlatform()
            res = twin.simulate_end_to_end_winery_lifecycle(
                cultivar=req.cultivar,
                harvest_sugar_brix=req.harvest_sugar_brix,
                must_tartaric_acid=req.must_tartaric_acid,
                must_ph=req.must_ph,
                oak_regime=req.oak_regime,
                oak_duration_months=req.oak_duration_months,
                stabilization_method=req.stabilization_method,
                closure_choice=req.closure_choice,
                shipping_route=req.shipping_route,
                appellation_claimed=req.appellation_claimed,
                production_lot_bottles=req.production_lot_bottles
            )
            return DigitalWineryTwinResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/sustainability/lca-carbon-footprint",
        response_model=SustainabilityLCAResponse,
        summary="Calculate Scope 1, 2, and 3 carbon footprint (g CO2e/bottle) and water footprint",
        tags=["Logistics & Terroir Twin (Day 15)"]
    )
    def sustainability_lca_endpoint(req: SustainabilityLCARequest):
        try:
            from src.transport_terroir_hedonic_and_twin import UnifiedDigitalWineryTwinPlatform
            twin = UnifiedDigitalWineryTwinPlatform()
            res = twin.calculate_sustainability_lca(
                bottle_glass_weight_g=req.bottle_glass_weight_g,
                is_organic_certified=req.is_organic_certified,
                fermentation_co2_captured=req.fermentation_co2_captured,
                transit_distance_km=req.transit_distance_km,
                freight_mode=req.freight_mode
            )
            return SustainabilityLCAResponse(**res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Attach service instance to app state for test inspection
    app.state.service = service
    return app


# Default application instance for uvicorn
app = create_app()


def get_test_client():
    """Create a FastAPI TestClient instance for automated tests and notebooks."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from starlette.testclient import TestClient
        return TestClient(app)
