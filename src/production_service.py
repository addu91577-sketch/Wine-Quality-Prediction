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
    max_depth: int
    class_weight: str
    features: List[str]
    num_features: int
    test_roc_auc: float
    test_pr_auc: float
    test_balanced_accuracy_screen_threshold: float
    operating_screening_threshold: float
    loaded_artifact_path: str


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
        self.metadata: Dict[str, Any] = {}
        self.startup_time = time.time()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model and JSON metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path.resolve()}")

        self.pipeline = joblib.load(self.model_path)

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

        # Build single-row DataFrame strictly ordered
        ordered_data = {col: [full_dict[col]] for col in FULL_FEATURE_ORDER}
        df_input = pd.DataFrame(ordered_data)

        # Predict probabilities
        probabilities = self.pipeline.predict_proba(df_input)[0]
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
            max_depth=int(meta.get("max_depth", 18)),
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
