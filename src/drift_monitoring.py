"""
Continuous MLOps Monitoring, Statistical Drift Detection, and Retraining Pipeline.

Provides production routines for:
1. Statistical Covariate Shift Detection:
   - Population Stability Index (PSI) with monotonic quantile binning.
   - Two-sample Kolmogorov-Smirnov (K-S) goodness-of-fit testing.
   - Earth Mover's (Wasserstein) Distance and Jensen-Shannon Divergence.
2. Multivariate Chemical Anomaly & Outlier Screening (Isolation Forest).
3. Longitudinal Sommelier Sensory Audit Tracking & Concept Drift Monitoring.
4. Continuous Champion-Challenger Shadow Retraining & Automated Promotion Logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.feature_engineering import create_engineered_features, create_binned_targets

# Physicochemical raw features and engineered interaction features
RAW_PHYSICOCHEMICAL_FEATURES: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

ALL_MODEL_FEATURES: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "total_acidity",
    "bound_sulfur_dioxide",
    "free_to_total_so2_ratio",
    "alcohol_to_density_ratio",
    "sulphate_to_chloride_ratio"
]

# Standard PSI Risk Categorization Thresholds
PSI_THRESHOLD_GREEN = 0.10
PSI_THRESHOLD_AMBER = 0.25


def calculate_psi(
    reference: Union[np.ndarray, pd.Series],
    target: Union[np.ndarray, pd.Series],
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate the Population Stability Index (PSI) between a reference and target distribution.

    PSI Interpretation:
    - PSI < 0.10: No significant distribution change (Green / Stable).
    - 0.10 <= PSI < 0.25: Moderate shift; monitoring warning (Amber / Warning).
    - PSI >= 0.25: Significant distribution shift; action/retraining required (Red / Alert).

    Formula:
        PSI = sum_{i=1}^k (Target%_i - Ref%_i) * ln(Target%_i / Ref%_i)

    Parameters
    ----------
    reference : array-like
        Baseline / reference feature values (e.g. historical training partition).
    target : array-like
        Current / inference feature values (e.g. incoming harvest vintage).
    num_bins : int, default=10
        Number of quantile bins to partition the reference distribution.
    epsilon : float, default=1e-4
        Small offset to prevent division by zero or log of zero.

    Returns
    -------
    float
        Calculated PSI value rounded to 4 decimal places.
    """
    ref = np.asarray(reference, dtype=float)
    tar = np.asarray(target, dtype=float)

    ref = ref[~np.isnan(ref)]
    tar = tar[~np.isnan(tar)]

    if len(ref) == 0 or len(tar) == 0:
        return 0.0

    # Quantile bin edges based on reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, percentiles)

    # Ensure strictly monotonic bin edges
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        # If reference distribution has zero variance
        return 0.0
    
    unique_edges[0] = -np.inf
    unique_edges[-1] = np.inf

    # Calculate frequency counts in each bin
    ref_counts, _ = np.histogram(ref, bins=unique_edges)
    tar_counts, _ = np.histogram(tar, bins=unique_edges)

    # Compute proportions with smoothing epsilon
    k = len(ref_counts)
    ref_props = (ref_counts + epsilon) / (len(ref) + k * epsilon)
    tar_props = (tar_counts + epsilon) / (len(tar) + k * epsilon)

    psi_val = np.sum((tar_props - ref_props) * np.log(tar_props / ref_props))
    return float(np.round(np.maximum(0.0, psi_val), 4))


def compute_kolmogorov_smirnov(
    reference: Union[np.ndarray, pd.Series],
    target: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Compute the two-sample Kolmogorov-Smirnov test between reference and target.

    Tests the null hypothesis that both samples are drawn from the same continuous distribution.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'statistic' (supremum distance) and 'p_value'.
    """
    ref = np.asarray(reference, dtype=float)
    tar = np.asarray(target, dtype=float)
    ref = ref[~np.isnan(ref)]
    tar = tar[~np.isnan(tar)]

    res = stats.ks_2samp(ref, tar)
    return {
        "statistic": float(np.round(res.statistic, 4)),
        "p_value": float(res.pvalue)
    }


def compute_wasserstein_distance(
    reference: Union[np.ndarray, pd.Series],
    target: Union[np.ndarray, pd.Series]
) -> float:
    """
    Compute Earth Mover's (Wasserstein-1) Distance between reference and target.
    """
    ref = np.asarray(reference, dtype=float)
    tar = np.asarray(target, dtype=float)
    ref = ref[~np.isnan(ref)]
    tar = tar[~np.isnan(tar)]

    return float(np.round(stats.wasserstein_distance(ref, tar), 4))


def compute_feature_drift(
    reference_col: Union[np.ndarray, pd.Series],
    target_col: Union[np.ndarray, pd.Series],
    feature_name: str,
    alpha: float = 0.01,
    psi_green_limit: float = PSI_THRESHOLD_GREEN,
    psi_amber_limit: float = PSI_THRESHOLD_AMBER
) -> Dict[str, Any]:
    """
    Compute multi-metric drift analysis for an individual feature.

    Returns
    -------
    Dict[str, Any]
        Feature-level drift evaluation dictionary.
    """
    psi = calculate_psi(reference_col, target_col)
    ks = compute_kolmogorov_smirnov(reference_col, target_col)
    w_dist = compute_wasserstein_distance(reference_col, target_col)

    ref_arr = np.asarray(reference_col, dtype=float)
    ref_std = np.nanstd(ref_arr)
    norm_w_dist = float(np.round(w_dist / (ref_std + 1e-6), 4))

    # Determine alert level and drift status
    if psi >= psi_amber_limit or (ks["p_value"] < alpha and psi >= 0.15):
        status = "Severe Drift"
        alert = "RED"
    elif psi >= psi_green_limit or ks["p_value"] < alpha:
        status = "Moderate Drift"
        alert = "AMBER"
    else:
        status = "Stable"
        alert = "GREEN"

    return {
        "feature": feature_name,
        "psi": psi,
        "ks_statistic": ks["statistic"],
        "ks_p_value": ks["p_value"],
        "wasserstein_distance": w_dist,
        "normalized_wasserstein": norm_w_dist,
        "drift_status": status,
        "alert_level": alert
    }


def compute_dataset_drift_summary(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.01
) -> pd.DataFrame:
    """
    Compute a consolidated multi-metric drift scorecard across all features.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline reference dataframe (e.g. historical harvest).
    target_df : pd.DataFrame
        Target batch dataframe (e.g. newly harvested vintage).
    feature_cols : List[str], optional
        List of features to analyze. Defaults to RAW_PHYSICOCHEMICAL_FEATURES.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: Feature, PSI, KS_Stat, KS_p_val, Wasserstein, Status, Alert.
    """
    cols = feature_cols if feature_cols is not None else RAW_PHYSICOCHEMICAL_FEATURES
    results = []

    for col in cols:
        if col in reference_df.columns and col in target_df.columns:
            drift_res = compute_feature_drift(
                reference_df[col],
                target_df[col],
                feature_name=col,
                alpha=alpha
            )
            results.append({
                "Feature": drift_res["feature"],
                "PSI": drift_res["psi"],
                "KS_Statistic": drift_res["ks_statistic"],
                "KS_P_Value": drift_res["ks_p_value"],
                "Wasserstein_Dist": drift_res["wasserstein_distance"],
                "Norm_Wasserstein": drift_res["normalized_wasserstein"],
                "Drift_Status": drift_res["drift_status"],
                "Alert_Level": drift_res["alert_level"]
            })

    return pd.DataFrame(results)


def simulate_vintage_drift(
    df: pd.DataFrame,
    scenario: str = "climate_warming",
    severity: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Simulate realistic enological covariate shift scenarios on wine physicochemical profiles.

    Scenarios:
    1. 'climate_warming': Extended heatwave leading to higher sugar/alcohol, reduced fixed acidity,
       elevated pH, and higher volatile acidity from microbial activity.
    2. 'cool_rainy_harvest': Excessive precipitation diluting grape berries (lower alcohol, higher density),
       and requiring heavy sulphur additions by cellar masters to prevent fungal infection.
    3. 'instrument_drift': Laboratory equipment calibration offset on volatile acidity & sulphites.
    4. 'stable_harvest': Random resampling with zero systematic shift (control).

    Parameters
    ----------
    df : pd.DataFrame
        Original unshifted baseline wine dataframe.
    scenario : str
        Simulation scenario key.
    severity : float, default=1.0
        Multiplier scaling the magnitude of the drift.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shifted harvest dataframe.
    """
    np.random.seed(random_state)
    df_shifted = df.copy()

    if scenario == "climate_warming":
        # Elevated temperatures boost grape maturity and sugar conversion to alcohol
        df_shifted["alcohol"] = df_shifted["alcohol"] + np.random.normal(0.85 * severity, 0.2, len(df))
        df_shifted["alcohol"] = np.clip(df_shifted["alcohol"], 8.0, 16.5)

        # Thermal degradation of natural tartaric/malic organic acids
        df_shifted["fixed acidity"] = df_shifted["fixed acidity"] - np.random.normal(0.65 * severity, 0.25, len(df))
        df_shifted["fixed acidity"] = np.clip(df_shifted["fixed acidity"], 4.5, 15.0)

        # Resulting upward shift in pH
        df_shifted["pH"] = df_shifted["pH"] + np.random.normal(0.12 * severity, 0.03, len(df))
        df_shifted["pH"] = np.clip(df_shifted["pH"], 2.8, 4.2)

        # Faster fermentation temperatures increase acetic acid production
        df_shifted["volatile acidity"] = df_shifted["volatile acidity"] + np.random.normal(0.09 * severity, 0.03, len(df))
        df_shifted["volatile acidity"] = np.clip(df_shifted["volatile acidity"], 0.15, 1.6)

    elif scenario == "cool_rainy_harvest":
        # Precipitation dilutes juice sugars, decreasing potential alcohol
        df_shifted["alcohol"] = df_shifted["alcohol"] - np.random.normal(0.75 * severity, 0.2, len(df))
        df_shifted["alcohol"] = np.clip(df_shifted["alcohol"], 8.0, 14.0)

        # Winemakers add higher potassium metabisulfite to combat grey rot (Botrytis cinerea)
        df_shifted["total sulfur dioxide"] = df_shifted["total sulfur dioxide"] + np.random.normal(24.0 * severity, 5.0, len(df))
        df_shifted["free sulfur dioxide"] = df_shifted["free sulfur dioxide"] + np.random.normal(8.5 * severity, 2.0, len(df))
        
        # High unfermented hydrometer density
        df_shifted["density"] = df_shifted["density"] + (0.0022 * severity)

    elif scenario == "instrument_drift":
        # Systematic calibration bias on HPLC & aeration-oxidation analytical instruments
        df_shifted["volatile acidity"] = df_shifted["volatile acidity"] * (1.0 + 0.25 * severity)
        df_shifted["sulphates"] = df_shifted["sulphates"] * (1.0 - 0.20 * severity)

    elif scenario == "stable_harvest":
        # Null scenario: strictly stochastic variation without distributional shift
        pass

    # Ensure consistency: total SO2 >= free SO2
    df_shifted["total sulfur dioxide"] = np.maximum(
        df_shifted["total sulfur dioxide"],
        df_shifted["free sulfur dioxide"] + 2.0
    )

    return df_shifted


class MultivariateAnomalyDetector:
    """
    Isolation Forest multi-attribute chemical outlier screening engine.
    Detects adulterated, contaminated, or aberrant laboratory wine samples prior to inference.
    """

    def __init__(self, contamination: float = 0.04, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.feature_cols: List[str] = RAW_PHYSICOCHEMICAL_FEATURES
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MultivariateAnomalyDetector":
        """Fit Isolation Forest on historical baseline reference data."""
        X = df[self.feature_cols].copy()
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Classify samples:
        1: Inlier (Standard / Clean sample)
        -1: Anomaly (Physicochemically aberrant sample)
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before predicting.")
        X = df[self.feature_cols].copy()
        return self.model.predict(X)

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """Compute anomaly score (lower is more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before scoring.")
        X = df[self.feature_cols].copy()
        return self.model.score_samples(X)

    def filter_clean_samples(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split input dataset into clean inliers and anomalous outliers."""
        preds = self.predict(df)
        clean_df = df[preds == 1].copy()
        anomaly_df = df[preds == -1].copy()
        return clean_df, anomaly_df


class AuditPerformanceTracker:
    """
    Evaluates delayed ground-truth sensory panel audit batches to detect concept drift.
    Calculates accuracy, balanced accuracy, F1, and ROC-AUC degradation over time.
    """

    def __init__(self, baseline_benchmarks: Optional[Dict[str, float]] = None):
        self.baseline_benchmarks = baseline_benchmarks or {
            "roc_auc": 0.9368,
            "balanced_accuracy": 0.8520,
            "f1_score": 0.6970,
            "accuracy": 0.8625
        }
        self.audit_log: List[Dict[str, Any]] = []

    def evaluate_audit_batch(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: np.ndarray,
        threshold: float = 0.35,
        batch_id: str = "vintage_audit"
    ) -> Dict[str, Any]:
        """
        Compute standard discrimination and calibration metrics on an audited sensory batch.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_pred = (y_pred_proba >= threshold).astype(int)

        acc = float(np.round(accuracy_score(y_t, y_pred), 4))
        bal_acc = float(np.round(balanced_accuracy_score(y_t, y_pred), 4))
        prec = float(np.round(precision_score(y_t, y_pred, zero_division=0), 4))
        rec = float(np.round(recall_score(y_t, y_pred, zero_division=0), 4))
        f1 = float(np.round(f1_score(y_t, y_pred, zero_division=0), 4))
        
        try:
            auc = float(np.round(roc_auc_score(y_t, y_pred_proba), 4))
        except Exception:
            auc = 0.5

        try:
            pr_auc = float(np.round(average_precision_score(y_t, y_pred_proba), 4))
        except Exception:
            pr_auc = 0.0

        # Calculate degradation against production SLA benchmarks
        delta_auc = float(np.round(auc - self.baseline_benchmarks["roc_auc"], 4))
        delta_bal_acc = float(np.round(bal_acc - self.baseline_benchmarks["balanced_accuracy"], 4))

        # Alert if performance drops significantly
        concept_drift_detected = (delta_auc < -0.05) or (bal_acc < 0.78)

        record = {
            "batch_id": batch_id,
            "sample_count": len(y_t),
            "threshold": threshold,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "delta_roc_auc": delta_auc,
            "delta_balanced_accuracy": delta_bal_acc,
            "concept_drift_detected": concept_drift_detected,
            "alert_status": "RED" if concept_drift_detected else "GREEN"
        }
        self.audit_log.append(record)
        return record


class DriftRetrainingPipeline:
    """
    Automated Retraining, Champion-Challenger Shadow Validation, and Hot Reload Framework.
    """

    def __init__(
        self,
        champion_pipeline: Pipeline,
        reference_X: pd.DataFrame,
        reference_y: pd.Series,
        operating_threshold: float = 0.35
    ):
        self.champion_pipeline = champion_pipeline
        self.reference_X = reference_X.copy()
        self.reference_y = reference_y.copy()
        self.operating_threshold = operating_threshold
        self.challenger_pipeline: Optional[Pipeline] = None
        self.retraining_history: List[Dict[str, Any]] = []

    def evaluate_retraining_trigger(
        self,
        drift_summary_df: pd.DataFrame,
        audit_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate multi-criteria automated retraining trigger rules:
        Rule 1: >= 2 physicochemical features exhibit RED PSI (>= 0.25).
        Rule 2: Critical enological drivers (alcohol, volatile acidity, sulphates) exhibit RED PSI.
        Rule 3: Sensory audit ROC-AUC drops by > 0.05 or Balanced Accuracy drops below 0.78.

        Returns
        -------
        Tuple[bool, List[str]]
            (Should Retrain boolean, List of triggered alarm reasons).
        """
        reasons: List[str] = []
        should_retrain = False

        # Rule 1: Feature count with RED alert
        red_features = drift_summary_df[drift_summary_df["Alert_Level"] == "RED"]["Feature"].tolist()
        if len(red_features) >= 2:
            reasons.append(f"Multiple features ({len(red_features)}) exceeded Red PSI threshold: {red_features}")
            should_retrain = True

        # Rule 2: Critical lever drift
        critical_levers = ["alcohol", "volatile acidity", "sulphates"]
        red_critical = [f for f in red_features if f in critical_levers]
        if red_critical:
            reasons.append(f"Critical enological quality driver(s) drifted significantly: {red_critical}")
            should_retrain = True

        # Rule 3: Sensory concept drift
        if audit_metrics and audit_metrics.get("concept_drift_detected", False):
            reasons.append(
                f"Sensory audit performance decayed below SLA: "
                f"ROC-AUC={audit_metrics['roc_auc']} (Delta={audit_metrics['delta_roc_auc']}), "
                f"Bal_Acc={audit_metrics['balanced_accuracy']}"
            )
            should_retrain = True

        return should_retrain, reasons

    def train_challenger(
        self,
        new_X: pd.DataFrame,
        new_y: pd.Series,
        cv_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train a candidate Challenger Random Forest pipeline on combined historical + new audited vintage data.
        """
        combined_X = pd.concat([self.reference_X, new_X], ignore_index=True)
        combined_y = pd.concat([self.reference_y, new_y], ignore_index=True)

        challenger = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=250,
                max_depth=16,
                min_samples_split=3,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        # Run 5-fold Stratified Cross-Validation on combined corpus
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(challenger, combined_X, combined_y, cv=cv, scoring="roc_auc")
        cv_bal_acc = cross_val_score(challenger, combined_X, combined_y, cv=cv, scoring="balanced_accuracy")

        # Fit challenger on the full combined dataset
        challenger.fit(combined_X, combined_y)
        self.challenger_pipeline = challenger

        return {
            "challenger_pipeline": challenger,
            "training_samples": len(combined_X),
            "cv_roc_auc_mean": float(np.round(np.mean(cv_scores), 4)),
            "cv_roc_auc_std": float(np.round(np.std(cv_scores), 4)),
            "cv_balanced_accuracy_mean": float(np.round(np.mean(cv_bal_acc), 4)),
            "cv_balanced_accuracy_std": float(np.round(np.std(cv_bal_acc), 4))
        }

    def execute_champion_challenger_tournament(
        self,
        audit_test_X: pd.DataFrame,
        audit_test_y: pd.Series
    ) -> Dict[str, Any]:
        """
        Side-by-side tournament comparing incumbent Champion against retrained Challenger on fresh audited test set.
        """
        if self.challenger_pipeline is None:
            raise RuntimeError("Challenger pipeline must be trained prior to running tournament.")

        y_true = audit_test_y.values

        # Score Champion
        champ_proba = self.champion_pipeline.predict_proba(audit_test_X)[:, 1]
        champ_pred = (champ_proba >= self.operating_threshold).astype(int)
        champ_auc = float(np.round(roc_auc_score(y_true, champ_proba), 4))
        champ_bal_acc = float(np.round(balanced_accuracy_score(y_true, champ_pred), 4))
        champ_f1 = float(np.round(f1_score(y_true, champ_pred, zero_division=0), 4))

        # Score Challenger
        chal_proba = self.challenger_pipeline.predict_proba(audit_test_X)[:, 1]
        chal_pred = (chal_proba >= self.operating_threshold).astype(int)
        chal_auc = float(np.round(roc_auc_score(y_true, chal_proba), 4))
        chal_bal_acc = float(np.round(balanced_accuracy_score(y_true, chal_pred), 4))
        chal_f1 = float(np.round(f1_score(y_true, chal_pred, zero_division=0), 4))

        # Promotion Criterion: Challenger must outperform or match Champion across Balanced Accuracy and ROC-AUC
        promote = (chal_auc >= champ_auc) and (chal_bal_acc >= champ_bal_acc)
        decision = "PROMOTED" if promote else "REJECTED"

        tournament_result = {
            "decision": decision,
            "promoted": promote,
            "champion_metrics": {
                "roc_auc": champ_auc,
                "balanced_accuracy": champ_bal_acc,
                "f1_score": champ_f1
            },
            "challenger_metrics": {
                "roc_auc": chal_auc,
                "balanced_accuracy": chal_bal_acc,
                "f1_score": chal_f1
            },
            "delta_roc_auc": float(np.round(chal_auc - champ_auc, 4)),
            "delta_balanced_accuracy": float(np.round(chal_bal_acc - champ_bal_acc, 4)),
            "delta_f1": float(np.round(chal_f1 - champ_f1, 4))
        }

        if promote:
            # Swap champion
            self.champion_pipeline = self.challenger_pipeline

        self.retraining_history.append(tournament_result)
        return tournament_result

    def export_governance_report(self, filepath: Path) -> None:
        """Export comprehensive retraining audit trial to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "tournament_records": self.retraining_history,
            "operating_threshold": self.operating_threshold,
            "governance_status": "COMPLIANT"
        }
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=4)
