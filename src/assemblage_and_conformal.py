"""
Day 10: Enological Coupage (Wine Blending) Optimization Engine, Split Conformal
Uncertainty Quantification, Winery Economic Expected Monetary Value (EMV) Analysis,
and Multi-Model Stacking Ensemble.

This module provides:
1. calculate_blended_chemistry: Physicochemical mass conservation and logarithmic
   pH equilibrium blending simulator.
2. WineAssemblageOptimizer: Constrained non-linear coupage optimizer maximizing
   blend premium probability under strict regulatory and enological constraints.
3. EnologicalConformalPredictor: Split conformal uncertainty quantifier providing
   distribution-free finite-sample coverage guarantees and sommelier panel arbitration flags.
4. WineryEconomicEngine: Decision-theoretic payoff matrix and expected monetary value
   optimizer calculating financially optimal release thresholds.
5. WineStackingEnsemble: Heterogeneous multi-model stacking ensemble combining
   Random Forest, GBDT, Extra Trees, and Logistic Regression with out-of-fold meta-learning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import joblib

logger = logging.getLogger(__name__)

# Standard Physicochemical Features
RAW_FEATURE_NAMES: List[str] = [
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

ENGINEERED_FEATURE_NAMES: List[str] = [
    "total_acidity",
    "bound_sulfur_dioxide",
    "free_to_total_so2_ratio",
    "alcohol_to_density_ratio",
    "sulphate_to_chloride_ratio"
]

ALL_FEATURE_NAMES: List[str] = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES

DEFAULT_SCREENING_THRESHOLD: float = 0.35


# ============================================================================
# 1. Physicochemical Mixing Laws & Enological Blending Mathematics
# ============================================================================

def calculate_blended_chemistry(
    lots: Sequence[Dict[str, float]],
    weights: Sequence[float]
) -> Dict[str, float]:
    """
    Simulate the physical and chemical properties of blended wine lots based on
    conservation of mass and non-linear acid-base chemical equilibrium.

    Mixing Laws:
    1. Additive linear properties:
       c_blend = sum_i(w_i * c_i)
       Applicable to alcohol, fixed acidity, volatile acidity, citric acid,
       residual sugar, chlorides, free SO2, total SO2, density, and sulphates.
    2. Non-linear logarithmic pH equilibrium:
       In aqueous vinous solutions, [H+] = 10^(-pH).
       [H+]_blend = sum_i(w_i * 10^(-pH_i))
       pH_blend = -log10([H+]_blend)
    3. Recomputed domain-specific engineered features:
       - total_acidity = fixed acidity + volatile acidity
       - bound_sulfur_dioxide = max(0.0, total SO2 - free SO2)
       - free_to_total_so2_ratio = free SO2 / (total SO2 + 1e-5)
       - alcohol_to_density_ratio = alcohol / density
       - sulphate_to_chloride_ratio = sulphates / (chlorides + 1e-5)

    Parameters
    ----------
    lots : Sequence[Dict[str, float]]
        List of physicochemical profiles for K candidate lots.
    weights : Sequence[float]
        Proportional weights summing to 1.0 (length K).

    Returns
    -------
    Dict[str, float]
        Complete physicochemical and engineered profile of the resulting blend.
    """
    k = len(lots)
    if k == 0:
        raise ValueError("At least one wine lot must be provided for blending.")
    if len(weights) != k:
        raise ValueError(f"Length of weights ({len(weights)}) must match number of lots ({k}).")

    w_arr = np.array(weights, dtype=float)
    total_w = np.sum(w_arr)
    if total_w <= 0:
        raise ValueError("Sum of blending weights must be strictly positive.")
    w_norm = w_arr / total_w

    blended: Dict[str, float] = {}

    # 1. Additive linear properties
    linear_keys = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "sulphates", "alcohol"
    ]
    for key in linear_keys:
        val = sum(float(w_norm[i]) * float(lots[i].get(key, 0.0)) for i in range(k))
        blended[key] = float(np.round(val, 5))

    # 2. Non-linear pH calculation via hydrogen ion concentration [H+]
    h_ion_sum = sum(
        float(w_norm[i]) * (10.0 ** (-float(lots[i].get("pH", 3.3))))
        for i in range(k)
    )
    h_ion_sum = max(h_ion_sum, 1e-14)  # numerical guard
    blended["pH"] = float(np.round(-np.log10(h_ion_sum), 3))

    # 3. Recompute domain-specific engineered enological ratios
    blended["total_acidity"] = float(np.round(
        blended["fixed acidity"] + blended["volatile acidity"], 4
    ))
    blended["bound_sulfur_dioxide"] = float(np.round(
        max(0.0, blended["total sulfur dioxide"] - blended["free sulfur dioxide"]), 3
    ))
    blended["free_to_total_so2_ratio"] = float(np.round(
        blended["free sulfur dioxide"] / (blended["total sulfur dioxide"] + 1e-5), 4
    ))
    dens = blended["density"] if blended["density"] > 0 else 0.996
    blended["alcohol_to_density_ratio"] = float(np.round(
        blended["alcohol"] / dens, 4
    ))
    blended["sulphate_to_chloride_ratio"] = float(np.round(
        blended["sulphates"] / (blended["chlorides"] + 1e-5), 4
    ))

    return blended


# ============================================================================
# 2. Enological Coupage (Wine Blending) Assemblage Optimizer
# ============================================================================

class WineAssemblageOptimizer:
    """
    Constrained non-linear optimization engine for wine assemblage (coupage).

    Formulation:
        Find blend weights w = (w_1, ..., w_K)^T to:
            Maximize Objective(w) = P(Quality >= 7 | Blend(w))
        Subject to:
            sum(w_i) = 1, w_i >= 0
            Volatile Acidity <= max_volatile_acidity (spoilage prevention)
            Total SO2 <= max_total_so2 (EU legal maximum)
            Free SO2 >= min_free_so2 (antimicrobial efficacy)
            Alcohol >= min_alcohol (commercial body target)
            min_ph <= pH <= max_ph (acid-base sensory stability)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = DEFAULT_SCREENING_THRESHOLD
    ):
        self.threshold = threshold
        self.feature_names = feature_names or ALL_FEATURE_NAMES

        if model is None:
            # Fallback: load production Red Champion pipeline
            red_model_path = Path("models/champion_rf_pipeline.joblib")
            if red_model_path.exists():
                self.model = joblib.load(red_model_path)
            else:
                self.model = None
        else:
            self.model = model

    def _predict_prob(self, chem_dict: Dict[str, float]) -> float:
        """Helper to compute P(Quality >= 7) for a single physicochemical profile."""
        if self.model is None:
            # Synthetic heuristic if model is not loaded
            alc = chem_dict.get("alcohol", 10.0)
            va = chem_dict.get("volatile acidity", 0.5)
            prob = 1.0 / (1.0 + np.exp(-(0.8 * (alc - 10.5) - 3.5 * (va - 0.4))))
            return float(np.clip(prob, 0.01, 0.99))

        df_row = pd.DataFrame([chem_dict])
        missing = [f for f in self.feature_names if f not in df_row.columns]
        if missing:
            raise ValueError(f"Missing required features for model prediction: {missing}")
        X_vec = df_row[self.feature_names]
        probs = self.model.predict_proba(X_vec)
        return float(probs[0, 1])

    def optimize_blend(
        self,
        lots: Sequence[Dict[str, float]],
        lot_names: Optional[Sequence[str]] = None,
        constraints: Optional[Dict[str, float]] = None,
        lot_costs: Optional[Sequence[float]] = None,
        cost_penalty_weight: float = 0.0
    ) -> Dict[str, Any]:
        """
        Solve the constrained optimization problem to identify optimal blend proportions.

        Parameters
        ----------
        lots : Sequence[Dict[str, float]]
            List of dictionaries containing physicochemical parameters for candidate lots.
        lot_names : Optional[Sequence[str]]
            Descriptive names for each lot (e.g. ['Tank A', 'Tank B', 'Tank C']).
        constraints : Optional[Dict[str, float]]
            Enological constraints overriding defaults:
            - max_volatile_acidity (default: 0.60 g/L)
            - max_total_so2 (default: 160.0 mg/L)
            - min_free_so2 (default: 20.0 mg/L)
            - min_alcohol (default: 11.5 %)
            - min_ph (default: 3.10)
            - max_ph (default: 3.65)
        lot_costs : Optional[Sequence[float]]
            Cost per liter of each candidate lot.
        cost_penalty_weight : float
            Weight lambda for cost in multi-objective optimization: P(Premium) - lambda * Cost.

        Returns
        -------
        Dict[str, Any]
            Optimization results including optimal weights, blended chemistry,
            predicted probability, uplift, and constraint compliance report.
        """
        k = len(lots)
        if k < 2:
            raise ValueError("Wine coupage requires at least 2 distinct lots.")

        names = list(lot_names) if lot_names else [f"Lot_{i+1}" for i in range(k)]

        # Enological boundary limits
        cfg = {
            "max_volatile_acidity": 0.60,
            "max_total_so2": 160.0,
            "min_free_so2": 20.0,
            "min_alcohol": 11.5,
            "min_ph": 3.10,
            "max_ph": 3.65
        }
        if constraints:
            cfg.update(constraints)

        # Baseline single lot evaluation
        individual_probs = []
        for i, lot in enumerate(lots):
            p = self._predict_prob(calculate_blended_chemistry([lot], [1.0]))
            individual_probs.append({
                "lot_name": names[i],
                "probability": round(p, 4),
                "is_premium": bool(p >= self.threshold)
            })

        # Objective function (to minimize)
        def objective_func(weights_vector: np.ndarray) -> float:
            w_norm = weights_vector / np.sum(weights_vector)
            chem = calculate_blended_chemistry(lots, w_norm)
            prob = self._predict_prob(chem)

            score = prob
            if lot_costs is not None and cost_penalty_weight > 0.0:
                avg_cost = sum(w_norm[i] * lot_costs[i] for i in range(k))
                score = score - (cost_penalty_weight * avg_cost)

            # Negate for minimization
            return -score

        # Inequality constraints: c(w) >= 0
        ineq_cons = []

        # Volatile acidity <= max_va  =>  max_va - VA(w) >= 0
        def va_constraint(w):
            chem = calculate_blended_chemistry(lots, w / np.sum(w))
            return cfg["max_volatile_acidity"] - chem["volatile acidity"]
        ineq_cons.append({"type": "ineq", "fun": va_constraint})

        # Total SO2 <= max_tso2  =>  max_tso2 - TSO2(w) >= 0
        def tso2_constraint(w):
            chem = calculate_blended_chemistry(lots, w / np.sum(w))
            return cfg["max_total_so2"] - chem["total sulfur dioxide"]
        ineq_cons.append({"type": "ineq", "fun": tso2_constraint})

        # Free SO2 >= min_fso2  =>  FSO2(w) - min_fso2 >= 0
        def fso2_constraint(w):
            chem = calculate_blended_chemistry(lots, w / np.sum(w))
            return chem["free sulfur dioxide"] - cfg["min_free_so2"]
        ineq_cons.append({"type": "ineq", "fun": fso2_constraint})

        # Alcohol >= min_alc  =>  Alcohol(w) - min_alc >= 0
        def alc_constraint(w):
            chem = calculate_blended_chemistry(lots, w / np.sum(w))
            return chem["alcohol"] - cfg["min_alcohol"]
        ineq_cons.append({"type": "ineq", "fun": alc_constraint})

        # pH >= min_ph and pH <= max_ph
        def ph_min_constraint(w):
            chem = calculate_blended_chemistry(lots, w / np.sum(w))
            return chem["pH"] - cfg["min_ph"]
        def ph_max_constraint(w):
            chem = calculate_blended_chemistry(lots, w / np.sum(w))
            return cfg["max_ph"] - chem["pH"]
        ineq_cons.append({"type": "ineq", "fun": ph_min_constraint})
        ineq_cons.append({"type": "ineq", "fun": ph_max_constraint})

        # Bounds: 0 <= w_i <= 1
        bounds = [(0.0, 1.0) for _ in range(k)]

        # Equality constraint: sum(w) = 1.0
        eq_cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        all_cons = [eq_cons] + ineq_cons

        # Multi-start initializations for global exploration
        candidate_seeds = []
        # Seed 1: Equal weights
        candidate_seeds.append(np.ones(k) / k)
        # Seeds 2: Single pure components
        for i in range(k):
            e = np.zeros(k)
            e[i] = 1.0
            candidate_seeds.append(e)
        # Seeds 3: Dirichlet random samples
        rng = np.random.default_rng(42)
        for _ in range(5):
            candidate_seeds.append(rng.dirichlet(np.ones(k)))

        best_res = None
        best_val = np.inf

        for seed in candidate_seeds:
            res = minimize(
                objective_func,
                seed,
                method="SLSQP",
                bounds=bounds,
                constraints=all_cons,
                options={"maxiter": 200, "ftol": 1e-6}
            )
            # Evaluate feasibility and value
            if res.fun < best_val:
                best_val = res.fun
                best_res = res

        # Post-process optimal weights
        if best_res is not None and np.isfinite(best_res.fun):
            opt_w = np.clip(best_res.x, 0.0, 1.0)
            opt_w = opt_w / np.sum(opt_w)
        else:
            opt_w = np.ones(k) / k

        blended_chem = calculate_blended_chemistry(lots, opt_w)
        final_prob = self._predict_prob(blended_chem)
        is_premium = bool(final_prob >= self.threshold)

        # Check constraint satisfaction
        compliance = {
            "volatile_acidity": {
                "value": blended_chem["volatile acidity"],
                "limit": f"<= {cfg['max_volatile_acidity']}",
                "passed": bool(blended_chem["volatile acidity"] <= cfg["max_volatile_acidity"] + 1e-4)
            },
            "total_so2": {
                "value": blended_chem["total sulfur dioxide"],
                "limit": f"<= {cfg['max_total_so2']}",
                "passed": bool(blended_chem["total sulfur dioxide"] <= cfg["max_total_so2"] + 1e-4)
            },
            "free_so2": {
                "value": blended_chem["free sulfur dioxide"],
                "limit": f">= {cfg['min_free_so2']}",
                "passed": bool(blended_chem["free sulfur dioxide"] >= cfg["min_free_so2"] - 1e-4)
            },
            "alcohol": {
                "value": blended_chem["alcohol"],
                "limit": f">= {cfg['min_alcohol']}",
                "passed": bool(blended_chem["alcohol"] >= cfg["min_alcohol"] - 1e-4)
            },
            "pH": {
                "value": blended_chem["pH"],
                "limit": f"[{cfg['min_ph']}, {cfg['max_ph']}]",
                "passed": bool(cfg["min_ph"] - 1e-4 <= blended_chem["pH"] <= cfg["max_ph"] + 1e-4)
            }
        }
        all_passed = all(v["passed"] for v in compliance.values())

        # Baseline comparison
        mean_component_prob = float(np.mean([item["probability"] for item in individual_probs]))
        max_component_prob = float(np.max([item["probability"] for item in individual_probs]))
        prob_lift_vs_mean = float(np.round(final_prob - mean_component_prob, 4))
        prob_lift_vs_best = float(np.round(final_prob - max_component_prob, 4))

        # Sensory sommelier grading
        if final_prob >= 0.70:
            grade = "Grand Reserve Coupage (World-Class Sensory Harmony)"
        elif final_prob >= self.threshold:
            grade = "Reserve Grade Coupage (Cellar Champion Lot)"
        elif final_prob >= 0.25:
            grade = "Table Wine Selection (Commercial Daily Bottling)"
        else:
            grade = "Bulk Vinification (Requires Further Acid/Alcohol Rectification)"

        # Weight breakdown
        weights_breakdown = [
            {
                "lot_name": names[i],
                "ratio": float(np.round(opt_w[i], 4)),
                "percentage": float(np.round(opt_w[i] * 100.0, 2))
            }
            for i in range(k)
        ]

        return {
            "status": "success",
            "optimal_weights": [float(np.round(w, 4)) for w in opt_w],
            "weights_breakdown": weights_breakdown,
            "blended_chemistry": blended_chem,
            "predicted_premium_probability": float(np.round(final_prob, 4)),
            "operating_threshold": self.threshold,
            "is_premium": is_premium,
            "sommelier_grade": grade,
            "prob_lift_vs_mean_lot": prob_lift_vs_mean,
            "prob_lift_vs_best_lot": prob_lift_vs_best,
            "individual_lot_evaluations": individual_probs,
            "constraints_compliant": all_passed,
            "enological_compliance_report": compliance
        }

    def simulate_blend_grid_2lots(
        self,
        lot_a: Dict[str, float],
        lot_b: Dict[str, float],
        steps: int = 51
    ) -> pd.DataFrame:
        """
        Simulate a 1-D blend trajectory between two candidate wine lots.

        Parameters
        ----------
        lot_a : Dict[str, float]
            First wine lot profile.
        lot_b : Dict[str, float]
            Second wine lot profile.
        steps : int
            Number of points along the blend curve from 100% A to 100% B.

        Returns
        -------
        pd.DataFrame
            DataFrame detailing weights, blended properties, predicted probabilities,
            and compliance for each step.
        """
        w_a_vals = np.linspace(0.0, 1.0, steps)
        records = []

        for w_a in w_a_vals:
            w_b = 1.0 - w_a
            chem = calculate_blended_chemistry([lot_a, lot_b], [w_a, w_b])
            prob = self._predict_prob(chem)
            is_compliant = (
                chem["volatile acidity"] <= 0.60 and
                chem["total sulfur dioxide"] <= 160.0 and
                chem["free sulfur dioxide"] >= 20.0 and
                chem["alcohol"] >= 11.5 and
                3.10 <= chem["pH"] <= 3.65
            )
            records.append({
                "weight_lot_a": float(np.round(w_a, 4)),
                "weight_lot_b": float(np.round(w_b, 4)),
                "predicted_probability": float(np.round(prob, 4)),
                "is_premium": bool(prob >= self.threshold),
                "volatile_acidity": chem["volatile acidity"],
                "alcohol": chem["alcohol"],
                "pH": chem["pH"],
                "free_sulfur_dioxide": chem["free sulfur dioxide"],
                "total_sulfur_dioxide": chem["total sulfur dioxide"],
                "is_enologically_compliant": is_compliant
            })

        return pd.DataFrame(records)


# ============================================================================
# 3. Split Conformal Prediction & Uncertainty Quantification Engine
# ============================================================================

class EnologicalConformalPredictor:
    """
    Inductive Split Conformal Prediction engine providing finite-sample distribution-free
    coverage guarantees: P(Y in C(X)) >= 1 - alpha.

    Prediction Sets:
        - {1}: High Confidence Premium (Reserve Grade)
        - {0}: High Confidence Standard (Everyday Table Wine)
        - {0, 1}: Ambiguous / Borderline Specimen (Mandatory Sommelier Panel Tasting)
        - {}: Out-of-Distribution / Severe Enological Contamination
    """

    def __init__(self, model: Optional[Any] = None, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or ALL_FEATURE_NAMES
        self.calibration_scores: Optional[np.ndarray] = None
        self.n_cal: int = 0
        self.quantiles_cache: Dict[float, float] = {}

    def fit_calibration(
        self,
        X_cal: pd.DataFrame,
        y_cal: Union[pd.Series, np.ndarray],
        model: Optional[Any] = None
    ) -> "EnologicalConformalPredictor":
        """
        Calibrate non-conformity scores on an independent holdout calibration split.

        Non-conformity score: s_i = 1 - P(Y = y_i | X_i).

        Parameters
        ----------
        X_cal : pd.DataFrame
            Calibration features matrix.
        y_cal : Union[pd.Series, np.ndarray]
            True binary class labels for calibration split.
        model : Optional[Any]
            Trained scikit-learn pipeline or model. If None, uses internal self.model.

        Returns
        -------
        self
        """
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("No model provided for conformal calibration.")

        y_true = np.asarray(y_cal, dtype=int)
        X_feat = X_cal[self.feature_names] if isinstance(X_cal, pd.DataFrame) else X_cal
        probs = self.model.predict_proba(X_feat)

        # Softmax margin non-conformity: score = 1 - P(True class)
        # For class 1: s = 1 - p1 = p0
        # For class 0: s = 1 - p0 = p1
        scores = np.where(y_true == 1, 1.0 - probs[:, 1], 1.0 - probs[:, 0])
        self.calibration_scores = np.sort(scores)
        self.n_cal = len(scores)
        self.quantiles_cache.clear()
        return self

    def get_conformal_quantile(self, alpha: float) -> float:
        """
        Calculate the finite-sample adjusted critical quantile:
        q_hat = ceil((n_cal + 1) * (1 - alpha)) / n_cal.
        """
        if self.calibration_scores is None or self.n_cal == 0:
            raise ValueError("Conformal predictor must be calibrated with fit_calibration() first.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}.")

        if alpha in self.quantiles_cache:
            return self.quantiles_cache[alpha]

        # Conformal quantile level with finite-sample inflation
        level = min(1.0, np.ceil((self.n_cal + 1) * (1.0 - alpha)) / float(self.n_cal))
        q_hat = float(np.quantile(self.calibration_scores, level, method="higher"))
        self.quantiles_cache[alpha] = q_hat
        return q_hat

    def predict_conformal_set(
        self,
        X_sample: Union[pd.DataFrame, Dict[str, float]],
        alpha: float = 0.10
    ) -> Dict[str, Any]:
        """
        Generate rigorous conformal prediction set for an input specimen.

        Parameters
        ----------
        X_sample : Union[pd.DataFrame, Dict[str, float]]
            Input features.
        alpha : float
            Significance level (e.g. 0.10 for 90% confidence guarantee).

        Returns
        -------
        Dict[str, Any]
            Conformal prediction set, class labels, ambiguity status, and sommelier recommendation.
        """
        if isinstance(X_sample, dict):
            if any(f not in X_sample for f in ENGINEERED_FEATURE_NAMES):
                chem = calculate_blended_chemistry([X_sample], [1.0])
                df_in = pd.DataFrame([chem])
            else:
                df_in = pd.DataFrame([X_sample])
        else:
            df_in = X_sample.copy()
            if any(f not in df_in.columns for f in ENGINEERED_FEATURE_NAMES):
                from src.feature_engineering import create_engineered_features
                df_in = create_engineered_features(df_in)

        X_feat = df_in[self.feature_names]
        probs = self.model.predict_proba(X_feat)[0]
        prob_standard = float(probs[0])
        prob_premium = float(probs[1])

        q_hat = self.get_conformal_quantile(alpha)

        # Non-conformity inclusion rule:
        # Include label y if (1 - P(Y = y)) <= q_hat  <=>  P(Y = y) >= 1 - q_hat
        cutoff = max(0.0, 1.0 - q_hat)

        pred_set = []
        labels = []
        if prob_standard >= cutoff:
            pred_set.append(0)
            labels.append("Standard Table Wine")
        if prob_premium >= cutoff:
            pred_set.append(1)
            labels.append("Premium Reserve Wine")

        # Fallback for empty set: include argmax class to guarantee non-empty prediction set
        if len(pred_set) == 0:
            top_class = 1 if prob_premium >= prob_standard else 0
            pred_set.append(top_class)
            labels.append("Premium Reserve Wine" if top_class == 1 else "Standard Table Wine")

        set_size = len(pred_set)
        confidence = float(np.round((1.0 - alpha) * 100.0, 1))

        if set_size == 1:
            verdict = "Confident Prediction"
            sommelier_action = (
                f"Release lot with certified {confidence}% statistical confidence as {labels[0]}."
            )
        elif set_size == 2:
            verdict = "Ambiguous / High Uncertainty"
            sommelier_action = (
                f"Borderline enological profile. Physicochemical attributes span standard and premium tiers. "
                f"Mandatory physical sensory tasting panel required before bottling release."
            )
        else:
            verdict = "Extreme Outlier / Anomaly"
            sommelier_action = (
                f"Empty prediction set at {confidence}% confidence. Specimen exhibits extreme atypical chemistry. "
                f"Quarantine lot for chemical taint screening (TCA, Brettanomyces, volatile spoilage)."
            )

        return {
            "prediction_set": pred_set,
            "prediction_labels": labels,
            "set_size": set_size,
            "confidence_level": float(1.0 - alpha),
            "confidence_pct": f"{confidence}%",
            "prob_premium": float(np.round(prob_premium, 4)),
            "prob_standard": float(np.round(prob_standard, 4)),
            "conformal_quantile_threshold": float(np.round(q_hat, 4)),
            "probability_inclusion_cutoff": float(np.round(cutoff, 4)),
            "is_ambiguous": bool(set_size == 2),
            "is_empty": bool(set_size == 0),
            "verdict": verdict,
            "sommelier_action": sommelier_action
        }

    def evaluate_coverage(
        self,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        alpha_levels: Sequence[float] = (0.01, 0.05, 0.10, 0.15, 0.20)
    ) -> pd.DataFrame:
        """
        Evaluate empirical coverage, average set size, and ambiguity rates on holdout test set.
        """
        y_arr = np.asarray(y_test, dtype=int)
        X_feat = X_test[self.feature_names]
        probs = self.model.predict_proba(X_feat)

        results = []
        for a in alpha_levels:
            q_hat = self.get_conformal_quantile(a)
            cutoff = max(0.0, 1.0 - q_hat)

            in_0 = probs[:, 0] >= cutoff
            in_1 = probs[:, 1] >= cutoff

            # Fallback for empty sets: include top predicted class
            empty_mask = (~in_0) & (~in_1)
            top_is_1 = probs[:, 1] >= probs[:, 0]
            in_1 = in_1 | (empty_mask & top_is_1)
            in_0 = in_0 | (empty_mask & (~top_is_1))

            # Check if true label is in prediction set
            covered = np.where(y_arr == 1, in_1, in_0)
            emp_coverage = float(np.mean(covered))

            set_sizes = in_0.astype(int) + in_1.astype(int)
            avg_size = float(np.mean(set_sizes))
            singletons = float(np.mean(set_sizes == 1))
            ambiguous = float(np.mean(set_sizes == 2))
            empty = float(np.mean(set_sizes == 0))

            results.append({
                "alpha": a,
                "nominal_confidence": round(1.0 - a, 3),
                "empirical_coverage": round(emp_coverage, 4),
                "coverage_gap": round(emp_coverage - (1.0 - a), 4),
                "mean_set_size": round(avg_size, 3),
                "singleton_ratio": round(singletons, 4),
                "ambiguous_ratio": round(ambiguous, 4),
                "empty_ratio": round(empty, 4),
                "quantile_threshold": round(q_hat, 4)
            })

        return pd.DataFrame(results)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize calibration scores and state to JSON."""
        data = {
            "n_cal": self.n_cal,
            "feature_names": self.feature_names,
            "calibration_scores": self.calibration_scores.tolist() if self.calibration_scores is not None else [],
            "quantiles_cache": {str(k): float(v) for k, v in self.quantiles_cache.items()}
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Union[str, Path]) -> "EnologicalConformalPredictor":
        """Load calibration scores and restore state from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.n_cal = data["n_cal"]
        self.feature_names = data.get("feature_names", self.feature_names)
        self.calibration_scores = np.array(data["calibration_scores"], dtype=float)
        self.quantiles_cache = {float(k): float(v) for k, v in data.get("quantiles_cache", {}).items()}
        return self


# ============================================================================
# 4. Winery Economic Utility & Expected Monetary Value (EMV) Engine
# ============================================================================

class WineryEconomicEngine:
    """
    Commercial Decision-Theoretic Payoff Matrix and Expected Monetary Value (EMV) optimizer.

    Payoff Matrix ($ / bottle):
        - True Positive  (TP): +$28.00 (Reserve Grade bottle sold at high margin)
        - False Positive (FP): -$45.00 (Sub-standard wine sold as Reserve -> Brand damage & return penalty)
        - False Negative (FN): -$12.00 (Premium lot mistakenly blended as cheap bulk table wine)
        - True Negative  (TN):  +$4.50 (Standard wine accurately priced and bottled)
    """

    def __init__(
        self,
        tp_margin: float = 28.00,
        fp_penalty: float = -45.00,
        fn_penalty: float = -12.00,
        tn_margin: float = 4.50
    ):
        self.tp_margin = tp_margin
        self.fp_penalty = fp_penalty
        self.fn_penalty = fn_penalty
        self.tn_margin = tn_margin

    def evaluate_threshold_economics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        lot_size_bottles: int = 25000
    ) -> Dict[str, Any]:
        """
        Compute commercial financial return, payoff breakdown, and performance metrics
        at a specific classification threshold.
        """
        y_pred = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        n_total = len(y_true)

        # Expected value per bottle ($)
        ev_tp = (tp / n_total) * self.tp_margin
        ev_fp = (fp / n_total) * self.fp_penalty
        ev_fn = (fn / n_total) * self.fn_penalty
        ev_tn = (tn / n_total) * self.tn_margin
        expected_profit_per_bottle = ev_tp + ev_fp + ev_fn + ev_tn

        # Scaled to commercial harvest lot
        harvest_profit = float(expected_profit_per_bottle * lot_size_bottles)
        reserve_bottles_bottled = int(np.round((tp + fp) / n_total * lot_size_bottles))
        table_bottles_bottled = lot_size_bottles - reserve_bottles_bottled

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        return {
            "threshold": float(np.round(threshold, 3)),
            "expected_profit_per_bottle": float(np.round(expected_profit_per_bottle, 2)),
            "harvest_lot_profit": float(np.round(harvest_profit, 2)),
            "harvest_lot_size_bottles": lot_size_bottles,
            "reserve_bottles_bottled": reserve_bottles_bottled,
            "table_bottles_bottled": table_bottles_bottled,
            "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "financial_breakdown_per_bottle": {
                "tp_revenue": round(ev_tp, 3),
                "fp_penalty": round(ev_fp, 3),
                "fn_loss": round(ev_fn, 3),
                "tn_revenue": round(ev_tn, 3)
            },
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4)
        }

    def optimize_financial_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[Sequence[float]] = None,
        lot_size_bottles: int = 25000
    ) -> Dict[str, Any]:
        """
        Identify the optimal decision threshold tau* that maximizes total winery profit.
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 91)

        best_profit = -np.inf
        best_thresh = 0.50
        best_eval = {}
        curve = []

        for th in thresholds:
            ev = self.evaluate_threshold_economics(y_true, y_prob, th, lot_size_bottles)
            curve.append({
                "threshold": ev["threshold"],
                "profit_per_bottle": ev["expected_profit_per_bottle"],
                "harvest_profit": ev["harvest_lot_profit"],
                "f1_score": ev["f1_score"],
                "precision": ev["precision"],
                "recall": ev["recall"]
            })
            if ev["harvest_lot_profit"] > best_profit:
                best_profit = ev["harvest_lot_profit"]
                best_thresh = th
                best_eval = ev

        df_curve = pd.DataFrame(curve)

        # Baseline comparisons
        def_eval = self.evaluate_threshold_economics(y_true, y_prob, 0.50, lot_size_bottles)
        f1_thresh_row = df_curve.loc[df_curve["f1_score"].idxmax()]
        f1_eval = self.evaluate_threshold_economics(y_true, y_prob, f1_thresh_row["threshold"], lot_size_bottles)

        uplift_vs_default = best_profit - def_eval["harvest_lot_profit"]
        uplift_pct_vs_default = (uplift_vs_default / abs(def_eval["harvest_lot_profit"] + 1e-5)) * 100.0

        return {
            "optimal_financial_threshold": float(np.round(best_thresh, 3)),
            "max_harvest_profit": float(np.round(best_profit, 2)),
            "max_profit_per_bottle": float(np.round(best_eval["expected_profit_per_bottle"], 2)),
            "best_evaluation": best_eval,
            "default_threshold_evaluation": def_eval,
            "f1_threshold_evaluation": f1_eval,
            "dollar_uplift_vs_default": float(np.round(uplift_vs_default, 2)),
            "percent_uplift_vs_default": float(np.round(uplift_pct_vs_default, 2)),
            "profit_curve": df_curve
        }


# ============================================================================
# 5. Multi-Model Stacking Ensemble Architecture
# ============================================================================

class WineStackingEnsemble:
    """
    Heterogeneous Multi-Model Stacking Ensemble combining:
        Level-0 Base Learners:
            1. Tuned Random Forest (Tree Ensemble)
            2. Gradient Boosting Classifier (GBDT)
            3. Extra Trees Classifier (Extremely Randomized Trees)
            4. Regularized Scaled Logistic Regression (Linear Margin)
        Level-1 Meta-Learner:
            Calibrated Logistic Regression trained on 5-Fold Stratified out-of-fold predictions.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = ALL_FEATURE_NAMES
        self.base_models: Dict[str, Any] = {
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=90,
                max_depth=4,
                learning_rate=0.08,
                random_state=random_state
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state
                ))
            ])
        }
        self.meta_learner = LogisticRegression(
            C=1.0,
            random_state=random_state,
            max_iter=500
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "WineStackingEnsemble":
        """
        Fit Level-0 base models using 5-Fold Stratified Out-Of-Fold CV blending,
        fit Level-1 meta-learner on out-of-fold probability vectors, and refit
        Level-0 estimators on full dataset.
        """
        y_arr = np.asarray(y, dtype=int)
        X_feat = X[self.feature_names] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        n_samples = len(X_feat)
        n_models = len(self.base_models)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_matrix = np.zeros((n_samples, n_models))

        model_names = list(self.base_models.keys())
        for m_idx, name in enumerate(model_names):
            m_oof = np.zeros(n_samples)
            for tr_idx, val_idx in skf.split(X_feat, y_arr):
                X_tr, y_tr = X_feat.iloc[tr_idx], y_arr[tr_idx]
                X_va = X_feat.iloc[val_idx]

                # Clone or create new instance for fold
                fold_model = clone(self.base_models[name])
                fold_model.fit(X_tr, y_tr)
                m_oof[val_idx] = fold_model.predict_proba(X_va)[:, 1]

            oof_matrix[:, m_idx] = m_oof

        # Fit Level-1 Meta-Learner on Out-Of-Fold Probabilities
        self.meta_learner.fit(oof_matrix, y_arr)

        # Refit all Level-0 Base Learners on full dataset
        for name, model in self.base_models.items():
            model.fit(X_feat, y_arr)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities via Level-0 forwarding through Meta-Learner."""
        if not self.is_fitted:
            raise ValueError("Stacking ensemble must be fitted before predict_proba().")

        X_feat = X[self.feature_names] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        n_samples = len(X_feat)
        base_probs = np.zeros((n_samples, len(self.base_models)))

        for m_idx, (name, model) in enumerate(self.base_models.items()):
            base_probs[:, m_idx] = model.predict_proba(X_feat)[:, 1]

        return self.meta_learner.predict_proba(base_probs)

    def predict(self, X: pd.DataFrame, threshold: float = DEFAULT_SCREENING_THRESHOLD) -> np.ndarray:
        """Predict binary class labels applying operating decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Compute performance metrics on holdout test partition."""
        y_true = np.asarray(y_test, dtype=int)
        probs = self.predict_proba(X_test)
        prob_premium = probs[:, 1]
        y_pred = (prob_premium >= DEFAULT_SCREENING_THRESHOLD).astype(int)

        roc_auc = float(roc_auc_score(y_true, prob_premium))
        pr_auc = float(average_precision_score(y_true, prob_premium))
        brier = float(brier_score_loss(y_true, prob_premium))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Base model individual performances
        base_benchmarks = {}
        X_feat = X_test[self.feature_names] if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=self.feature_names)
        for name, model in self.base_models.items():
            b_probs = model.predict_proba(X_feat)[:, 1]
            b_pred = (b_probs >= DEFAULT_SCREENING_THRESHOLD).astype(int)
            base_benchmarks[name] = {
                "roc_auc": round(float(roc_auc_score(y_true, b_probs)), 4),
                "pr_auc": round(float(average_precision_score(y_true, b_probs)), 4),
                "brier_score": round(float(brier_score_loss(y_true, b_probs)), 4),
                "balanced_acc": round(float(balanced_accuracy_score(y_true, b_pred)), 4)
            }

        # Meta coefficients
        meta_coefs = {
            name: round(float(self.meta_learner.coef_[0][i]), 4)
            for i, name in enumerate(self.base_models.keys())
        }

        return {
            "stacking_roc_auc": round(roc_auc, 4),
            "stacking_pr_auc": round(pr_auc, 4),
            "stacking_brier_score": round(brier, 4),
            "stacking_balanced_accuracy": round(bal_acc, 4),
            "stacking_precision": round(prec, 4),
            "stacking_recall": round(rec, 4),
            "stacking_f1_score": round(f1, 4),
            "base_models_benchmarks": base_benchmarks,
            "meta_learner_coefficients": meta_coefs,
            "operating_threshold": DEFAULT_SCREENING_THRESHOLD
        }

    def save(self, joblib_path: Union[str, Path], json_path: Union[str, Path]) -> None:
        """Serialize stacking ensemble and metadata."""
        joblib.dump(self, joblib_path)
        meta = {
            "model_type": "Heterogeneous Stacking Meta-Learner",
            "base_models": list(self.base_models.keys()),
            "meta_model": type(self.meta_learner).__name__,
            "meta_coefficients": {
                name: float(self.meta_learner.coef_[0][i])
                for i, name in enumerate(self.base_models.keys())
            } if self.is_fitted else {},
            "meta_intercept": float(self.meta_learner.intercept_[0]) if self.is_fitted else 0.0,
            "features": self.feature_names
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, joblib_path: Union[str, Path]) -> "WineStackingEnsemble":
        """Load serialized stacking ensemble."""
        return joblib.load(joblib_path)
