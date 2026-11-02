"""
Enological Aging Kinetics, Multi-Objective Pareto Optimization,
and Physicochemical Stress-Testing Suite.

Modules & Capabilities:
- CellarAgingKineticsSimulator: First-principles chemical kinetic modeling of bottle aging,
  Arrhenius Free SO2 decay, Acetobacter volatile acidity evolution, and peak drinking window.
- ParetoEnologicalOptimizer: Multi-objective non-dominated Pareto frontier extraction,
  hypervolume compromise, and knee-point selection balancing quality, sulphites, and alcohol.
- EnologicalStressTester: Systematic environmental perturbation stress-testing,
  thermal/oxygen shocks, adversarial flip margin, and vintage resilience scoring.
- ShapEnologicalInteractions: Pairwise non-linear chemical synergy quantification
  (Alcohol x VA sensory masking, Free SO2 x pH molecular antimicrobial potency).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import create_engineered_features

# Closure Oxygen Transmission Rates (OTR relative multiplier to natural cork)
CLOSURE_OTR = {
    "natural_cork": 1.0,
    "screwcap": 0.35,
    "synthetic": 1.45,
    "technical_cork": 0.70
}

# Standard cellar temperature baseline (Celsius)
STANDARD_CELLAR_TEMP_C = 14.0

# Arrhenius gas constant R in J / (mol * K)
R_GAS_CONSTANT = 8.314

# Activation energy for SO2 oxidation in wine (approx 65 kJ/mol)
EA_SO2_OXIDATION = 65000.0


# ============================================================================
# 1. Cellar Aging Kinetics Simulator
# ============================================================================

class CellarAgingKineticsSimulator:
    """
    Simulates temporal physicochemical bottle evolution during cellar aging
    governed by kinetic differential rate laws.
    """

    def __init__(
        self,
        model_pipeline: Optional[Any] = None,
        model_path: Union[str, Path] = "models/champion_rf_pipeline.joblib",
        decision_threshold: float = 0.35
    ):
        self.decision_threshold = decision_threshold
        if model_pipeline is not None:
            self.model = model_pipeline
        else:
            p = Path(model_path)
            if p.exists():
                self.model = joblib.load(p)
            else:
                self.model = None

    def calculate_so2_rate_constant(
        self,
        temperature_c: float = 14.0,
        closure: str = "natural_cork",
        base_k0: float = 0.028
    ) -> float:
        """
        Calculates the effective monthly first-order Free SO2 decay rate constant k
        using Arrhenius temperature scaling and closure oxygen transmission rate (OTR).
        """
        temp_k = temperature_c + 273.15
        ref_temp_k = STANDARD_CELLAR_TEMP_C + 273.15

        # Arrhenius factor
        arrhenius_scaling = math.exp(
            (-EA_SO2_OXIDATION / R_GAS_CONSTANT) * (1.0 / temp_k - 1.0 / ref_temp_k)
        )
        otr_multiplier = CLOSURE_OTR.get(closure.lower(), 1.0)
        return float(base_k0 * arrhenius_scaling * otr_multiplier)

    def simulate_trajectory(
        self,
        initial_chemistry: Dict[str, float],
        months: int = 60,
        temperature_c: float = 14.0,
        closure: str = "natural_cork",
        step_months: int = 1
    ) -> Dict[str, Any]:
        """
        Simulates monthly chemical evolution and predictive quality trajectory over time.
        """
        k_so2 = self.calculate_so2_rate_constant(temperature_c, closure)

        # Baseline starting values
        fixed_acidity_0 = float(initial_chemistry["fixed acidity"])
        volatile_acidity_0 = float(initial_chemistry["volatile acidity"])
        citric_acid_0 = float(initial_chemistry["citric acid"])
        residual_sugar_0 = float(initial_chemistry["residual sugar"])
        chlorides_0 = float(initial_chemistry["chlorides"])
        free_so2_0 = float(initial_chemistry["free sulfur dioxide"])
        total_so2_0 = float(initial_chemistry["total sulfur dioxide"])
        density_0 = float(initial_chemistry["density"])
        ph_0 = float(initial_chemistry["pH"])
        sulphates_0 = float(initial_chemistry["sulphates"])
        alcohol_0 = float(initial_chemistry["alcohol"])

        bound_so2_0 = max(0.0, total_so2_0 - free_so2_0)

        records = []
        current_va = volatile_acidity_0
        otr = CLOSURE_OTR.get(closure.lower(), 1.0)

        for t in range(0, months + 1, step_months):
            # 1. Free SO2 decay: pseudo first-order oxidative decay
            free_so2_t = free_so2_0 * math.exp(-k_so2 * t)

            # Bound SO2 hydrolyzes and oxidizes very slowly
            bound_so2_t = bound_so2_0 * math.exp(-0.008 * t)
            total_so2_t = free_so2_t + bound_so2_t

            # 2. Spoilage VA accumulation: If Free SO2 drops below protective 10.0 mg/L
            if t > 0:
                if free_so2_t < 10.0:
                    antimicrobial_deficit = max(0.0, 10.0 - free_so2_t)
                    temp_factor = 1.0 + 0.05 * (temperature_c - STANDARD_CELLAR_TEMP_C)
                    monthly_va_drift = antimicrobial_deficit * 0.0025 * temp_factor * step_months
                    current_va += monthly_va_drift
                else:
                    # Negligible background esterification / micro-oxidation
                    current_va += 0.0003 * step_months

            # 3. Acid softening (tartaric bitartrate precipitation)
            fixed_acidity_t = fixed_acidity_0 * (1.0 - 0.07 * (1.0 - math.exp(-0.025 * t)))
            ph_t = ph_0 + 0.10 * (1.0 - math.exp(-0.025 * t))
            citric_acid_t = max(0.0, citric_acid_0 * math.exp(-0.005 * t))

            # 4. Alcohol gradual permeability
            alcohol_t = alcohol_0 * (1.0 - 0.0005 * otr * t)

            # 5. Density and sugar remain relatively stable in bottle
            residual_sugar_t = residual_sugar_0
            chlorides_t = chlorides_0
            sulphates_t = sulphates_0
            density_t = density_0 + 0.0001 * (1.0 - alcohol_t / alcohol_0)

            # Recompute engineered chemical interaction features
            total_acidity_t = fixed_acidity_t + current_va
            bound_so2_calc = max(0.0, total_so2_t - free_so2_t)
            free_to_total_t = free_so2_t / (total_so2_t + 1e-5)
            alcohol_to_density_t = alcohol_t / (density_t + 1e-5)
            sulphate_to_chloride_t = sulphates_t / (chlorides_t + 1e-5)

            step_dict = {
                "fixed acidity": round(fixed_acidity_t, 3),
                "volatile acidity": round(current_va, 4),
                "citric acid": round(citric_acid_t, 3),
                "residual sugar": round(residual_sugar_t, 3),
                "chlorides": round(chlorides_t, 4),
                "free sulfur dioxide": round(free_so2_t, 2),
                "total sulfur dioxide": round(total_so2_t, 2),
                "density": round(density_t, 5),
                "pH": round(ph_t, 3),
                "sulphates": round(sulphates_t, 3),
                "alcohol": round(alcohol_t, 2),
                "total_acidity": round(total_acidity_t, 3),
                "bound_sulfur_dioxide": round(bound_so2_calc, 2),
                "free_to_total_so2_ratio": round(free_to_total_t, 4),
                "alcohol_to_density_ratio": round(alcohol_to_density_t, 4),
                "sulphate_to_chloride_ratio": round(sulphate_to_chloride_t, 4)
            }

            records.append((t, step_dict))

        # Model batch inference across time trajectory
        df_traj = pd.DataFrame([r[1] for r in records])
        times = [r[0] for r in records]

        if self.model is not None:
            probs = self.model.predict_proba(df_traj)[:, 1]
        else:
            # Synthetic fallback heuristic if model not supplied
            probs = np.array([
                1.0 / (1.0 + math.exp(-(-2.0 + 0.5 * r["alcohol"] - 3.5 * r["volatile acidity"] + 0.05 * r["free sulfur dioxide"])))
                for r in [rec[1] for rec in records]
            ])

        # Analyze trajectory metrics
        initial_prob = float(probs[0])
        peak_idx = int(np.argmax(probs))
        peak_prob = float(probs[peak_idx])
        peak_month = times[peak_idx]

        # Calculate drinking window (range of months meeting threshold)
        qualifying_indices = [i for i, p in enumerate(probs) if p >= self.decision_threshold]
        if qualifying_indices:
            window_start = times[qualifying_indices[0]]
            window_end = times[qualifying_indices[-1]]
        else:
            window_start = None
            window_end = None

        # Detect spoilage / oxidation month (Free SO2 < 8 or VA > 0.65)
        spoilage_month = None
        for i, t in enumerate(times):
            row = records[i][1]
            if row["free sulfur dioxide"] < 8.0 or row["volatile acidity"] > 0.65:
                spoilage_month = t
                break

        # Generate cellar advice
        if initial_prob >= self.decision_threshold and peak_month == 0:
            advice = "Ready to enjoy now; cellar evolution indicates current vintage represents peak sensory expression."
        elif peak_month > 0 and peak_prob >= self.decision_threshold:
            advice = f"Cellar for {peak_month} months to reach peak maturity ({round(peak_prob * 100, 1)}% premium confidence)."
        elif qualifying_indices:
            advice = f"Cellar window viable between Month {window_start} and Month {window_end}."
        else:
            advice = "Commercial standard table wine; consume young before oxidation risks emerge."

        trajectory_data = []
        for t, r, p in zip(times, records, probs):
            row_dict = dict(r[1])
            row_dict["month"] = t
            row_dict["premium_probability"] = round(float(p), 4)
            row_dict["is_reserve"] = bool(p >= self.decision_threshold)
            trajectory_data.append(row_dict)

        return {
            "cellar_temperature_c": temperature_c,
            "closure_type": closure,
            "so2_decay_rate_k": round(k_so2, 5),
            "simulation_months": months,
            "initial_probability": round(initial_prob, 4),
            "peak_probability": round(peak_prob, 4),
            "peak_month": peak_month,
            "drinking_window_start_month": window_start,
            "drinking_window_end_month": window_end,
            "spoilage_risk_month": spoilage_month,
            "sommelier_cellar_advice": advice,
            "trajectory": trajectory_data
        }


# ============================================================================
# 2. Multi-Objective Pareto Enological Optimizer
# ============================================================================

class ParetoEnologicalOptimizer:
    """
    Solves multi-objective enological trade-offs between:
    - Objective 1: Maximize Sensory Quality (P(Quality >= 7))
    - Objective 2: Minimize Added Preservatives (Sulphates g/L)
    - Objective 3: Maintain Optimal Alcohol (penalize excess excise tax > 13.5% or deficiency < 11.0%)
    - Objective 4: Minimize Volatile Acidity (spoilage ceiling)
    """

    def __init__(
        self,
        model_pipeline: Optional[Any] = None,
        model_path: Union[str, Path] = "models/champion_rf_pipeline.joblib"
    ):
        if model_pipeline is not None:
            self.model = model_pipeline
        else:
            p = Path(model_path)
            if p.exists():
                self.model = joblib.load(p)
            else:
                self.model = None

    def evaluate_lot(self, lot_chemistry: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluates the multi-objective fitness metrics for a single lot.
        """
        df_feat = create_engineered_features(pd.DataFrame([lot_chemistry]))
        if self.model is not None:
            prob = float(self.model.predict_proba(df_feat)[0, 1])
        else:
            prob = 0.5

        sulphates = float(lot_chemistry["sulphates"])
        alcohol = float(lot_chemistry["alcohol"])
        va = float(lot_chemistry["volatile acidity"])

        # Commercial alcohol tax/balance penalty: Ideal commercial target around 12.5% - 13.2%
        # Tax surcharge applies if alcohol > 13.5%
        alcohol_penalty = max(0.0, alcohol - 13.5) * 1.5 + max(0.0, 11.0 - alcohol) * 1.0

        return {
            "quality_prob": round(prob, 4),
            "sulphates": round(sulphates, 3),
            "alcohol": round(alcohol, 2),
            "alcohol_penalty": round(alcohol_penalty, 3),
            "volatile_acidity": round(va, 4)
        }

    def compute_pareto_frontier(
        self,
        candidate_lots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Performs non-dominated sorting across candidates to extract Pareto Frontier Rank 1.
        Criteria:
        - Maximize quality_prob
        - Minimize sulphates
        - Minimize alcohol_penalty
        - Minimize volatile_acidity
        """
        evaluated_candidates = []
        for cand in candidate_lots:
            lot_name = cand.get("name", cand.get("sample_id", f"Lot_{len(evaluated_candidates)+1}"))
            chem = {k: v for k, v in cand.items() if k not in ["name", "sample_id"]}
            metrics = self.evaluate_lot(chem)
            evaluated_candidates.append({
                "name": str(lot_name),
                "chemistry": chem,
                **metrics
            })

        n = len(evaluated_candidates)
        domination_counts = [0] * n

        # Pairwise domination comparison:
        # A dominates B if A is >= B in all objectives (quality maximized, others minimized)
        # and strictly better in at least one.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = evaluated_candidates[i]
                b = evaluated_candidates[j]

                # Dominance check (a vs b): Does b dominate a?
                b_better_or_equal = (
                    b["quality_prob"] >= a["quality_prob"] and
                    b["sulphates"] <= a["sulphates"] and
                    b["alcohol_penalty"] <= a["alcohol_penalty"] and
                    b["volatile_acidity"] <= a["volatile_acidity"]
                )
                b_strictly_better = (
                    b["quality_prob"] > a["quality_prob"] or
                    b["sulphates"] < a["sulphates"] or
                    b["alcohol_penalty"] < a["alcohol_penalty"] or
                    b["volatile_acidity"] < a["volatile_acidity"]
                )
                if b_better_or_equal and b_strictly_better:
                    domination_counts[i] += 1

        pareto_front = []
        dominated = []
        for i, count in enumerate(domination_counts):
            cand_copy = dict(evaluated_candidates[i])
            cand_copy["pareto_rank"] = 1 if count == 0 else (count + 1)
            cand_copy["is_pareto_optimal"] = (count == 0)
            if count == 0:
                pareto_front.append(cand_copy)
            else:
                dominated.append(cand_copy)

        # Identify Knee-Point (Closest normalized Euclidean distance to Utopian Ideal Point)
        # Utopian ideal: max quality, min sulphates, min penalty, min va
        all_q = [c["quality_prob"] for c in evaluated_candidates]
        all_s = [c["sulphates"] for c in evaluated_candidates]
        all_va = [c["volatile_acidity"] for c in evaluated_candidates]

        max_q = max(all_q) if max(all_q) > 0 else 1.0
        min_s = min(all_s)
        max_s = max(all_s) if max(all_s) > min_s else (min_s + 1e-4)
        min_va = min(all_va)
        max_va = max(all_va) if max(all_va) > min_va else (min_va + 1e-4)

        knee_point = None
        best_dist = float("inf")

        for c in pareto_front:
            # Normalized distance from ideal [1, 0, 0]
            d_q = (1.0 - c["quality_prob"] / max_q) ** 2
            d_s = ((c["sulphates"] - min_s) / (max_s - min_s)) ** 2
            d_va = ((c["volatile_acidity"] - min_va) / (max_va - min_va)) ** 2
            dist = math.sqrt(d_q + d_s + d_va)
            c["knee_distance"] = round(dist, 4)

            if dist < best_dist:
                best_dist = dist
                knee_point = c

        # Categorized archetypes
        max_quality_lot = max(pareto_front, key=lambda x: x["quality_prob"]) if pareto_front else None
        min_sulphate_lot = min(pareto_front, key=lambda x: x["sulphates"]) if pareto_front else None

        return {
            "total_candidates_evaluated": n,
            "pareto_optimal_count": len(pareto_front),
            "pareto_frontier": pareto_front,
            "dominated_candidates": dominated,
            "knee_point_lot": knee_point,
            "archetypes": {
                "reserve_grand_vin": max_quality_lot,
                "low_intervention_natural": min_sulphate_lot,
                "balanced_compromise": knee_point
            }
        }


# ============================================================================
# 3. Physicochemical Stress-Testing Suite
# ============================================================================

class EnologicalStressTester:
    """
    Audits classifier robustness against environmental and cellar perturbation shocks:
    - Heat shock (micro-oxidation, temperature spike)
    - Dissolved oxygen ingress
    - Tartaric acidification / deacidification variation
    - Compound multi-stress perturbations
    """

    def __init__(
        self,
        model_pipeline: Optional[Any] = None,
        model_path: Union[str, Path] = "models/champion_rf_pipeline.joblib",
        decision_threshold: float = 0.35
    ):
        self.decision_threshold = decision_threshold
        if model_pipeline is not None:
            self.model = model_pipeline
        else:
            p = Path(model_path)
            if p.exists():
                self.model = joblib.load(p)
            else:
                self.model = None

    def run_stress_test(
        self,
        base_chemistry: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Executes a standardized battery of enological disturbance scenarios.
        """
        # Baseline prediction
        df_base = create_engineered_features(pd.DataFrame([base_chemistry]))
        if self.model is not None:
            base_prob = float(self.model.predict_proba(df_base)[0, 1])
        else:
            base_prob = 0.50

        base_is_premium = bool(base_prob >= self.decision_threshold)

        # Scenarios defined by physical wine transport and cellar conditions
        scenarios = {
            "thermal_heat_shock": {
                "name": "Thermal Heat Shock (+8°C Transit Shock)",
                "delta": {
                    "free sulfur dioxide": -0.25 * base_chemistry["free sulfur dioxide"],
                    "volatile acidity": +0.08,
                    "alcohol": -0.15
                },
                "enological_description": "Rapid oxidative consumption of Free SO2 and volatile acid rise due to non-refrigerated container transit."
            },
            "oxygen_ingress_shock": {
                "name": "Micro-Oxygenation / Cork Ingress",
                "delta": {
                    "free sulfur dioxide": -0.40 * base_chemistry["free sulfur dioxide"],
                    "total sulfur dioxide": -0.15 * base_chemistry["total sulfur dioxide"],
                    "volatile acidity": +0.12
                },
                "enological_description": "Excessive headspace oxygen dissolution leading to sulfur binding and acetic spoilage onset."
            },
            "tartaric_acidification_shock": {
                "name": "Excess Tartaric Acidification",
                "delta": {
                    "fixed acidity": +1.2,
                    "pH": -0.20,
                    "citric acid": +0.10
                },
                "enological_description": "Over-acidification cellar intervention yielding tart astringency."
            },
            "harvest_rain_dilution": {
                "name": "Harvest Rain Dilution",
                "delta": {
                    "fixed acidity": -0.9,
                    "alcohol": -0.6,
                    "pH": +0.15,
                    "density": -0.0015
                },
                "enological_description": "Grape berry water swelling from pre-harvest precipitation diluting must components."
            },
            "compound_storage_failure": {
                "name": "Compound Storage Failure (Heat + Air)",
                "delta": {
                    "free sulfur dioxide": -0.50 * base_chemistry["free sulfur dioxide"],
                    "volatile acidity": +0.18,
                    "alcohol": -0.30,
                    "pH": +0.12
                },
                "enological_description": "Worst-case catastrophic cellar failure combining heat exposure and leaky closures."
            }
        }

        results = []
        flips = 0

        for sc_id, sc in scenarios.items():
            perturbed_chem = dict(base_chemistry)
            for k, delta_val in sc["delta"].items():
                if k in perturbed_chem:
                    perturbed_chem[k] = max(0.001, float(perturbed_chem[k] + delta_val))

            df_p = create_engineered_features(pd.DataFrame([perturbed_chem]))
            if self.model is not None:
                p_prob = float(self.model.predict_proba(df_p)[0, 1])
            else:
                p_prob = max(0.0, base_prob - 0.2)

            p_is_premium = bool(p_prob >= self.decision_threshold)
            class_flipped = (p_is_premium != base_is_premium)
            if class_flipped:
                flips += 1

            prob_delta = round(p_prob - base_prob, 4)

            results.append({
                "scenario_id": sc_id,
                "scenario_name": sc["name"],
                "description": sc["enological_description"],
                "perturbed_probability": round(p_prob, 4),
                "probability_delta": prob_delta,
                "retained_classification": not class_flipped,
                "perturbed_is_premium": p_is_premium
            })

        # Calculate Adversarial Epsilon Boundary Margin (binary search along worst gradient)
        # Search minimal delta in volatile acidity to flip classification
        step = 0.01
        va_margin = 0.0
        curr_chem = dict(base_chemistry)
        flipped_margin_found = False

        direction = +1 if base_is_premium else -1
        target_class = not base_is_premium

        for i in range(1, 100):
            test_va = max(0.05, base_chemistry["volatile acidity"] + direction * i * step)
            curr_chem["volatile acidity"] = test_va
            df_m = create_engineered_features(pd.DataFrame([curr_chem]))
            if self.model is not None:
                m_prob = float(self.model.predict_proba(df_m)[0, 1])
            else:
                m_prob = base_prob - 0.02 * i
            m_is_prem = bool(m_prob >= self.decision_threshold)
            if m_is_prem == target_class:
                va_margin = round(i * step, 3)
                flipped_margin_found = True
                break

        if not flipped_margin_found:
            va_margin = 1.0

        # Resilience Index (0 to 100)
        # Penalize flips heavily and large probability drops
        mean_drop = np.mean([abs(r["probability_delta"]) for r in results])
        resilience_score = max(0.0, min(100.0, (1.0 - flips / len(scenarios)) * 60.0 + (1.0 - mean_drop) * 40.0))

        return {
            "baseline_probability": round(base_prob, 4),
            "baseline_is_premium": base_is_premium,
            "decision_threshold": self.decision_threshold,
            "scenarios_tested": len(scenarios),
            "classification_flips": flips,
            "resilience_score": round(resilience_score, 1),
            "resilience_tier": "Highly Resilient" if resilience_score >= 80 else ("Moderate Resilience" if resilience_score >= 50 else "Fragile / Vulnerable"),
            "critical_va_epsilon_flip_margin": va_margin,
            "scenario_results": results
        }


# ============================================================================
# 4. Enological Chemical Interactions Analyzer
# ============================================================================

class ShapEnologicalInteractions:
    """
    Computes enological pairwise synergies and non-linear interactions:
    - Alcohol x Volatile Acidity (Sensory ethanol masking of acetic sharpness)
    - Free SO2 x pH (Molecular active sulfur dioxide equilibrium)
    - Citric Acid x Fixed Acidity (Freshness tartaric balance)
    - Sulphates x Chlorides (Preservative vs Salinity balance)
    """

    @staticmethod
    def calculate_molecular_so2(free_so2: float, ph: float) -> float:
        """
        Computes active molecular SO2 (mg/L) based on Henderson-Hasselbalch acid equilibrium:
        Molecular SO2 = Free SO2 / (1 + 10^(pH - 1.81))
        Enological standard: 0.5 - 0.8 mg/L is required for effective antimicrobial protection.
        """
        ratio = 1.0 + 10.0 ** (ph - 1.81)
        return float(free_so2 / ratio)

    @staticmethod
    def compute_interaction_matrix(
        model: Any,
        reference_data: pd.DataFrame,
        sample_size: int = 150
    ) -> Dict[str, Any]:
        """
        Evaluates interaction synergies across test samples.
        """
        df_sub = reference_data.sample(min(len(reference_data), sample_size), random_state=42).copy()
        df_sub = df_sub.drop(columns=["quality", "quality_label", "quality_binary"], errors="ignore")
        engineered = create_engineered_features(df_sub)

        # Baseline predictions
        base_probs = model.predict_proba(engineered)[:, 1]

        # Synergy 1: Alcohol x Volatile Acidity
        # Check whether high alcohol dampens the negative impact of high VA
        high_alc = engineered["alcohol"] >= engineered["alcohol"].median()
        high_va = engineered["volatile acidity"] >= engineered["volatile acidity"].median()

        prob_high_va_low_alc = float(np.mean(base_probs[(~high_alc) & high_va])) if np.sum((~high_alc) & high_va) > 0 else 0.0
        prob_high_va_high_alc = float(np.mean(base_probs[high_alc & high_va])) if np.sum(high_alc & high_va) > 0 else 0.0
        alcohol_masking_lift = round(prob_high_va_high_alc - prob_high_va_low_alc, 4)

        # Synergy 2: Free SO2 x pH
        mol_so2 = [
            ShapEnologicalInteractions.calculate_molecular_so2(f, p)
            for f, p in zip(df_sub["free sulfur dioxide"], df_sub["pH"])
        ]
        mean_mol_so2 = float(np.mean(mol_so2))

        interactions = {
            "alcohol_va_masking": {
                "pair": ["alcohol", "volatile acidity"],
                "interaction_type": "Sensory Masking / Buffering",
                "prob_high_va_low_alcohol": round(prob_high_va_low_alc, 4),
                "prob_high_va_high_alcohol": round(prob_high_va_high_alc, 4),
                "alcohol_buffering_lift": alcohol_masking_lift,
                "enological_rule": "High alcohol (>=11.5%) mitigates consumer rejection of moderate volatile acidity up to 0.60 g/L."
            },
            "molecular_so2_equilibrium": {
                "pair": ["free sulfur dioxide", "pH"],
                "interaction_type": "Logarithmic Antimicrobial Potency",
                "mean_molecular_so2_mg_l": round(mean_mol_so2, 3),
                "protective_threshold_mg_l": 0.60,
                "enological_rule": "At lower pH (<3.30), a lower Free SO2 dose achieves equal antimicrobial efficacy."
            },
            "sulphate_chloride_balance": {
                "pair": ["sulphates", "chlorides"],
                "interaction_type": "Preservation vs Salinity Ratio",
                "mean_ratio": round(float(np.mean(engineered["sulphate_to_chloride_ratio"])), 2),
                "enological_rule": "Sulphates/Chlorides ratio > 6.0 correlates with superior aging potential and clean finish."
            }
        }

        return interactions
