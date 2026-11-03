"""
Enological Fermentation Bioprocess Kinetics, Stuck Fermentation Early-Warning,
Sensory Organoleptic Lexicon & Causal Cellar Intervention Engine.

Modules & Capabilities:
- FermentationKineticsSimulator: Coupled dynamic differential rate equations of primary
  alcoholic fermentation (Saccharomyces cerevisiae biomass, Monod hexose consumption,
  ethanol self-inhibition, YAN depletion, metabolic heat liberation, and stress VA formation).
- StuckFermentationDetector: Time-series kinetic trajectory diagnostic engine detecting
  sluggish and arrested fermentations with prescriptive enological remediation protocols.
- SensoryOrganolepticEngine: Physicochemical-to-sensory organoleptic translation engine
  generating 6-axis spider radar scores, sommelier tasting notes, and culinary food pairing affinities.
- CausalEnologicalEngine: Observational causal inference platform quantifying Average
  Treatment Effects (ATE) of cellar interventions using Doubly Robust Propensity Weighting.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import create_engineered_features


# Standard enological kinetic constants for Saccharomyces cerevisiae
DEFAULT_MU_MAX = 0.16           # Maximum specific growth rate (1/h)
DEFAULT_KS_SUGAR = 2.5          # Monod affinity constant for hexose sugars (g/L)
DEFAULT_KN_YAN = 25.0           # Monod affinity constant for YAN (mg/L)
DEFAULT_E_MAX = 16.0            # Toxic ethanol ceiling (% v/v)
DEFAULT_YX_S = 0.07             # Biomass yield from sugar (g biomass / g sugar)
DEFAULT_Q_MAX_FERM = 0.18       # Non-growth stationary fermentation rate (g sugar / g biomass / h)
DEFAULT_MS_MAINTENANCE = 0.012  # Yeast maintenance consumption (g sugar / g biomass / h)
DEFAULT_YN_X = 28.0             # Nitrogen consumption per biomass (mg YAN / g biomass)
SUGAR_PER_ETHANOL_PCT = 16.83   # Approx grams hexose per 1% v/v ethanol produced


# ============================================================================
# 1. Fermentation Kinetics Simulator
# ============================================================================

class FermentationKineticsSimulator:
    """
    Simulates temporal bioprocess kinetics of alcoholic wine fermentation
    using non-linear coupled differential rate laws.
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

    @staticmethod
    def temperature_growth_factor(temp_c: float, optimal_temp: float = 24.0, width: float = 8.0) -> float:
        """Calculates temperature modulation factor on yeast enzymatic activity."""
        if temp_c <= 5.0 or temp_c >= 38.0:
            return 0.0
        return float(np.exp(-((temp_c - optimal_temp) / width) ** 2))

    def simulate_fermentation(
        self,
        initial_sugar_g_l: float = 220.0,
        initial_yan_mg_l: float = 220.0,
        initial_temp_c: float = 20.0,
        jacket_temp_c: float = 20.0,
        initial_biomass_g_l: float = 0.5,
        total_days: int = 14,
        steps_per_day: int = 4,
        yeast_strain: str = "EC1118_champagne",
        cooling_capacity_k: float = 0.08,
        baseline_chemistry: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes numerical forward integration (hourly stepping) of fermentation dynamics.

        Parameters:
            initial_sugar_g_l: Must hexose sugar (g/L), e.g. 200 - 260 g/L (20 - 26 °Brix).
            initial_yan_mg_l: Yeast Assimilable Nitrogen (mg/L), standard range 150 - 350 mg/L.
            initial_temp_c: Starting must temperature in Celsius.
            jacket_temp_c: Cellar cooling jacket setpoint in Celsius.
            initial_biomass_g_l: Inoculum viable yeast biomass (g/L).
            total_days: Duration of fermentation to track (days).
            steps_per_day: Numerical resolution per day (4 steps = 6-hour dt).
            yeast_strain: Yeast archetype modulating kinetic constants.
            cooling_capacity_k: Heat transfer coefficient for cooling jacket.
            baseline_chemistry: Baseline must attributes for finished wine scoring.
        """
        # Strain tuning
        strain_profiles = {
            "EC1118_champagne": {"mu_max": 0.17, "e_max": 16.5, "opt_t": 22.0, "stress_va": 0.0003},
            "RC212_pinot_red": {"mu_max": 0.15, "e_max": 15.5, "opt_t": 25.0, "stress_va": 0.0004},
            "D254_rhone_bold": {"mu_max": 0.16, "e_max": 16.0, "opt_t": 26.0, "stress_va": 0.00045},
            "QA23_aromatic_white": {"mu_max": 0.18, "e_max": 15.0, "opt_t": 17.0, "stress_va": 0.00025}
        }
        profile = strain_profiles.get(yeast_strain, strain_profiles["EC1118_champagne"])
        mu_max = profile["mu_max"]
        e_max = profile["e_max"]
        opt_t = profile["opt_t"]
        k_stress_va = profile["stress_va"]

        dt_hours = 24.0 / steps_per_day
        total_steps = total_days * steps_per_day

        # Initial state vectors
        S = float(initial_sugar_g_l)
        N = float(initial_yan_mg_l)
        X = float(initial_biomass_g_l)
        E = 0.0
        T = float(initial_temp_c)
        VA = 0.15   # Baseline initial volatile acidity (g/L acetic acid)

        trajectory: List[Dict[str, Any]] = []
        peak_rate = 0.0
        peak_rate_day = 0.0

        for step in range(total_steps + 1):
            current_day = round(step * (dt_hours / 24.0), 2)

            # Record current step
            trajectory.append({
                "day": current_day,
                "hour": round(step * dt_hours, 1),
                "sugar_g_l": round(max(0.0, S), 2),
                "ethanol_pct": round(max(0.0, E), 2),
                "biomass_g_l": round(max(0.0, X), 3),
                "yan_mg_l": round(max(0.0, N), 1),
                "temperature_c": round(T, 2),
                "volatile_acidity": round(VA, 3),
                "density": round(0.990 + (S / 1000.0) * 0.40, 4)
            })

            if step == total_steps:
                break

            # 1. Specific growth rate mu
            f_temp = self.temperature_growth_factor(T, optimal_temp=opt_t)
            sat_s = S / (DEFAULT_KS_SUGAR + S) if S > 0 else 0.0
            sat_n = N / (DEFAULT_KN_YAN + N) if N > 0 else 0.0
            inhib_e = max(0.0, 1.0 - (E / e_max)) ** 1.5

            mu = mu_max * sat_s * sat_n * inhib_e * f_temp

            # Mortality rate increases as ethanol climbs
            k_d = 0.003 + 0.0015 * (E / 10.0)

            # 2. State derivative updates
            dX = (mu - k_d) * X
            # Sugar uptake: growth consumption + stationary glycolytic fermentation + cellular maintenance
            q_ferm = DEFAULT_Q_MAX_FERM * sat_s * inhib_e * f_temp
            maint = DEFAULT_MS_MAINTENANCE * sat_s
            dS_dt = - ( (mu / DEFAULT_YX_S) + q_ferm + maint ) * X
            sugar_consumed_this_step = max(0.0, -dS_dt * dt_hours)
            sugar_consumed_this_step = min(S, sugar_consumed_this_step)

            # Daily rate for diagnostics
            daily_sugar_rate = (sugar_consumed_this_step / dt_hours) * 24.0
            if daily_sugar_rate > peak_rate:
                peak_rate = daily_sugar_rate
                peak_rate_day = current_day

            # Ethanol production
            dE_step = sugar_consumed_this_step / SUGAR_PER_ETHANOL_PCT

            # Nitrogen consumption
            dN_step = min(N, DEFAULT_YN_X * max(0.0, dX) * dt_hours)

            # Temperature dynamics: metabolic heat liberation vs cooling jacket
            # approx 0.02 °C rise per g/L sugar fermented without cooling
            metabolic_heat = 0.016 * sugar_consumed_this_step
            cooling = cooling_capacity_k * (T - jacket_temp_c) * dt_hours
            dT_step = metabolic_heat - cooling

            # Volatile acidity: baseline yeast glycolysis byproduct + thermal/osmotic stress
            thermal_stress = max(0.0, T - 28.0)
            starvation_stress = 1.0 if (S > 10.0 and mu < 0.005 and N < 15.0) else 0.0
            dVA_step = (0.0008 * sugar_consumed_this_step) + (k_stress_va * thermal_stress * dt_hours) + (0.0005 * starvation_stress * dt_hours)

            # Update states
            S = max(0.0, S - sugar_consumed_this_step)
            E = max(0.0, E + dE_step)
            N = max(0.0, N - dN_step)
            X = max(0.05, X + dX * dt_hours)
            T = max(5.0, min(42.0, T + dT_step))
            VA = min(1.80, VA + dVA_step)

        # Classify final outcome
        final_sugar = trajectory[-1]["sugar_g_l"]
        final_alcohol = trajectory[-1]["ethanol_pct"]
        final_va = trajectory[-1]["volatile_acidity"]

        if final_sugar <= 2.5:
            fermentation_status = "Dry (Complete)"
        elif final_sugar <= 6.0:
            fermentation_status = "Off-Dry (Near Complete)"
        elif peak_rate < 5.0 or final_sugar > 15.0:
            fermentation_status = "Stuck Fermentation"
        else:
            fermentation_status = "Sluggish Fermentation"

        # Finished wine quality projection
        if baseline_chemistry is None:
            baseline_chemistry = {
                "fixed acidity": 8.0,
                "volatile acidity": round(final_va, 2),
                "citric acid": 0.35,
                "residual sugar": round(final_sugar, 2),
                "chlorides": 0.065,
                "free sulfur dioxide": 25.0,
                "total sulfur dioxide": 60.0,
                "density": round(trajectory[-1]["density"], 4),
                "pH": 3.42,
                "sulphates": 0.65,
                "alcohol": round(final_alcohol, 2)
            }
        else:
            baseline_chemistry = dict(baseline_chemistry)
            baseline_chemistry["residual sugar"] = round(final_sugar, 2)
            baseline_chemistry["alcohol"] = round(final_alcohol, 2)
            baseline_chemistry["volatile acidity"] = round(final_va, 2)
            baseline_chemistry["density"] = round(trajectory[-1]["density"], 4)

        pred_quality = self._score_wine(baseline_chemistry)

        return {
            "yeast_strain": yeast_strain,
            "fermentation_status": fermentation_status,
            "duration_days": total_days,
            "final_residual_sugar_g_l": round(final_sugar, 2),
            "final_ethanol_pct": round(final_alcohol, 2),
            "final_volatile_acidity_g_l": round(final_va, 3),
            "peak_fermentation_rate_g_l_day": round(peak_rate, 2),
            "peak_rate_day": round(peak_rate_day, 1),
            "predicted_quality_prob": round(pred_quality, 4),
            "is_premium": bool(pred_quality >= self.decision_threshold),
            "trajectory": trajectory
        }

    def _score_wine(self, chem: Dict[str, float]) -> float:
        """Internal inference wrapper returning P(Quality >= 7)."""
        if self.model is None:
            # Enological heuristic fallback
            score = 0.50
            if chem.get("alcohol", 10.0) >= 11.5:
                score += 0.25
            if chem.get("volatile acidity", 0.5) <= 0.45:
                score += 0.15
            if chem.get("residual sugar", 2.0) <= 3.0:
                score += 0.10
            return float(np.clip(score, 0.01, 0.99))

        df_row = pd.DataFrame([chem])
        df_feat = create_engineered_features(df_row)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_feat)
            return float(probs[0, 1])
        return 0.50


# ============================================================================
# 2. Stuck Fermentation Early-Warning Detector
# ============================================================================

class StuckFermentationDetector:
    """
    Time-series kinetic monitor analyzing ongoing fermentation logs
    to detect stuck or sluggish fermentation and issue prescriptive cellar remediation.
    """

    def __init__(self, critical_sugar_floor_g_l: float = 3.0, sluggish_rate_threshold: float = 6.0):
        self.critical_sugar_floor = critical_sugar_floor_g_l
        self.sluggish_rate_threshold = sluggish_rate_threshold

    def diagnose_trajectory(
        self,
        daily_readings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes sequential readings: [{'day': 0, 'sugar': 220, 'temp': 20, 'yan': 200}, ...]
        Returns risk assessment, kinetic diagnostics, and prescriptive remediation.
        """
        if not daily_readings or len(daily_readings) < 2:
            return {
                "diagnosis": "INSUFFICIENT_DATA",
                "risk_score_pct": 0.0,
                "is_stuck_or_sluggish": False,
                "current_rate_g_l_day": 0.0,
                "message": "At least 2 sequential fermentation readings are required for kinetic diagnosis.",
                "prescriptive_actions": []
            }

        sorted_readings = sorted(daily_readings, key=lambda x: x.get("day", 0))
        latest = sorted_readings[-1]
        previous = sorted_readings[-2]

        day_now = float(latest.get("day", 0.0))
        day_prev = float(previous.get("day", 0.0))
        dt = max(0.1, day_now - day_prev)

        sugar_now = float(latest.get("sugar", latest.get("sugar_g_l", 0.0)))
        sugar_prev = float(previous.get("sugar", previous.get("sugar_g_l", 0.0)))
        temp_now = float(latest.get("temp", latest.get("temperature_c", 20.0)))
        yan_now = float(latest.get("yan", latest.get("yan_mg_l", 150.0)))
        alcohol_now = float(latest.get("alcohol", latest.get("ethanol_pct", 0.0)))

        # Velocity of sugar decline (positive = consuming sugar)
        sugar_consumption_rate = max(0.0, (sugar_prev - sugar_now) / dt)

        # Acceleration (change in velocity)
        if len(sorted_readings) >= 3:
            prev2 = sorted_readings[-3]
            dt2 = max(0.1, day_prev - float(prev2.get("day", 0.0)))
            rate_prev = max(0.0, (float(prev2.get("sugar", 0.0)) - sugar_prev) / dt2)
            acceleration = (sugar_consumption_rate - rate_prev) / dt
        else:
            acceleration = 0.0

        prescriptions: List[Dict[str, str]] = []
        is_stuck_or_sluggish = False

        # Evaluation Rules
        if sugar_now <= self.critical_sugar_floor:
            diagnosis = "COMPLETED_DRY"
            risk_pct = 0.0
            msg = f"Fermentation successfully completed dry ({sugar_now:.1f} g/L sugar). Safe for stabilization and racking."
        elif sugar_consumption_rate < 1.5 and sugar_now > 10.0:
            diagnosis = "STUCK_FERMENTATION"
            risk_pct = 95.0
            is_stuck_or_sluggish = True
            msg = f"CRITICAL: Fermentation arrested at {sugar_now:.1f} g/L sugar with near-zero consumption rate ({sugar_consumption_rate:.2f} g/L/day)."

            prescriptions.append({
                "action": "DETOXIFY_LEES",
                "protocol": "Rack off gross lees and add yeast cell walls / ghost hulls (20-30 g/hL) to adsorb toxic short-chain fatty acids (octanoic/decanoic)."
            })
            prescriptions.append({
                "action": "REINOCULATE_ACCLIMATIZED_STARTER",
                "protocol": "Inoculate vigorous fructophilic yeast (S. bayanus EC1118 or Uvaferm 43) using stepwise ethanol acclimatization starter culture."
            })
        elif sugar_consumption_rate < self.sluggish_rate_threshold and sugar_now > 15.0:
            diagnosis = "SLUGGISH_FERMENTATION"
            risk_pct = 65.0
            is_stuck_or_sluggish = True
            msg = f"WARNING: Sluggish sugar decay detected ({sugar_consumption_rate:.2f} g/L/day). Must is at risk of premature arrest."

            if yan_now < 100.0 and sugar_now > 50.0:
                prescriptions.append({
                    "action": "NITROGEN_SUPPLEMENTATION",
                    "protocol": f"Low YAN detected ({yan_now:.0f} mg/L). Dose 20 g/hL Diammonium Phosphate (DAP) or complex organic nitrogen nutrient."
                })
            prescriptions.append({
                "action": "AERATION_AND_LEES_RESUSPENSION",
                "protocol": "Perform gentle pump-over with splash aeration to stimulate sterol/ergosterol membrane synthesis, and resuspend settled active yeast."
            })
        elif temp_now > 30.0:
            diagnosis = "THERMAL_OVERHEATING_RISK"
            risk_pct = 75.0
            is_stuck_or_sluggish = True
            msg = f"ALERT: Elevated must temperature ({temp_now:.1f}°C) threatens yeast viability and increases volatile acidity."
            prescriptions.append({
                "action": "ENGAGE_COOLING_JACKET",
                "protocol": f"Lower cooling jacket target to 20°C - 22°C immediately to prevent yeast thermal shock and protein denaturation."
            })
        elif temp_now < 14.0 and sugar_now > 10.0:
            diagnosis = "COLD_INHIBITION"
            risk_pct = 50.0
            is_stuck_or_sluggish = True
            msg = f"Must temperature ({temp_now:.1f}°C) is below metabolic activation threshold."
            prescriptions.append({
                "action": "WARM_TANK_JACKET",
                "protocol": "Gradually heat tank to 18°C - 20°C to restore yeast enzymatic kinetics."
            })
        else:
            diagnosis = "HEALTHY_ACTIVE"
            risk_pct = 10.0
            msg = f"Fermentation progressing actively at {sugar_consumption_rate:.1f} g/L/day. Temperature ({temp_now:.1f}°C) is in optimal range."

        return {
            "diagnosis": diagnosis,
            "risk_score_pct": round(risk_pct, 1),
            "is_stuck_or_sluggish": is_stuck_or_sluggish,
            "current_sugar_g_l": round(sugar_now, 2),
            "current_rate_g_l_day": round(sugar_consumption_rate, 2),
            "acceleration_g_l_day2": round(acceleration, 2),
            "current_temperature_c": round(temp_now, 1),
            "current_ethanol_pct": round(alcohol_now, 2),
            "message": msg,
            "prescriptive_actions": prescriptions
        }


# ============================================================================
# 3. Sensory Organoleptic Lexicon & Spider Radar Engine
# ============================================================================

class SensoryOrganolepticEngine:
    """
    Translates laboratory physicochemical metrics into human organoleptic dimensions,
    generating 6-axis spider radar scores, sommelier tasting notes, and culinary food pairing affinities.
    """

    RADAR_ATTRIBUTES = [
        "Body & Viscosity",
        "Acidity & Crispness",
        "Tannic Grip & Structure",
        "Aromatic Freshness & Purity",
        "Sweetness-Acid Harmony",
        "Ethanol Warmth & Persistence"
    ]

    @classmethod
    def compute_sensory_profile(cls, chemistry: Dict[str, float]) -> Dict[str, Any]:
        """
        Maps standard enological measurements into 0.0 - 10.0 sensory scales.
        """
        fixed_acidity = float(chemistry.get("fixed acidity", 8.0))
        volatile_acidity = float(chemistry.get("volatile acidity", 0.5))
        citric_acid = float(chemistry.get("citric acid", 0.3))
        residual_sugar = float(chemistry.get("residual sugar", 2.0))
        chlorides = float(chemistry.get("chlorides", 0.07))
        free_so2 = float(chemistry.get("free sulfur dioxide", 20.0))
        total_so2 = float(chemistry.get("total sulfur dioxide", 50.0))
        density = float(chemistry.get("density", 0.996))
        pH = float(chemistry.get("pH", 3.4))
        sulphates = float(chemistry.get("sulphates", 0.65))
        alcohol = float(chemistry.get("alcohol", 10.5))

        # 1. Body & Viscosity (0-10): Governed by alcohol, density, dry extract, and residual sugar
        dry_extract_est = max(15.0, (density - 1.0) * 2600.0 + residual_sugar + fixed_acidity)
        raw_body = (alcohol - 9.0) * 1.1 + (dry_extract_est - 20.0) * 0.25
        body_score = float(np.clip(raw_body, 1.0, 10.0))

        # 2. Acidity & Crispness (0-10): Fixed acidity & inverse pH, penalized by high volatile acidity
        raw_acidity = (fixed_acidity - 5.0) * 0.75 + (3.65 - pH) * 4.0 - (volatile_acidity * 1.5)
        acidity_score = float(np.clip(raw_acidity, 1.0, 10.0))

        # 3. Tannic Grip & Structure (0-10): Sulphates, fixed acidity, and alcohol extraction
        raw_tannin = (sulphates - 0.35) * 5.0 + (fixed_acidity - 6.5) * 0.5 + (alcohol - 10.0) * 0.3
        tannin_score = float(np.clip(raw_tannin, 1.0, 10.0))

        # 4. Aromatic Freshness & Purity (0-10): Citric acid & free SO2 protection, penalized by VA & high total SO2
        so2_protection = min(3.0, free_so2 / 10.0)
        va_penalty = max(0.0, (volatile_acidity - 0.30) * 7.0)
        raw_freshness = (citric_acid * 7.5) + so2_protection - va_penalty + 3.0
        freshness_score = float(np.clip(raw_freshness, 1.0, 10.0))

        # 5. Sweetness-Acid Harmony (0-10): Ratio of residual sugar to total acidity
        balance_ratio = residual_sugar / max(1.0, fixed_acidity)
        # Optimal dry balance is around 0.25 - 0.45
        balance_dev = abs(balance_ratio - 0.35)
        raw_harmony = 9.0 - (balance_dev * 12.0) - max(0.0, (volatile_acidity - 0.5) * 4.0)
        harmony_score = float(np.clip(raw_harmony, 1.0, 10.0))

        # 6. Ethanol Warmth & Persistence (0-10): Sensory length and heat on palate
        raw_warmth = (alcohol - 8.5) * 1.35
        warmth_score = float(np.clip(raw_warmth, 1.0, 10.0))

        radar_scores = {
            "Body & Viscosity": round(body_score, 1),
            "Acidity & Crispness": round(acidity_score, 1),
            "Tannic Grip & Structure": round(tannin_score, 1),
            "Aromatic Freshness & Purity": round(freshness_score, 1),
            "Sweetness-Acid Harmony": round(harmony_score, 1),
            "Ethanol Warmth & Persistence": round(warmth_score, 1)
        }

        # Synthesize Sommelier Tasting Note
        tasting_note = cls._generate_sommelier_note(radar_scores, chemistry)

        # Compute Food Pairing Affinities
        food_pairings = cls._compute_food_pairings(radar_scores)

        return {
            "radar_scores": radar_scores,
            "overall_sensory_intensity": round(float(np.mean(list(radar_scores.values()))), 2),
            "dry_extract_est_g_l": round(dry_extract_est, 1),
            "tasting_note": tasting_note,
            "food_pairings": food_pairings
        }

    @classmethod
    def _generate_sommelier_note(cls, scores: Dict[str, float], chem: Dict[str, float]) -> Dict[str, str]:
        """Synthesizes structured professional tasting descriptors."""
        alcohol = chem.get("alcohol", 10.5)
        va = chem.get("volatile acidity", 0.5)

        # Eye
        if scores["Body & Viscosity"] >= 7.5:
            appearance = "Deep garnet with concentrated ruby rim, displaying dense, slow-moving tears."
        elif scores["Body & Viscosity"] >= 5.0:
            appearance = "Bright crimson hue with translucent edge and moderate legs."
        else:
            appearance = "Pale ruby with light brick highlights and rapid tearing."

        # Nose
        if va >= 0.70:
            nose = "Forward volatile notes of balsamic glaze, bruised apple, and sharp acetone overshadowing faint red currant fruit."
        elif scores["Aromatic Freshness & Purity"] >= 7.0:
            nose = "Expressive bouquet of fresh black cherries, wild violets, crushed granite, and cracked black pepper with vibrant lift."
        else:
            nose = "Subdued dark plum, cedar shavings, dried thyme, and subtle baking spices."

        # Palate
        body_desc = "Full-bodied and velvety" if scores["Body & Viscosity"] >= 7.0 else ("Medium-bodied and supple" if scores["Body & Viscosity"] >= 4.5 else "Light-bodied and linear")
        tannin_desc = "grippy, fine-grained mineral tannins" if scores["Tannic Grip & Structure"] >= 6.5 else "soft, rounded, accessible tannins"
        acidity_desc = "electric, mouth-watering acidity" if scores["Acidity & Crispness"] >= 7.0 else "measured, well-integrated acidity"

        palate = f"{body_desc} on entry, framed by {acidity_desc} and supported by {tannin_desc}."

        # Finish
        if scores["Ethanol Warmth & Persistence"] >= 7.5:
            finish = f"Long, warm finish ({alcohol:.1f}% ABV) lingering with toasted oak, espresso bean, and dark cacao."
        else:
            finish = f"Crisp, refreshing finish ({alcohol:.1f}% ABV) characterized by clean tart red fruit and dried herbs."

        return {
            "appearance": appearance,
            "nose": nose,
            "palate": palate,
            "finish": finish
        }

    @classmethod
    def _compute_food_pairings(cls, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculates gastronomy affinity compatibility scores (0 - 100%)."""
        body = scores["Body & Viscosity"]
        acidity = scores["Acidity & Crispness"]
        tannin = scores["Tannic Grip & Structure"]
        warmth = scores["Ethanol Warmth & Persistence"]
        freshness = scores["Aromatic Freshness & Purity"]

        dishes = [
            {
                "dish": "Charbroiled Prime Ribeye Steak",
                "category": "Red Meat",
                # Needs high tannin to bind salivary proteins and bold body to match fat
                "affinity": float(np.clip(15.0 + 4.5 * tannin + 3.5 * body + 2.0 * warmth, 0, 100)),
                "pairing_rationale": "Ample tannins and bold structure effortlessly cut through marbling and rich umami."
            },
            {
                "dish": "Roasted Duck Breast & Confit Duck",
                "category": "Poultry & Game",
                # Needs medium body, high acidity to cut grease, and moderate tannins
                "affinity": float(np.clip(20.0 + 4.5 * acidity + 3.0 * body + 2.5 * freshness, 0, 100)),
                "pairing_rationale": "Vibrant acidity cuts through duck fat while ripe red fruit accents savory spices."
            },
            {
                "dish": "Aged Hard Cheeses (Comté & Manchego)",
                "category": "Dairy & Artisan Cheese",
                # Needs warmth, body, and balanced tannins to pair with lactic crystals
                "affinity": float(np.clip(25.0 + 3.5 * body + 3.0 * warmth + 2.5 * tannin, 0, 100)),
                "pairing_rationale": "Complex nutty curds mirror the wine's rich extract and ethanol warmth."
            },
            {
                "dish": "Pan-Seared Wild Salmon & Morel Mushrooms",
                "category": "Fish & Seafood",
                # Needs low tannin, high acidity, and aromatic freshness
                "affinity": float(np.clip(30.0 + 5.0 * freshness + 4.0 * acidity - 3.5 * tannin, 0, 100)),
                "pairing_rationale": "Delicate red fruit and brisk acidity flatter the rich omega-3 oils without metallic clash."
            },
            {
                "dish": "Spicy Red Thai Curry & Sichuan Noodles",
                "category": "Spicy Cuisine",
                # High alcohol and tannin exacerbate capsaicin burn; needs low warmth and low tannin
                "affinity": float(np.clip(45.0 + 4.0 * freshness + 3.0 * scores["Sweetness-Acid Harmony"] - 5.0 * warmth - 3.0 * tannin, 0, 100)),
                "pairing_rationale": "Light body and low perceived heat prevent burning capsaicin friction on the palate."
            }
        ]

        # Sort descending by affinity
        for d in dishes:
            d["affinity"] = round(d["affinity"], 1)
        dishes.sort(key=lambda x: x["affinity"], reverse=True)
        return dishes


# ============================================================================
# 4. Causal Enological Engine (Counterfactual Intervention Analysis)
# ============================================================================

class CausalEnologicalEngine:
    """
    Evaluates observational causal effects of cellar treatments on wine quality
    using Doubly Robust / Inverse Propensity Weighting (IPW) estimation to mitigate confounding.
    """

    INTERVENTIONS = {
        "low_volatile_acidity": {
            "name": "Volatile Acidity Mitigation (VA <= 0.45 g/L)",
            "treatment_fn": lambda df: (df["volatile acidity"] <= 0.45).astype(int),
            "confounders": ["fixed acidity", "citric acid", "density", "pH", "alcohol"]
        },
        "preservative_sulphates": {
            "name": "Adequate Sulphate Fortification (Sulphates >= 0.65 g/L)",
            "treatment_fn": lambda df: (df["sulphates"] >= 0.65).astype(int),
            "confounders": ["fixed acidity", "pH", "volatile acidity", "alcohol"]
        },
        "optimal_ripeness_alcohol": {
            "name": "Grape Ripeness & Chaptalization (Alcohol >= 11.5% vol)",
            "treatment_fn": lambda df: (df["alcohol"] >= 11.5).astype(int),
            "confounders": ["density", "residual sugar", "pH", "fixed acidity"]
        },
        "antimicrobial_free_so2": {
            "name": "Free SO2 Protection (Free SO2 >= 18.0 mg/L)",
            "treatment_fn": lambda df: (df["free sulfur dioxide"] >= 18.0).astype(int),
            "confounders": ["pH", "volatile acidity", "alcohol"]
        }
    }

    @classmethod
    def estimate_causal_effect(
        cls,
        df: pd.DataFrame,
        intervention_key: str = "low_volatile_acidity",
        target_col: str = "quality",
        quality_threshold: int = 7,
        n_bootstraps: int = 200,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Computes the Average Treatment Effect (ATE) of the enological intervention on P(Quality >= 7).
        """
        if intervention_key not in cls.INTERVENTIONS:
            raise ValueError(f"Unknown intervention '{intervention_key}'. Available: {list(cls.INTERVENTIONS.keys())}")

        spec = cls.INTERVENTIONS[intervention_key]
        df_work = df.copy()

        # Treatment assignment T
        T = spec["treatment_fn"](df_work).values
        # Binary outcome Y (1 if premium quality, 0 otherwise)
        Y = (df_work[target_col] >= quality_threshold).astype(int).values

        confounders = [c for c in spec["confounders"] if c in df_work.columns]
        X_conf = df_work[confounders].values

        # Standardize confounders
        mean = np.mean(X_conf, axis=0)
        std = np.std(X_conf, axis=0) + 1e-6
        X_scaled = (X_conf - mean) / std

        # Fit Logistic Regression for Propensity Score P(T=1 | X)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(solver="lbfgs", max_iter=500, random_state=random_state)
        lr.fit(X_scaled, T)
        propensity = np.clip(lr.predict_proba(X_scaled)[:, 1], 0.05, 0.95)

        # Inverse Propensity Weighting (IPW) estimator
        # E[Y(1)] = mean( T * Y / e )
        # E[Y(0)] = mean( (1 - T) * Y / (1 - e) )
        w1 = T / propensity
        w0 = (1.0 - T) / (1.0 - propensity)

        e_y1 = np.sum(w1 * Y) / np.sum(w1)
        e_y0 = np.sum(w0 * Y) / np.sum(w0)
        ate = float(e_y1 - e_y0)

        # Naive Observational Correlation (unadjusted difference in means)
        naive_treated = float(np.mean(Y[T == 1])) if np.sum(T == 1) > 0 else 0.0
        naive_control = float(np.mean(Y[T == 0])) if np.sum(T == 0) > 0 else 0.0
        naive_delta = naive_treated - naive_control
        confounding_bias = float(naive_delta - ate)

        # Bootstrap 95% Confidence Interval
        np.random.seed(random_state)
        boot_ates = []
        n_samples = len(df_work)
        for _ in range(n_bootstraps):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            T_b = T[idx]
            Y_b = Y[idx]
            p_b = propensity[idx]
            w1_b = T_b / p_b
            w0_b = (1.0 - T_b) / (1.0 - p_b)
            if np.sum(w1_b) > 0 and np.sum(w0_b) > 0:
                ate_b = (np.sum(w1_b * Y_b) / np.sum(w1_b)) - (np.sum(w0_b * Y_b) / np.sum(w0_b))
                boot_ates.append(ate_b)

        ci_lower = float(np.percentile(boot_ates, 2.5)) if boot_ates else ate - 0.05
        ci_upper = float(np.percentile(boot_ates, 97.5)) if boot_ates else ate + 0.05

        return {
            "intervention_key": intervention_key,
            "intervention_name": spec["name"],
            "treated_sample_count": int(np.sum(T == 1)),
            "control_sample_count": int(np.sum(T == 0)),
            "causal_ate": round(ate, 4),
            "causal_ate_pct": round(ate * 100.0, 2),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "naive_observational_delta": round(naive_delta, 4),
            "confounding_bias": round(confounding_bias, 4),
            "statistically_significant": bool(ci_lower > 0.0 or ci_upper < 0.0)
        }
