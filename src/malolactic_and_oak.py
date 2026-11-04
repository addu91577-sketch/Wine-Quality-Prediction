"""
Enological Secondary Malolactic Fermentation (MLF) Biotransformation,
Precision Oak Barrel Maturation & Wood Phenolics Extraction Dynamics,
Digital Cellar Master Dispatch Engine, and HACCP Enological Risk Auditor.

Modules & Capabilities:
- MalolacticFermentationSimulator: Simulates Oenococcus oeni decarboxylation of L-malic acid
  into L-lactic acid and CO2, tracking pH evolution, fixed acidity shift, and diacetyl buttery synthesis.
- OakBarrelMaturationSimulator: Models first-order extraction kinetics of wood ellagitannins,
  oak lactones, vanillin, and guaiacol smoke across French, American, and Hungarian oak,
  together with micro-oxygenation tannin-anthocyanin polymeric pigment color stabilization.
- CellarMasterDispatchEngine: Multi-vessel triage engine calculating the Enological Risk Priority
  Index (ERPI) and generating prescriptive cellar work orders with exact chemical additions.
- EnologicalHaccpAuditor: Critical Control Point (CCP) verification engine evaluating molecular
  SO2 preservative sufficiency, legal bounds, and biogenic amine (histamine/tyramine) hazard indices.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import create_engineered_features


# ============================================================================
# 1. Malolactic Fermentation (MLF) Simulator
# ============================================================================

# Physical & Biochemical Constants for Malolactic Fermentation
MW_MALIC_ACID = 134.09    # g/mol
MW_LACTIC_ACID = 90.08    # g/mol
STOICH_MALIC_TO_LACTIC = MW_LACTIC_ACID / MW_MALIC_ACID  # ~0.6718 g lactic per g malic

DEFAULT_MU_MAX_MLF = 0.35      # Maximum specific growth rate of Oenococcus oeni (1/day)
DEFAULT_KM_MALIC = 0.35         # Affinity constant for malic acid (g/L)
DEFAULT_KD_MLF = 0.012          # Natural bacterial decay rate (1/day)
DEFAULT_YX_M = 0.06             # Biomass yield from malic acid (g biomass / g malic acid)
DEFAULT_QM_MAX = 1.85           # Maximum specific malic degradation rate (g malic / g biomass / day)
DEFAULT_MAINTENANCE_M = 0.05    # Maintenance malic consumption (g malic / g biomass / day)


class MalolacticFermentationSimulator:
    """
    Simulates secondary malolactic fermentation (MLF) biotransformation by Oenococcus oeni.
    Tracks malic acid depletion, lactic acid accumulation, upward pH shift,
    titratable acidity reduction, and diacetyl flavor synthesis.
    """

    def __init__(
        self,
        mu_max: float = DEFAULT_MU_MAX_MLF,
        km_malic: float = DEFAULT_KM_MALIC,
        kd: float = DEFAULT_KD_MLF,
        yx_m: float = DEFAULT_YX_M,
        qm_max: float = DEFAULT_QM_MAX,
    ):
        self.mu_max = mu_max
        self.km_malic = km_malic
        self.kd = kd
        self.yx_m = yx_m
        self.qm_max = qm_max

    def _environmental_inhibition_factor(
        self,
        temperature: float,
        ph: float,
        alcohol: float,
        free_so2: float,
    ) -> float:
        """
        Calculates compounded environmental inhibition scalar [0.0, 1.0] for O. oeni.
        """
        # 1. Temperature factor: Optimum 18-22°C; bell curve width ~5.0°C
        if temperature < 10.0 or temperature > 32.0:
            f_temp = 0.02
        else:
            f_temp = math.exp(-((temperature - 20.0) ** 2) / (2 * (5.0 ** 2)))

        # 2. pH factor: active between 3.2 and 3.8
        f_ph = max(0.04, min(1.0, (ph - 2.9) / 0.5))

        # 3. Alcohol factor: moderate tolerance up to 14% ABV, dropping sharply above 15.5%
        f_eth = max(0.02, min(1.0, 1.0 - (alcohol / 16.5) ** 3))

        # 4. Free SO2 factor: Sensitive to Free SO2 > 10 mg/L, severely inhibited near 20 mg/L
        f_so2 = max(0.02, min(1.0, 1.0 - (free_so2 / 20.0) ** 2))

        # Combined multiplicative gating
        gamma = f_temp * f_ph * f_eth * f_so2
        return float(max(0.005, min(1.0, gamma)))

    def simulate(
        self,
        initial_malic_acid: float = 2.8,       # g/L
        initial_ph: float = 3.35,
        initial_fixed_acidity: float = 7.2,    # g/L tartaric equivalent
        alcohol: float = 13.0,                 # % v/v
        free_so2: float = 6.0,                 # mg/L
        temperature: float = 20.0,             # °C
        inoculated_biomass: float = 0.12,      # g/L initial O. oeni biomass
        duration_days: int = 35,
        step_days: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Executes dynamic numerical integration of MLF over specified duration.
        """
        steps = int(duration_days / step_days)
        time_points = [round(i * step_days, 2) for i in range(steps + 1)]

        malic = float(initial_malic_acid)
        lactic = 0.0
        biomass = float(inoculated_biomass)
        curr_ph = float(initial_ph)
        curr_fa = float(initial_fixed_acidity)
        diacetyl = 0.0  # mg/L

        history: Dict[str, List[float]] = {
            "day": [],
            "malic_acid_g_l": [],
            "lactic_acid_g_l": [],
            "biomass_g_l": [],
            "ph": [],
            "fixed_acidity_g_l": [],
            "diacetyl_mg_l": [],
            "conversion_rate_pct": [],
            "inhibition_factor": [],
        }

        completion_day: Optional[float] = None
        peak_diacetyl = 0.0

        for t in time_points:
            conversion = max(0.0, min(100.0, ((initial_malic_acid - malic) / max(0.001, initial_malic_acid)) * 100.0))
            if malic <= 0.10 and completion_day is None:
                completion_day = t

            env_factor = self._environmental_inhibition_factor(
                temperature=temperature,
                ph=curr_ph,
                alcohol=alcohol,
                free_so2=free_so2,
            )

            history["day"].append(t)
            history["malic_acid_g_l"].append(round(malic, 4))
            history["lactic_acid_g_l"].append(round(lactic, 4))
            history["biomass_g_l"].append(round(biomass, 4))
            history["ph"].append(round(curr_ph, 3))
            history["fixed_acidity_g_l"].append(round(curr_fa, 3))
            history["diacetyl_mg_l"].append(round(diacetyl, 4))
            history["conversion_rate_pct"].append(round(conversion, 2))
            history["inhibition_factor"].append(round(env_factor, 3))

            peak_diacetyl = max(peak_diacetyl, diacetyl)

            # Numerical derivative step
            if malic > 0.01:
                mu = self.mu_max * env_factor * (malic / (self.km_malic + malic))
                d_biomass = (mu - self.kd) * biomass
                q_malic = self.qm_max * env_factor * (malic / (self.km_malic + malic))
                d_malic = - q_malic * biomass
            else:
                d_biomass = - self.kd * biomass
                d_malic = 0.0

            dt = step_days
            consumed = min(malic, max(0.0, -d_malic * dt))
            malic = max(0.0, malic - consumed)
            lactic = lactic + consumed * STOICH_MALIC_TO_LACTIC
            biomass = max(0.005, min(1.5, biomass + d_biomass * dt))

            # Enological pH & fixed acidity shifts
            consumed_total = initial_malic_acid - malic
            curr_fa = max(3.5, initial_fixed_acidity - 0.58 * consumed_total)
            curr_ph = min(4.2, initial_ph + 0.065 * consumed_total)

            # Diacetyl kinetics
            rate_consumption = consumed / dt
            d_diacetyl_synth = 0.85 * rate_consumption * env_factor
            d_diacetyl_red = 0.25 * diacetyl * (biomass / (0.05 + biomass))
            diacetyl = max(0.0, diacetyl + (d_diacetyl_synth - d_diacetyl_red) * dt)

        final_malic = history["malic_acid_g_l"][-1]
        final_lactic = history["lactic_acid_g_l"][-1]
        final_conversion = history["conversion_rate_pct"][-1]
        is_complete = final_malic <= 0.10
        is_stuck = not is_complete and (len(history["malic_acid_g_l"]) > 10 and 
                                       history["malic_acid_g_l"][-1] >= history["malic_acid_g_l"][-5] - 0.03)

        # Sensory impact summary
        if final_conversion > 90.0:
            sensory_profile = "Complete MLF: Velvety mouthfeel, rounded deacidification, integrated lactic richness."
        elif final_conversion > 50.0:
            sensory_profile = "Partial MLF: Crisp malic core with emerging buttery/brioche complexity."
        else:
            sensory_profile = "Inhibited/Minimal MLF: High malic tartness, taut linear acidity preserved."

        return {
            "initial_state": {
                "malic_acid_g_l": initial_malic_acid,
                "ph": initial_ph,
                "fixed_acidity_g_l": initial_fixed_acidity,
                "alcohol_pct": alcohol,
                "free_so2_mg_l": free_so2,
                "temperature_c": temperature,
            },
            "final_state": {
                "malic_acid_g_l": final_malic,
                "lactic_acid_g_l": final_lactic,
                "ph": round(curr_ph, 3),
                "fixed_acidity_g_l": round(curr_fa, 3),
                "diacetyl_mg_l": round(diacetyl, 3),
                "peak_diacetyl_mg_l": round(peak_diacetyl, 3),
                "conversion_rate_pct": final_conversion,
            },
            "diagnostics": {
                "is_complete": is_complete,
                "is_stuck": is_stuck,
                "completion_day": completion_day,
                "microbial_stability_score": round(max(0.0, min(100.0, (1.0 - final_malic / max(0.1, initial_malic_acid)) * 100.0)), 1),
                "sensory_descriptor": sensory_profile,
            },
            "trajectory": history,
        }


# ============================================================================
# 2. Oak Barrel Maturation & Wood Phenolics Extraction Simulator
# ============================================================================

# Wood extractive species profiles: Max extraction capacity (mg/L in 225L barrique)
OAK_SPECIES_PROFILES = {
    "french": {
        "species_name": "Quercus petraea / Quercus robur (French Oak)",
        "ellagitannins_max": 240.0,     # High structural tannins, tight grain
        "whiskey_lactones_max": 1.4,     # Subtle coconut/woody
        "vanillin_max": 3.2,             # Elegant spice & vanilla
        "guaiacol_max": 0.8,             # Mild smoke
        "furfural_max": 14.0,            # Toasted almond / sweet bread
        "extraction_rate_k": 0.16,       # Monthly rate constant
    },
    "american": {
        "species_name": "Quercus alba (American White Oak)",
        "ellagitannins_max": 110.0,     # Lower tannins, softer astringency
        "whiskey_lactones_max": 4.6,     # Very high sweet lactone / coconut
        "vanillin_max": 5.8,             # Intense creamy vanilla
        "guaiacol_max": 1.2,             # Moderate smoke
        "furfural_max": 18.0,            # Sweet caramel / maple
        "extraction_rate_k": 0.22,       # Faster extraction kinetics
    },
    "hungarian": {
        "species_name": "Quercus frainetto / petraea (Hungarian / Slavonian Oak)",
        "ellagitannins_max": 180.0,     # Medium-high structure
        "whiskey_lactones_max": 2.2,     # Moderate aromatic intensity
        "vanillin_max": 4.1,             # Balanced baking spice
        "guaiacol_max": 1.0,             # Roasted coffee / hazelnut
        "furfural_max": 15.0,            # Roasted malt
        "extraction_rate_k": 0.18,       # Balanced rate constant
    },
}

# Toast level modifiers (multipliers for extractives)
TOAST_LEVEL_MODIFIERS = {
    "light": {
        "ellagitannins": 1.15,
        "whiskey_lactones": 0.70,
        "vanillin": 0.65,
        "guaiacol": 0.30,
        "furfural": 0.70,
    },
    "medium": {
        "ellagitannins": 1.00,
        "whiskey_lactones": 1.10,
        "vanillin": 1.20,
        "guaiacol": 0.90,
        "furfural": 1.15,
    },
    "medium_plus": {
        "ellagitannins": 0.80,
        "whiskey_lactones": 1.25,
        "vanillin": 1.35,
        "guaiacol": 1.50,
        "furfural": 1.40,
    },
    "heavy": {
        "ellagitannins": 0.55,
        "whiskey_lactones": 0.95,
        "vanillin": 1.10,
        "guaiacol": 2.40,
        "furfural": 1.65,
    },
}

# Barrel generation exhaustion factor (new vs used barrels)
BARREL_USE_FACTORS = {
    "new": 1.00,          # 100% extraction potential (1st fill)
    "first_fill": 0.55,    # 2nd wine (1-year used)
    "second_fill": 0.28,   # 3rd wine (2-years used)
    "neutral": 0.08,       # 4+ years used (minimal wood flavor, oxygen porosity only)
}


class OakBarrelMaturationSimulator:
    """
    Simulates dynamic oak barrel aging kinetics: wood extractive dissolution,
    species-specific aromatic compounds, and micro-oxygenation tannin-anthocyanin
    polymeric pigment color stabilization.
    """

    def __init__(self):
        pass

    def simulate(
        self,
        oak_species: str = "french",
        toast_level: str = "medium_plus",
        barrel_usage: str = "new",
        maturation_months: int = 18,
        barrel_volume_liters: float = 225.0,  # Standard Bordeaux barrique
        step_months: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Executes first-order Fickian dissolution kinetics of wood phenolics
        and micro-oxygenation tannin polymerization over maturation time.
        """
        species_key = oak_species.lower()
        if species_key not in OAK_SPECIES_PROFILES:
            raise ValueError(f"Unknown oak species '{oak_species}'. Supported: {list(OAK_SPECIES_PROFILES.keys())}")

        toast_key = toast_level.lower()
        if toast_key not in TOAST_LEVEL_MODIFIERS:
            raise ValueError(f"Unknown toast level '{toast_level}'. Supported: {list(TOAST_LEVEL_MODIFIERS.keys())}")

        usage_key = barrel_usage.lower()
        if usage_key not in BARREL_USE_FACTORS:
            raise ValueError(f"Unknown barrel usage '{barrel_usage}'. Supported: {list(BARREL_USE_FACTORS.keys())}")

        spec = OAK_SPECIES_PROFILES[species_key]
        toast = TOAST_LEVEL_MODIFIERS[toast_key]
        use_factor = BARREL_USE_FACTORS[usage_key]

        # Surface-to-volume ratio scaling: standard 225L barrique has S/V ~ 1.0 (baseline)
        # Larger barrels (e.g. 500L puncheon or 1000L foudre) extract proportionally less
        sv_ratio = (225.0 / max(50.0, barrel_volume_liters)) ** (1.0 / 3.0)

        # Asymptotic maximum concentrations for this configuration
        c_ellagitannins_max = spec["ellagitannins_max"] * toast["ellagitannins"] * use_factor * sv_ratio
        c_lactones_max = spec["whiskey_lactones_max"] * toast["whiskey_lactones"] * use_factor * sv_ratio
        c_vanillin_max = spec["vanillin_max"] * toast["vanillin"] * use_factor * sv_ratio
        c_guaiacol_max = spec["guaiacol_max"] * toast["guaiacol"] * use_factor * sv_ratio
        c_furfural_max = spec["furfural_max"] * toast["furfural"] * use_factor * sv_ratio
        k_rate = spec["extraction_rate_k"]

        steps = int(maturation_months / step_months)
        time_points = [round(i * step_months, 2) for i in range(steps + 1)]

        history: Dict[str, List[float]] = {
            "month": [],
            "ellagitannins_mg_l": [],
            "whiskey_lactones_mg_l": [],
            "vanillin_mg_l": [],
            "guaiacol_mg_l": [],
            "furfural_mg_l": [],
            "cumulative_o2_permeation_mg_l": [],
            "polymeric_pigment_index": [],
            "astringency_softening_pct": [],
        }

        # Passive stave oxygen ingress rate: ~2.2 mg O2 / L / month for 225L barrique
        o2_ingress_rate = 2.2 * sv_ratio

        for m in time_points:
            # First-order extraction model: C(t) = C_max * (1 - exp(-k * t))
            f_ext = 1.0 - math.exp(-k_rate * m) if m > 0 else 0.0

            ellagitannins = c_ellagitannins_max * f_ext
            lactones = c_lactones_max * f_ext
            vanillin = c_vanillin_max * f_ext
            guaiacol = c_guaiacol_max * f_ext
            furfural = c_furfural_max * f_ext

            # Cumulative oxygen permeation
            cum_o2 = o2_ingress_rate * m

            # Tannin polymerization & color stabilization:
            # Harsh monomeric catechins cross-link via acetaldehyde bridges into stable polymers
            # Polymeric pigment index rises from baseline 15.0 to upwards of 55.0
            poly_pigment = 15.0 + 40.0 * (1.0 - math.exp(-0.11 * m))

            # Astringency softening: sensory perception softens by up to 45% as tannins polymerize
            astringency_softening = min(50.0, 48.0 * (1.0 - math.exp(-0.13 * m)))

            history["month"].append(m)
            history["ellagitannins_mg_l"].append(round(ellagitannins, 2))
            history["whiskey_lactones_mg_l"].append(round(lactones, 3))
            history["vanillin_mg_l"].append(round(vanillin, 3))
            history["guaiacol_mg_l"].append(round(guaiacol, 3))
            history["furfural_mg_l"].append(round(furfural, 2))
            history["cumulative_o2_permeation_mg_l"].append(round(cum_o2, 2))
            history["polymeric_pigment_index"].append(round(poly_pigment, 2))
            history["astringency_softening_pct"].append(round(astringency_softening, 2))

        # Sensory oak intensity rating (0 - 100)
        final_lactones = history["whiskey_lactones_mg_l"][-1]
        final_vanillin = history["vanillin_mg_l"][-1]
        final_guaiacol = history["guaiacol_mg_l"][-1]
        oak_intensity_score = min(
            100.0,
            (final_lactones / 3.5) * 35.0 + (final_vanillin / 4.5) * 35.0 + (final_guaiacol / 1.5) * 30.0
        )

        return {
            "parameters": {
                "oak_species": spec["species_name"],
                "toast_level": toast_key.upper(),
                "barrel_usage": usage_key.upper(),
                "barrel_volume_liters": barrel_volume_liters,
                "maturation_months": maturation_months,
            },
            "final_concentrations": {
                "ellagitannins_mg_l": history["ellagitannins_mg_l"][-1],
                "whiskey_lactones_mg_l": history["whiskey_lactones_mg_l"][-1],
                "vanillin_mg_l": history["vanillin_mg_l"][-1],
                "guaiacol_mg_l": history["guaiacol_mg_l"][-1],
                "furfural_mg_l": history["furfural_mg_l"][-1],
                "polymeric_pigment_index": history["polymeric_pigment_index"][-1],
                "astringency_softening_pct": history["astringency_softening_pct"][-1],
                "cumulative_o2_permeation_mg_l": history["cumulative_o2_permeation_mg_l"][-1],
                "oak_intensity_score": round(oak_intensity_score, 1),
            },
            "trajectory": history,
        }


# ============================================================================
# 3. Digital Cellar Master Dispatch Engine
# ============================================================================

# Potassium metabisulfite (PMS, K2S2O5) active SO2 yield: 57.6%
PMS_SO2_YIELD = 0.576


class CellarMasterDispatchEngine:
    """
    Automated winery cellar management engine: calculates the Enological Risk
    Priority Index (ERPI) across cellar vessels, triages tanks/barrels into
    action tiers, and synthesizes prescriptive work orders with dosage calculations.
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_molecular_so2(free_so2: float, ph: float) -> float:
        """
        Calculates active antimicrobial molecular SO2 using Henderson-Hasselbalch equation.
        Molecular SO2 = Free SO2 / (1 + 10^(pH - 1.81))
        """
        if free_so2 <= 0.0 or ph <= 0.0:
            return 0.0
        return float(free_so2 / (1.0 + 10.0 ** (ph - 1.81)))

    @staticmethod
    def calculate_pms_addition_grams(
        target_free_so2: float,
        current_free_so2: float,
        volume_liters: float,
        binding_ratio: float = 0.35,  # ~35% of added SO2 binds with wine carbonyls/anthocyanins
    ) -> float:
        """
        Calculates required Potassium Metabisulfite (PMS, K2S2O5) addition in grams.
        """
        deficit_mg_l = max(0.0, target_free_so2 - current_free_so2)
        if deficit_mg_l == 0.0 or volume_liters <= 0:
            return 0.0

        # Required gross addition taking binding into account
        gross_addition_mg_l = deficit_mg_l / (1.0 - binding_ratio)
        total_so2_needed_mg = gross_addition_mg_l * volume_liters
        pms_needed_mg = total_so2_needed_mg / PMS_SO2_YIELD
        return float(round(pms_needed_mg / 1000.0, 2))  # Grams

    def evaluate_vessel(self, vessel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates ERPI score, urgency tier, and work orders for a single cellar vessel.
        """
        vessel_id = vessel.get("vessel_id", "UNKNOWN")
        vessel_type = vessel.get("vessel_type", "TANK")  # TANK, BARREL, PUNCHEON
        volume_l = float(vessel.get("volume_liters", 225.0))
        cultivar = vessel.get("cultivar", "red").lower()

        ph = float(vessel.get("ph", 3.4))
        free_so2 = float(vessel.get("free_so2", 15.0))
        total_so2 = float(vessel.get("total_so2", 60.0))
        va = float(vessel.get("volatile_acidity", 0.45))
        alcohol = float(vessel.get("alcohol", 13.0))
        temp = float(vessel.get("temperature", 18.0))
        residual_sugar = float(vessel.get("residual_sugar", 2.0))
        malic_acid = float(vessel.get("malic_acid", 0.05))
        maturation_days = int(vessel.get("maturation_days", 120))
        status = vessel.get("status", "aging").lower()  # fermentation, mlf, aging, resting

        # 1. Compute molecular SO2
        mol_so2 = self.calculate_molecular_so2(free_so2, ph)
        min_mol_so2 = 0.50 if cultivar == "red" else 0.80

        # 2. Risk Sub-Scores (0 to 100)
        # SO2 protection penalty: severe below target molecular SO2
        if mol_so2 < 0.25:
            so2_penalty = 100.0
        elif mol_so2 < min_mol_so2:
            so2_penalty = 40.0 + 60.0 * (1.0 - (mol_so2 - 0.25) / (min_mol_so2 - 0.25))
        else:
            so2_penalty = max(0.0, 15.0 * (1.0 - (mol_so2 - min_mol_so2) / 0.5))

        # Volatile Acidity risk penalty
        if va >= 1.0:
            va_penalty = 100.0
        elif va >= 0.70:
            va_penalty = 60.0 + 40.0 * ((va - 0.70) / 0.30)
        elif va >= 0.50:
            va_penalty = 20.0 + 40.0 * ((va - 0.50) / 0.20)
        else:
            va_penalty = max(0.0, 20.0 * (va / 0.50))

        # Microbial / MLF instability penalty: if malic acid is lingering in warm conditions with low SO2
        if malic_acid > 0.30 and status == "aging":
            mlf_risk_penalty = min(100.0, 50.0 + (malic_acid / 2.0) * 50.0)
        else:
            mlf_risk_penalty = 5.0

        # Temperature / Oxidation exposure penalty
        if temp > 22.0:
            temp_penalty = min(100.0, 40.0 + (temp - 22.0) * 12.0)
        elif temp < 10.0:
            temp_penalty = 20.0
        else:
            temp_penalty = 5.0

        # Residual sugar refermentation hazard penalty (sweet wine with low SO2)
        if residual_sugar > 4.0 and mol_so2 < 0.6:
            sugar_hazard_penalty = 75.0
        else:
            sugar_hazard_penalty = 5.0

        # Compounded Enological Risk Priority Index (ERPI)
        erpi = (
            0.35 * so2_penalty +
            0.30 * va_penalty +
            0.15 * mlf_risk_penalty +
            0.10 * temp_penalty +
            0.10 * sugar_hazard_penalty
        )
        erpi = float(round(max(0.0, min(100.0, erpi)), 1))

        # Action Tier Determination
        if erpi >= 68.0 or mol_so2 < 0.25 or va >= 0.90:
            action_tier = "CRITICAL_INTERVENTION"
            urgency_hours = 24
        elif erpi >= 42.0 or mol_so2 < min_mol_so2 or va >= 0.65:
            action_tier = "ACTIVE_MANAGEMENT"
            urgency_hours = 72
        elif erpi >= 20.0:
            action_tier = "ROUTINE_MONITORING"
            urgency_hours = 168  # 7 days
        else:
            action_tier = "STABLE_CELLAR"
            urgency_hours = 336  # 14 days

        # Generate Automated Prescriptive Work Orders
        work_orders: List[Dict[str, Any]] = []

        # Target Free SO2 calculation
        # Free SO2 needed for molecular target: Target = min_mol_so2 * (1 + 10^(pH - 1.81))
        target_free_so2 = min(45.0, max(22.0, min_mol_so2 * (1.0 + 10.0 ** (ph - 1.81))))

        if free_so2 < target_free_so2:
            pms_grams = self.calculate_pms_addition_grams(
                target_free_so2=target_free_so2,
                current_free_so2=free_so2,
                volume_liters=volume_l,
            )
            work_orders.append({
                "action": "ADD_SULPHUR_DIOXIDE",
                "dosage_grams": pms_grams,
                "chemical": "Potassium Metabisulfite (K2S2O5, 57.6% active SO2)",
                "target_free_so2_mg_l": round(target_free_so2, 1),
                "rationale": f"Current Free SO2 ({free_so2} mg/L) yields deficient molecular SO2 ({mol_so2:.2f} mg/L vs target {min_mol_so2} mg/L).",
                "hazard_prevention": "Prevents Brettanomyces decarboxylation, Acetobacter oxidation, and wild yeast bloom.",
            })

        if va >= 0.65:
            work_orders.append({
                "action": "VOLATILE_ACIDITY_TRIAGE",
                "chemical": "Cellar Temperature Reduction & Inert Gas Blanketing",
                "dosage_grams": 0.0,
                "target_free_so2_mg_l": None,
                "rationale": f"Elevated VA ({va} g/L) approaching critical threshold. Reduce cellar to 14°C and sparge headspace with Argon/N2.",
                "hazard_prevention": "Inhibits Acetobacter aceti acetic acid accumulation and ethyl acetate nail-polish odor.",
            })

        if malic_acid > 0.30 and status == "aging":
            work_orders.append({
                "action": "MLF_ARREST_OR_COMPLETION",
                "chemical": "Lysozyme (250 mg/L) or SO2 Fortification",
                "dosage_grams": round(0.25 * volume_l, 2),
                "target_free_so2_mg_l": None,
                "rationale": f"Unfinished malic acid ({malic_acid} g/L) in aging lot risks spontaneous post-bottling refermentation.",
                "hazard_prevention": "Microbial stabilization against lactic acid bacterial spoilage and diacetyl spikes.",
            })

        if maturation_days > 180 and vessel_type == "BARREL":
            work_orders.append({
                "action": "BARREL_TOPPING_AND_RACKING",
                "chemical": "Finished Reserve Wine Topping (Ouillage)",
                "dosage_grams": 0.0,
                "target_free_so2_mg_l": None,
                "rationale": f"Barrique maturation reached {maturation_days} days. Top ullage volume to prevent oxidative headspace.",
                "hazard_prevention": "Prevents aerobic film yeast growth and oxidative brown browning.",
            })

        if not work_orders:
            work_orders.append({
                "action": "NO_INTERVENTION_REQUIRED",
                "chemical": "N/A",
                "dosage_grams": 0.0,
                "target_free_so2_mg_l": None,
                "rationale": "Physicochemical parameters and preservative barriers are optimal.",
                "hazard_prevention": "Normal aging trajectory preserved.",
            })

        return {
            "vessel_id": vessel_id,
            "vessel_type": vessel_type,
            "cultivar": cultivar,
            "volume_liters": volume_l,
            "metrics": {
                "erpi_score": erpi,
                "molecular_so2_mg_l": round(mol_so2, 3),
                "minimum_required_molecular_so2": min_mol_so2,
                "free_so2_deficit_mg_l": round(max(0.0, target_free_so2 - free_so2), 1),
                "target_free_so2_mg_l": round(target_free_so2, 1),
            },
            "triage": {
                "action_tier": action_tier,
                "urgency_hours": urgency_hours,
                "risk_subscores": {
                    "so2_penalty": round(so2_penalty, 1),
                    "va_penalty": round(va_penalty, 1),
                    "mlf_risk_penalty": round(mlf_risk_penalty, 1),
                    "temp_penalty": round(temp_penalty, 1),
                    "sugar_hazard_penalty": round(sugar_hazard_penalty, 1),
                },
            },
            "work_orders": work_orders,
        }

    def dispatch_cellar_inventory(self, inventory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Triages an entire fleet of winery vessels, ranks them by ERPI priority,
        and aggregates cellar intervention metrics.
        """
        evaluated = [self.evaluate_vessel(v) for v in inventory]
        # Sort descending by ERPI risk score
        evaluated.sort(key=lambda x: x["metrics"]["erpi_score"], reverse=True)

        tier_counts = {
            "CRITICAL_INTERVENTION": sum(1 for v in evaluated if v["triage"]["action_tier"] == "CRITICAL_INTERVENTION"),
            "ACTIVE_MANAGEMENT": sum(1 for v in evaluated if v["triage"]["action_tier"] == "ACTIVE_MANAGEMENT"),
            "ROUTINE_MONITORING": sum(1 for v in evaluated if v["triage"]["action_tier"] == "ROUTINE_MONITORING"),
            "STABLE_CELLAR": sum(1 for v in evaluated if v["triage"]["action_tier"] == "STABLE_CELLAR"),
        }

        total_pms_grams = sum(
            wo["dosage_grams"]
            for v in evaluated
            for wo in v["work_orders"]
            if wo["action"] == "ADD_SULPHUR_DIOXIDE"
        )

        return {
            "total_vessels": len(evaluated),
            "tier_distribution": tier_counts,
            "cellar_health_score": round(100.0 - (sum(v["metrics"]["erpi_score"] for v in evaluated) / max(1, len(evaluated))), 1),
            "aggregate_pms_reagents_required_kg": round(total_pms_grams / 1000.0, 3),
            "dispatched_vessels": evaluated,
        }


# ============================================================================
# 4. HACCP Enological Food Safety & Biogenic Amine Risk Auditor
# ============================================================================

# International Legal Regulatory Standards (OIV / EU / US TTB)
OIV_MAX_TOTAL_SO2_RED = 150.0       # mg/L
OIV_MAX_TOTAL_SO2_WHITE = 200.0     # mg/L
OIV_MAX_VOLATILE_ACIDITY = 1.20     # g/L acetic acid equivalent
OIV_MIN_ALCOHOL = 8.5               # % v/v
OIV_MAX_FREE_SO2 = 50.0             # mg/L sensory ceiling


class EnologicalHaccpAuditor:
    """
    Automated HACCP Critical Control Point (CCP) verification engine
    and biogenic amine hazard risk assessment for enological production batches.
    """

    def __init__(self):
        pass

    def audit_wine_lot(
        self,
        lot_id: str,
        cultivar: str,
        ph: float,
        free_so2: float,
        total_so2: float,
        volatile_acidity: float,
        alcohol: float,
        residual_sugar: float = 2.0,
        malic_acid: float = 0.05,
        temperature: float = 16.0,
    ) -> Dict[str, Any]:
        """
        Executes multi-point HACCP CCP verification and biogenic amine risk scoring.
        """
        cultivar_type = cultivar.lower()
        max_total_so2 = OIV_MAX_TOTAL_SO2_WHITE if cultivar_type == "white" else OIV_MAX_TOTAL_SO2_RED

        # 1. CCP-1: Antimicrobial Preservative Sufficiency (Molecular SO2)
        mol_so2 = CellarMasterDispatchEngine.calculate_molecular_so2(free_so2, ph)
        target_mol_so2 = 0.80 if cultivar_type == "white" else 0.50
        ccp1_pass = bool(mol_so2 >= target_mol_so2)

        # 2. CCP-2: Legal Preservative Ceiling (Total SO2)
        ccp2_pass = bool(total_so2 <= max_total_so2)

        # 3. CCP-3: Volatile Acidity Spoilage Boundary (VA <= 1.20 g/L)
        ccp3_pass = bool(volatile_acidity <= OIV_MAX_VOLATILE_ACIDITY)

        # 4. CCP-4: Alcohol Legal Threshold (Alcohol >= 8.5% v/v)
        ccp4_pass = bool(alcohol >= OIV_MIN_ALCOHOL)

        # 5. Biogenic Amine Risk Scoring (Histamine, Tyramine, Putrescine)
        # Biogenic amines form via amino acid decarboxylation by lactic/spoilage bacteria.
        # High risk factors: pH > 3.65, incomplete MLF (malic > 0.2), Free SO2 < 15, temp > 18°C.
        bari = 0.0
        if ph > 3.65:
            bari += 30.0 * min(1.0, (ph - 3.65) / 0.45)
        if malic_acid > 0.15:
            bari += 25.0 * min(1.0, (malic_acid - 0.15) / 1.5)
        if free_so2 < 18.0:
            bari += 30.0 * (1.0 - (free_so2 / 18.0))
        if temperature > 18.0:
            bari += 15.0 * min(1.0, (temperature - 18.0) / 7.0)

        biogenic_amine_risk_score = float(round(max(0.0, min(100.0, bari)), 1))

        if biogenic_amine_risk_score < 25.0:
            amine_risk_level = "LOW"
            predicted_histamine_mg_l = "< 1.5 mg/L (Safe / Negligible)"
        elif biogenic_amine_risk_score < 60.0:
            amine_risk_level = "MODERATE"
            predicted_histamine_mg_l = "2.0 - 6.0 mg/L (Minor Allergen Risk)"
        else:
            amine_risk_level = "HIGH"
            predicted_histamine_mg_l = "> 8.0 mg/L (Histamine Headache / Allergy Hazard)"

        all_ccp_passed = bool(ccp1_pass and ccp2_pass and ccp3_pass and ccp4_pass)

        # Audit findings and remediation guidance
        findings: List[str] = []
        if not ccp1_pass:
            findings.append(f"CCP-1 FAIL: Molecular SO2 ({mol_so2:.3f} mg/L) below sanitary minimum ({target_mol_so2} mg/L). Wine vulnerable to Brettanomyces.")
        if not ccp2_pass:
            findings.append(f"CCP-2 FAIL: Total SO2 ({total_so2} mg/L) exceeds OIV legal ceiling ({max_total_so2} mg/L). Cannot be legally sold.")
        if not ccp3_pass:
            findings.append(f"CCP-3 FAIL: Volatile acidity ({volatile_acidity} g/L) exceeds statutory limit ({OIV_MAX_VOLATILE_ACIDITY} g/L).")
        if not ccp4_pass:
            findings.append(f"CCP-4 FAIL: Alcohol ({alcohol}%) below statutory wine threshold ({OIV_MIN_ALCOHOL}%).")
        if amine_risk_level == "HIGH":
            findings.append("HAZARD WARNING: High biogenic amine synthesis conditions detected due to elevated pH and sub-lethal sulfur barriers.")

        if all_ccp_passed:
            certification_status = "CERTIFIED_FOOD_GRADE"
            summary_statement = "Wine lot complies with all OIV, EU, and TTB statutory enological food safety standards."
        else:
            certification_status = "NON_COMPLIANT_HOLD"
            summary_statement = "Lot quarantined under enological hold pending corrective remediation."

        return {
            "lot_id": lot_id,
            "cultivar": cultivar_type,
            "audit_status": certification_status,
            "all_ccp_passed": all_ccp_passed,
            "summary": summary_statement,
            "ccp_verifications": {
                "ccp1_antimicrobial_molecular_so2": {
                    "passed": ccp1_pass,
                    "measured_molecular_so2_mg_l": round(mol_so2, 3),
                    "required_minimum_mg_l": target_mol_so2,
                },
                "ccp2_total_so2_statutory_ceiling": {
                    "passed": ccp2_pass,
                    "measured_total_so2_mg_l": total_so2,
                    "statutory_limit_mg_l": max_total_so2,
                },
                "ccp3_volatile_acidity_ceiling": {
                    "passed": ccp3_pass,
                    "measured_va_g_l": volatile_acidity,
                    "statutory_limit_g_l": OIV_MAX_VOLATILE_ACIDITY,
                },
                "ccp4_statutory_alcohol_floor": {
                    "passed": ccp4_pass,
                    "measured_alcohol_pct": alcohol,
                    "statutory_minimum_pct": OIV_MIN_ALCOHOL,
                },
            },
            "biogenic_amine_risk": {
                "risk_index_score": biogenic_amine_risk_score,
                "risk_level": amine_risk_level,
                "predicted_histamine_content": predicted_histamine_mg_l,
            },
            "audit_findings": findings,
        }
