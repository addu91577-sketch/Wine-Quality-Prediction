"""
Global Cold-Chain Transportation Stress Kinetics, Isotopic Terroir Fraud Detection,
Consumer Hedonic Market Valuation, and Unified Digital Winery Twin Platform.

Modules & Capabilities:
- ColdChainLogisticsSimulator: Models dynamic thermal excursions (Arrhenius Thermal Damage
  Units TDU, Degree-Hours > 25C and > 35C), headspace barometric pressure differentials
  (Delta P = P_headspace - P_ambient) during air cargo and high-altitude transits, closure
  seal integrity / cork extrusion risks ("leakers"), and hydrodynamic vibration energy.
- IsotopicTerroirAndFraudAuditor: Stable Isotope Ratio Analysis (delta13C VPDB for C3 grape
  vs C4 cane/corn sugar chaptalization, delta18O VSMOW for exogenous water dilution),
  87Sr/86Sr bedrock strontium isotope ratios, and multielemental ICP-MS mineral profiling
  (Li, Rb, Sr, Ba, Mn, B) with Mahalanobis distance anomaly detection for AOC/DOC/AVA authenticity.
- ConsumerHedonicAndMarketPricingOptimizer: Rosen's hedonic pricing econometric model
  estimating fair market bottle valuation ($/bottle), multi-regional consumer palate
  alignment scoring (North America, Continental Europe, East Asia, Nordic Monopoly), and
  price elasticity of demand margin optimization.
- UnifiedDigitalWineryTwinPlatform: End-to-end lifecycle simulator integrating grape intake,
  fermentation kinetics, malolactic bio-transformation, oak maturation, stabilization,
  crossflow microfiltration, bottling packaging, logistics, and release pricing, paired with
  Scope 1/2/3 greenhouse gas (GHG) Life Cycle Assessment (LCA) and water footprint scoring.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

from src.feature_engineering import create_engineered_features


# ============================================================================
# Physical, Chemical & Econometric Reference Constants
# ============================================================================

# Universal Gas Constant (J / (mol * K))
GAS_CONSTANT_R = 8.31446

# Reference Temperatures
TEMP_REF_CELLAR_K = 287.15    # 14 C baseline cellar storage
TEMP_ZERO_C_K = 273.15        # 0 C in Kelvin

# Atmospheric Pressures (kPa)
P_ATM_SEA_LEVEL = 101.325     # Sea level standard atmospheric pressure
P_ATM_AIR_CARGO_CRUISE = 75.2 # Standard pressurized aircraft cargo hold at 8,000 ft equivalent
P_ATM_MOUNTAIN_PASS = 80.0    # Truck transit over high-altitude alpine pass (~2,000 m)

# Enological Arrhenius Activation Energies (J/mol)
EA_ESTER_HYDROLYSIS = 78500.0   # Fruity aroma ester degradation (isoamyl acetate -> acetic acid)
EA_SO2_OXIDATION = 62000.0       # Free SO2 oxidative consumption acceleration
EA_BROWNING_PIGMENT = 84000.0    # Non-enzymatic quinone-tannin browning (A420)
EA_TDN_FORMATION = 92000.0       # 1,1,6-trimethyl-1,2-dihydronaphthalene (kerosene off-flavor)

# Closure Holding Differential Pressure Thresholds (kPa)
CLOSURE_PRESSURE_TOLERANCES: Dict[str, Dict[str, float]] = {
    "natural_cork_fleur": {"weep_threshold_kpa": 90.0, "extrude_threshold_kpa": 135.0},
    "technical_diam": {"weep_threshold_kpa": 115.0, "extrude_threshold_kpa": 160.0},
    "synthetic_nomacorc": {"weep_threshold_kpa": 85.0, "extrude_threshold_kpa": 125.0},
    "screwcap_sarantin": {"weep_threshold_kpa": 150.0, "extrude_threshold_kpa": 220.0},
    "screwcap_saranex": {"weep_threshold_kpa": 130.0, "extrude_threshold_kpa": 190.0},
}

# Stable Isotope Reference Baselines (per mil)
DELTA13C_C3_GRAPES_MEAN = -27.2       # Authentic Vitis vinifera C3 photosynthetic cycle
DELTA13C_C3_GRAPES_STD = 0.85
DELTA13C_C4_CHAPTALIZATION = -13.1    # Exogenous cane or maize sugar (C4 Hatch-Slack cycle)

DELTA18O_AUTHENTIC_WATER_MEAN = 3.8   # Grape berry water evapotranspiration enrichment
DELTA18O_AUTHENTIC_WATER_STD = 1.1
DELTA18O_TAP_WATER_DILUTION = -7.5    # Municipal / groundwater dilution profile

# Reference Terroir Bedrock Geochemical Profiles (ICP-MS mean and covariance specs)
TERROIR_PROFILES: Dict[str, Dict[str, Any]] = {
    "bordeaux_left_bank": {
        "appellation": "Bordeaux Left Bank (Pauillac / Margaux)",
        "soil_type": "Quaternary alluvial gravel over limestone",
        "sr_87_86_ratio": 0.7088,
        "sr_87_86_std": 0.0006,
        "minerals_ppm": {
            "lithium": 0.024,
            "rubidium": 0.85,
            "strontium": 0.48,
            "barium": 0.12,
            "manganese": 1.45,
            "boron": 4.80,
        },
        "mineral_stds": {
            "lithium": 0.006,
            "rubidium": 0.15,
            "strontium": 0.08,
            "barium": 0.03,
            "manganese": 0.25,
            "boron": 0.70,
        }
    },
    "burgundy_cote_dor": {
        "appellation": "Burgundy Côte d'Or (Côte de Nuits / Côte de Beaune)",
        "soil_type": "Bajocian and Bathonian Jurassic marine limestone and marl",
        "sr_87_86_ratio": 0.7081,
        "sr_87_86_std": 0.0004,
        "minerals_ppm": {
            "lithium": 0.015,
            "rubidium": 0.52,
            "strontium": 0.72,
            "barium": 0.08,
            "manganese": 0.88,
            "boron": 3.60,
        },
        "mineral_stds": {
            "lithium": 0.004,
            "rubidium": 0.11,
            "strontium": 0.10,
            "barium": 0.02,
            "manganese": 0.18,
            "boron": 0.55,
        }
    },
    "napa_valley_rutherford": {
        "appellation": "Napa Valley Rutherford Bench",
        "soil_type": "Franciscan volcanic complex and Bale gravelly loam",
        "sr_87_86_ratio": 0.7065,
        "sr_87_86_std": 0.0005,
        "minerals_ppm": {
            "lithium": 0.045,
            "rubidium": 1.25,
            "strontium": 0.35,
            "barium": 0.22,
            "manganese": 2.10,
            "boron": 6.20,
        },
        "mineral_stds": {
            "lithium": 0.009,
            "rubidium": 0.22,
            "strontium": 0.06,
            "barium": 0.04,
            "manganese": 0.35,
            "boron": 0.90,
        }
    },
    "rioja_alta": {
        "appellation": "Rioja Alta",
        "soil_type": "Calcareous clay over Tertiary sandstone terraces",
        "sr_87_86_ratio": 0.7093,
        "sr_87_86_std": 0.0005,
        "minerals_ppm": {
            "lithium": 0.032,
            "rubidium": 0.74,
            "strontium": 0.59,
            "barium": 0.15,
            "manganese": 1.20,
            "boron": 4.10,
        },
        "mineral_stds": {
            "lithium": 0.007,
            "rubidium": 0.14,
            "strontium": 0.09,
            "barium": 0.03,
            "manganese": 0.22,
            "boron": 0.65,
        }
    },
    "douro_schist": {
        "appellation": "Douro Superior / Cima Corgo",
        "soil_type": "Pre-Cambrian and Cambrian metamorphic schist bedrock",
        "sr_87_86_ratio": 0.7165,
        "sr_87_86_std": 0.0008,
        "minerals_ppm": {
            "lithium": 0.058,
            "rubidium": 1.65,
            "strontium": 0.28,
            "barium": 0.31,
            "manganese": 2.85,
            "boron": 5.40,
        },
        "mineral_stds": {
            "lithium": 0.012,
            "rubidium": 0.30,
            "strontium": 0.05,
            "barium": 0.06,
            "manganese": 0.45,
            "boron": 0.85,
        }
    }
}

# Regional Consumer Palate Preferences
REGIONAL_PALATE_PROFILES: Dict[str, Dict[str, Any]] = {
    "north_america": {
        "name": "North American Market (US & Canada)",
        "ideal_alcohol": 14.0,
        "ideal_ph": 3.65,
        "ideal_residual_sugar": 3.2,
        "ideal_sulphates": 0.70,
        "ideal_volatile_acidity": 0.45,
        "oak_affinity_weight": 1.35,
        "fruit_ripeness_affinity": 1.40,
        "acid_tolerance_penalty": 1.25,
        "base_wtp_usd": 28.50,
        "price_elasticity": -1.25,
    },
    "continental_europe": {
        "name": "Continental European Market (France, Germany, Italy)",
        "ideal_alcohol": 12.8,
        "ideal_ph": 3.32,
        "ideal_residual_sugar": 1.4,
        "ideal_sulphates": 0.58,
        "ideal_volatile_acidity": 0.48,
        "oak_affinity_weight": 0.85,
        "fruit_ripeness_affinity": 0.95,
        "acid_tolerance_penalty": 0.60,
        "base_wtp_usd": 24.00,
        "price_elasticity": -1.55,
    },
    "east_asia": {
        "name": "East Asian Market (Japan, South Korea, Singapore)",
        "ideal_alcohol": 12.5,
        "ideal_ph": 3.42,
        "ideal_residual_sugar": 2.2,
        "ideal_sulphates": 0.52,
        "ideal_volatile_acidity": 0.38,
        "oak_affinity_weight": 0.90,
        "fruit_ripeness_affinity": 1.10,
        "acid_tolerance_penalty": 0.85,
        "base_wtp_usd": 34.00,
        "price_elasticity": -0.95,
    },
    "nordic_monopoly": {
        "name": "Nordic Monopolies (Systembolaget, Alko, Vinmonopolet)",
        "ideal_alcohol": 13.0,
        "ideal_ph": 3.38,
        "ideal_residual_sugar": 1.8,
        "ideal_sulphates": 0.50,
        "ideal_volatile_acidity": 0.40,
        "oak_affinity_weight": 0.80,
        "fruit_ripeness_affinity": 1.00,
        "acid_tolerance_penalty": 0.75,
        "base_wtp_usd": 26.00,
        "price_elasticity": -1.35,
    }
}


# ============================================================================
# 1. Cold-Chain Transportation Stress Simulator
# ============================================================================

class ColdChainLogisticsSimulator:
    """
    Simulates global intermodal wine transportation stress across maritime reefer,
    dry container, refrigerated truck, and air cargo routes.

    Evaluates:
    1. Dynamic temperature profiles and Arrhenius Thermal Damage Units (TDU).
    2. Headspace gas thermal expansion and barometric pressure differentials (Delta P).
    3. Closure creep, seal weeping ("weepers"), and cork extrusion risk.
    4. Hydrodynamic vibration-induced colloidal destabilization.
    """

    def __init__(self):
        self.transit_presets: Dict[str, Dict[str, Any]] = {
            "maritime_reefer_rotterdam_singapore": {
                "name": "Maritime Reefer: Rotterdam to Singapore (28 Days)",
                "mode": "maritime_reefer",
                "duration_days": 28,
                "ambient_temp_mean_c": 14.2,
                "ambient_temp_amplitude_c": 1.2,
                "ambient_pressure_kpa": P_ATM_SEA_LEVEL,
                "vibration_rms_g": 0.04,
            },
            "dry_container_deck_panama": {
                "name": "Dry Container Top Deck: Trans-Panama Canal (24 Days)",
                "mode": "dry_container",
                "duration_days": 24,
                "ambient_temp_mean_c": 31.5,
                "ambient_temp_amplitude_c": 11.5,  # Peaks at 43 C
                "ambient_pressure_kpa": P_ATM_SEA_LEVEL,
                "vibration_rms_g": 0.06,
            },
            "cross_country_truck_express": {
                "name": "Intermodal Express Truck: California to New York (5 Days)",
                "mode": "truck_ambient",
                "duration_days": 5,
                "ambient_temp_mean_c": 24.0,
                "ambient_temp_amplitude_c": 9.0,   # High diurnal cycling
                "ambient_pressure_kpa": P_ATM_MOUNTAIN_PASS,  # Passes over Rockies
                "vibration_rms_g": 0.18,           # Highway shock
            },
            "air_freight_cargo_london_tokyo": {
                "name": "Air Cargo Freight: London Heathrow to Tokyo Narita (2 Days)",
                "mode": "air_cargo",
                "duration_days": 2,
                "ambient_temp_mean_c": 19.5,
                "ambient_temp_amplitude_c": 3.0,
                "ambient_pressure_kpa": P_ATM_AIR_CARGO_CRUISE,  # 75.2 kPa
                "vibration_rms_g": 0.12,
            }
        }

    def simulate_thermal_trajectory(
        self,
        duration_days: float,
        mean_temp_c: float,
        diurnal_amplitude_c: float,
        thermal_shock_spikes: Optional[List[Tuple[float, float, float]]] = None, # [(day_start, duration_days, peak_temp_c)]
        time_step_hours: float = 1.0
    ) -> pd.DataFrame:
        """
        Simulates hourly temperature exposure and kinetic Arrhenius damage accumulation.
        """
        total_hours = int(duration_days * 24.0)
        hours = np.arange(total_hours, dtype=float) * time_step_hours
        days = hours / 24.0

        # Baseline diurnal sinusoidal thermal cycle
        temp_c = mean_temp_c + diurnal_amplitude_c * np.sin(2.0 * np.pi * days - np.pi / 2.0)

        # Apply specific thermal shock events (e.g. tarmac sitting or customs inspection delay)
        if thermal_shock_spikes:
            for start_day, dur_days, peak_c in thermal_shock_spikes:
                mask = (days >= start_day) & (days <= (start_day + dur_days))
                if np.any(mask):
                    fraction = (days[mask] - start_day) / max(dur_days, 1e-4)
                    spike = (peak_c - mean_temp_c) * np.sin(np.pi * fraction)
                    temp_c[mask] = np.maximum(temp_c[mask], mean_temp_c + spike)

        temp_k = temp_c + TEMP_ZERO_C_K

        # Kinetic damage acceleration factor relative to 14 C reference cellar
        # Rel_Rate = exp(-Ea/R * (1/T - 1/T_ref))
        rel_rate_ester = np.exp(- (EA_ESTER_HYDROLYSIS / GAS_CONSTANT_R) * (1.0 / temp_k - 1.0 / TEMP_REF_CELLAR_K))
        rel_rate_so2 = np.exp(- (EA_SO2_OXIDATION / GAS_CONSTANT_R) * (1.0 / temp_k - 1.0 / TEMP_REF_CELLAR_K))
        rel_rate_browning = np.exp(- (EA_BROWNING_PIGMENT / GAS_CONSTANT_R) * (1.0 / temp_k - 1.0 / TEMP_REF_CELLAR_K))

        # Arrhenius Thermal Damage Units (TDU) = Cumulative normalized aging hours
        dt = time_step_hours
        cum_tdu = np.cumsum(rel_rate_ester * dt)

        # Degree-Hours above thresholds
        deg_hrs_25 = np.cumsum(np.maximum(0.0, temp_c - 25.0) * dt)
        deg_hrs_35 = np.cumsum(np.maximum(0.0, temp_c - 35.0) * dt)

        return pd.DataFrame({
            "hour": hours,
            "day": days,
            "temp_c": temp_c,
            "temp_k": temp_k,
            "rel_rate_ester": rel_rate_ester,
            "rel_rate_so2": rel_rate_so2,
            "rel_rate_browning": rel_rate_browning,
            "cumulative_tdu_hours": cum_tdu,
            "cumulative_deg_hrs_gt25": deg_hrs_25,
            "cumulative_deg_hrs_gt35": deg_hrs_35
        })

    def calculate_headspace_pressure_differential(
        self,
        liquid_temp_c: float,
        ambient_pressure_kpa: float = P_ATM_SEA_LEVEL,
        bottling_temp_c: float = 18.0,
        headspace_volume_ml: float = 4.5,
        dissolved_co2_mg_l: float = 850.0,
        liquid_volume_ml: float = 750.0,
        initial_headspace_vacuum_kpa: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculates the internal headspace pressure and net pressure differential (Delta P)
        exerted across the closure seal as a function of temperature and altitude.
        """
        t_bot_k = bottling_temp_c + TEMP_ZERO_C_K
        t_cur_k = liquid_temp_c + TEMP_ZERO_C_K

        # 1. Thermal expansion of liquid wine (volumetric expansion coef ~ 0.00035 / K)
        beta_wine = 0.00038
        delta_v_liquid_ml = liquid_volume_ml * beta_wine * (liquid_temp_c - bottling_temp_c)
        effective_headspace_ml = max(0.5, headspace_volume_ml - delta_v_liquid_ml)

        # 2. Air / inert gas compression in headspace (P1 V1 / T1 = P2 V2 / T2)
        initial_gas_p_kpa = P_ATM_SEA_LEVEL - initial_headspace_vacuum_kpa
        headspace_gas_p_kpa = initial_gas_p_kpa * (headspace_volume_ml / effective_headspace_ml) * (t_cur_k / t_bot_k)

        # 3. Dissolved CO2 Henry's law degassing into headspace at elevated temperature
        # Henry's constant decreases with temperature: CO2 desorbs into headspace
        henry_20c = 1.45  # g/(L*bar)
        henry_t = henry_20c * np.exp(2200.0 * (1.0 / t_cur_k - 1.0 / (20.0 + TEMP_ZERO_C_K)))
        # Additional CO2 partial pressure in headspace (kPa)
        co2_partial_p_kpa = max(0.0, (dissolved_co2_mg_l / 1000.0) / max(henry_t, 0.1)) * 10.0

        total_internal_pressure_kpa = headspace_gas_p_kpa + co2_partial_p_kpa
        delta_p_kpa = total_internal_pressure_kpa - ambient_pressure_kpa

        return {
            "liquid_temp_c": float(liquid_temp_c),
            "ambient_pressure_kpa": float(ambient_pressure_kpa),
            "effective_headspace_ml": float(round(effective_headspace_ml, 3)),
            "gas_partial_pressure_kpa": float(round(headspace_gas_p_kpa, 2)),
            "co2_partial_pressure_kpa": float(round(co2_partial_p_kpa, 2)),
            "total_internal_pressure_kpa": float(round(total_internal_pressure_kpa, 2)),
            "delta_p_differential_kpa": float(round(delta_p_kpa, 2))
        }

    def evaluate_transit_scenario(
        self,
        preset_name: str,
        closure_type: str = "natural_cork_fleur",
        initial_free_so2: float = 34.0,
        initial_a420_browning: float = 0.120,
        dissolved_co2_mg_l: float = 850.0
    ) -> Dict[str, Any]:
        """
        Runs complete transportation stress evaluation for a named shipping route.
        """
        if preset_name not in self.transit_presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.transit_presets.keys())}")

        preset = self.transit_presets[preset_name]
        dur_days = preset["duration_days"]
        mean_c = preset["ambient_temp_mean_c"]
        amp_c = preset["ambient_temp_amplitude_c"]
        p_amb = preset["ambient_pressure_kpa"]
        vib_g = preset["vibration_rms_g"]

        # Run thermal trajectory
        traj_df = self.simulate_thermal_trajectory(dur_days, mean_c, amp_c)
        max_temp_c = float(traj_df["temp_c"].max())
        min_temp_c = float(traj_df["temp_c"].min())
        avg_temp_c = float(traj_df["temp_c"].mean())

        total_tdu_hours = float(traj_df["cumulative_tdu_hours"].iloc[-1])
        deg_hrs_25 = float(traj_df["cumulative_deg_hrs_gt25"].iloc[-1])
        deg_hrs_35 = float(traj_df["cumulative_deg_hrs_gt35"].iloc[-1])

        # Equivalent cellar aging months suffered in transit:
        # TDU hours / (24 hours * 30.4 days baseline)
        equivalent_aging_months = total_tdu_hours / (24.0 * 30.417)

        # Chemical degradation during transit
        # Free SO2 consumption accelerated by temperature:
        so2_loss_mg_l = (total_tdu_hours / (24.0 * 30.417)) * 0.45 + (deg_hrs_35 / 100.0) * 1.5
        so2_loss_mg_l = min(initial_free_so2 * 0.75, so2_loss_mg_l)
        final_free_so2 = max(0.0, initial_free_so2 - so2_loss_mg_l)

        # Browning index progression (Delta A420)
        delta_a420 = (total_tdu_hours / (24.0 * 30.417)) * 0.0035 + (deg_hrs_35 / 100.0) * 0.015
        final_a420 = initial_a420_browning + delta_a420

        # Fruity ester retention fraction
        ester_retention_pct = max(10.0, 100.0 * np.exp(- 0.0006 * total_tdu_hours))

        # Peak headspace pressure differential
        p_diff = self.calculate_headspace_pressure_differential(
            liquid_temp_c=max_temp_c,
            ambient_pressure_kpa=p_amb,
            dissolved_co2_mg_l=dissolved_co2_mg_l
        )
        peak_delta_p = p_diff["delta_p_differential_kpa"]

        # Closure seal vulnerability check
        tol = CLOSURE_PRESSURE_TOLERANCES.get(closure_type, CLOSURE_PRESSURE_TOLERANCES["natural_cork_fleur"])
        weep_tol = tol["weep_threshold_kpa"]
        extrude_tol = tol["extrude_threshold_kpa"]

        if peak_delta_p >= extrude_tol:
            seal_status = "CRITICAL_FAILURE_EXTRUSION"
            risk_score = 95
        elif peak_delta_p >= weep_tol:
            seal_status = "WARNING_WEEPER_LEAKAGE"
            risk_score = 65
        elif deg_hrs_35 > 10.0 or deg_hrs_25 > 200.0:
            seal_status = "MODERATE_THERMAL_DEGRADATION"
            risk_score = 45
        else:
            seal_status = "PASS_OPTIMAL_COLD_CHAIN"
            risk_score = 10

        return {
            "preset_name": preset_name,
            "route_description": preset["name"],
            "transit_mode": preset["mode"],
            "duration_days": dur_days,
            "closure_type": closure_type,
            "temperature_summary": {
                "min_c": round(min_temp_c, 2),
                "max_c": round(max_temp_c, 2),
                "mean_c": round(avg_temp_c, 2),
            },
            "kinetic_stress_metrics": {
                "total_tdu_hours": round(total_tdu_hours, 1),
                "equivalent_cellar_aging_months": round(equivalent_aging_months, 2),
                "degree_hours_above_25c": round(deg_hrs_25, 1),
                "degree_hours_above_35c": round(deg_hrs_35, 1),
            },
            "chemical_impact": {
                "initial_free_so2_mg_l": round(initial_free_so2, 1),
                "final_free_so2_mg_l": round(final_free_so2, 1),
                "so2_loss_mg_l": round(so2_loss_mg_l, 1),
                "initial_a420": round(initial_a420_browning, 3),
                "final_a420": round(final_a420, 3),
                "delta_a420": round(delta_a420, 3),
                "fruity_ester_retention_pct": round(ester_retention_pct, 1),
            },
            "barometric_seal_integrity": {
                "ambient_pressure_kpa": p_amb,
                "peak_internal_pressure_kpa": p_diff["total_internal_pressure_kpa"],
                "peak_delta_p_kpa": peak_delta_p,
                "weep_threshold_kpa": weep_tol,
                "extrude_threshold_kpa": extrude_tol,
                "seal_status": seal_status,
                "vibration_rms_g": vib_g,
                "overall_transit_risk_score": risk_score,
            }
        }


# ============================================================================
# 2. Isotopic Terroir & Anti-Counterfeit Fraud Auditor
# ============================================================================

class IsotopicTerroirAndFraudAuditor:
    """
    Evaluates stable isotope ratios (delta13C, delta18O), radiogenic strontium (87Sr/86Sr),
    and multielemental ICP-MS trace mineral fingerprints against verified AOC/DOC/AVA
    appellations to detect fraud, chaptalization, water dilution, and provenance forgery.
    """

    def __init__(self):
        self.terroir_profiles = TERROIR_PROFILES

    def calculate_chaptalization_fraction(
        self,
        measured_delta13c: float,
        baseline_grape_delta13c: float = DELTA13C_C3_GRAPES_MEAN
    ) -> Dict[str, Any]:
        """
        Calculates exogenous C4 sugar chaptalization fraction using stable carbon isotope mass balance.
        delta13C_mix = f_grape * delta13C_grape + f_c4 * delta13C_c4
        where f_grape + f_c4 = 1.0.
        """
        # delta13C_c4 is ~ -13.1, delta13C_c3 is ~ -27.2
        # If measured is -24.0, f_c4 = (-24.0 - (-27.2)) / (-13.1 - (-27.2)) = 3.2 / 14.1 = 22.7%
        denominator = DELTA13C_C4_CHAPTALIZATION - baseline_grape_delta13c
        numerator = measured_delta13c - baseline_grape_delta13c

        f_c4 = max(0.0, numerator / denominator)
        f_c4 = min(1.0, f_c4)
        c4_sugar_pct = f_c4 * 100.0

        # OIV legal limit: Natural un-chaptalized wine is delta13C <= -25.5 per mil
        is_adulterated = bool(measured_delta13c > -25.5)
        severity = "LEGAL_AUTHENTIC"
        if measured_delta13c > -22.0:
            severity = "SEVERE_EXOGENOUS_C4_CHAPTALIZATION"
        elif measured_delta13c > -25.5:
            severity = "SUSPECT_MODERATE_CHAPTALIZATION"

        return {
            "measured_delta13c_permil": float(round(measured_delta13c, 2)),
            "baseline_grape_delta13c_permil": float(round(baseline_grape_delta13c, 2)),
            "estimated_c4_sugar_fraction_pct": float(round(c4_sugar_pct, 2)),
            "is_adulterated": is_adulterated,
            "compliance_verdict": severity
        }

    def calculate_water_dilution_fraction(
        self,
        measured_delta18o: float,
        expected_berry_delta18o: float = DELTA18O_AUTHENTIC_WATER_MEAN
    ) -> Dict[str, Any]:
        """
        Calculates exogenous water addition (watering down) using oxygen stable isotope ratio.
        """
        # delta18O_mix = f_wine * delta18O_berry + f_water * delta18O_tap
        denominator = DELTA18O_TAP_WATER_DILUTION - expected_berry_delta18o
        numerator = measured_delta18o - expected_berry_delta18o

        f_tap = max(0.0, numerator / denominator)
        f_tap = min(1.0, f_tap)
        water_addition_pct = f_tap * 100.0

        is_diluted = bool(measured_delta18o < 1.5)
        severity = "LEGAL_NATURAL_WATER"
        if measured_delta18o < -1.0:
            severity = "SEVERE_EXOGENOUS_WATER_ADDITION"
        elif measured_delta18o < 1.5:
            severity = "SUSPECT_WATER_DILUTION"

        return {
            "measured_delta18o_permil": float(round(measured_delta18o, 2)),
            "expected_berry_delta18o_permil": float(round(expected_berry_delta18o, 2)),
            "estimated_exogenous_water_pct": float(round(water_addition_pct, 2)),
            "is_diluted": is_diluted,
            "compliance_verdict": severity
        }

    def verify_appellation_terroir(
        self,
        claimed_appellation: str,
        sample_minerals: Dict[str, float],
        measured_sr_ratio: float,
        measured_delta13c: float = -27.0,
        measured_delta18o: float = 3.5
    ) -> Dict[str, Any]:
        """
        Verifies geographical provenance against reference terroir database using
        Mahalanobis distance across geochemical elements and isotope signatures.
        """
        if claimed_appellation not in self.terroir_profiles:
            raise ValueError(f"Unknown terroir profile: {claimed_appellation}. Available: {list(self.terroir_profiles.keys())}")

        profile = self.terroir_profiles[claimed_appellation]
        ref_minerals = profile["minerals_ppm"]
        ref_stds = profile["mineral_stds"]
        ref_sr = profile["sr_87_86_ratio"]
        ref_sr_std = profile["sr_87_86_std"]

        # 1. Z-scores across mineral components
        z_scores = {}
        sq_dist_sum = 0.0
        element_count = 0

        for elem, ref_val in ref_minerals.items():
            meas_val = sample_minerals.get(elem, ref_val)
            std_val = ref_stds.get(elem, 0.1 * ref_val)
            z = (meas_val - ref_val) / max(std_val, 1e-5)
            z_scores[elem] = round(float(z), 3)
            sq_dist_sum += z ** 2
            element_count += 1

        # Strontium isotope Z-score
        z_sr = (measured_sr_ratio - ref_sr) / max(ref_sr_std, 1e-5)
        z_scores["sr_87_86_ratio"] = round(float(z_sr), 3)
        sq_dist_sum += (z_sr ** 2) * 2.5  # Strontium ratio heavily weighted (geological tracer)
        element_count += 1

        # Pseudo-Mahalanobis Terroir Distance
        mahalanobis_d = math.sqrt(sq_dist_sum / max(element_count, 1))

        # Check isotopic adulterations
        chapt_res = self.calculate_chaptalization_fraction(measured_delta13c)
        water_res = self.calculate_water_dilution_fraction(measured_delta18o)

        # Authenticity classification
        # Chi-square p-value proxy
        p_val_approx = math.exp(- 0.5 * (mahalanobis_d ** 2))
        is_authentic = bool(mahalanobis_d <= 2.8 and not chapt_res["is_adulterated"] and not water_res["is_diluted"])

        if is_authentic:
            verdict = "VERIFIED_AUTHENTIC_TERROIR"
            confidence_pct = max(70.0, min(99.5, 100.0 * (1.0 - mahalanobis_d / 4.0)))
        elif chapt_res["is_adulterated"]:
            verdict = "FRAUD_DETECTED_C4_CHAPTALIZATION"
            confidence_pct = 95.0
        elif water_res["is_diluted"]:
            verdict = "FRAUD_DETECTED_EXOGENOUS_WATER"
            confidence_pct = 92.0
        else:
            verdict = "PROVENANCE_MISMATCH_SUSPECT_COUNTERFEIT"
            confidence_pct = max(75.0, min(99.0, 50.0 + mahalanobis_d * 12.0))

        return {
            "claimed_appellation": claimed_appellation,
            "appellation_name": profile["appellation"],
            "bedrock_geology": profile["soil_type"],
            "mahalanobis_terroir_distance": round(mahalanobis_d, 2),
            "terroir_p_value": round(p_val_approx, 4),
            "is_authentic": is_authentic,
            "verdict": verdict,
            "confidence_pct": round(confidence_pct, 1),
            "elemental_z_scores": z_scores,
            "isotopic_audits": {
                "chaptalization": chapt_res,
                "water_dilution": water_res,
                "measured_sr_87_86": round(measured_sr_ratio, 4),
                "expected_sr_87_86": round(ref_sr, 4),
            }
        }


# ============================================================================
# 3. Consumer Hedonic Econometrics & Market Valuation Optimizer
# ============================================================================

class ConsumerHedonicAndMarketPricingOptimizer:
    """
    Implements Rosen's hedonic pricing econometric model, cross-regional consumer
    palate alignment, and demand price-elasticity profit optimization.
    """

    def __init__(self):
        self.regional_profiles = REGIONAL_PALATE_PROFILES

    def estimate_bottle_market_value(
        self,
        predicted_quality: float,
        cultivar: str = "red",
        alcohol: float = 13.5,
        ph: float = 3.45,
        residual_sugar: float = 2.0,
        volatile_acidity: float = 0.42,
        free_so2: float = 30.0,
        cellar_aging_years: float = 2.0,
        oak_extract_index: float = 0.75,
        appellation_tier: str = "doc_quality" # table, igt, doc_quality, grand_cru
    ) -> Dict[str, Any]:
        """
        Hedonic pricing regression estimating equilibrium bottle release value ($/750mL).
        ln(Price) = beta0 + beta_q * Quality + beta_alc * Alcohol + ...
        """
        # Tier premiums
        tier_multipliers = {
            "table": 0.55,
            "igt": 0.85,
            "doc_quality": 1.25,
            "grand_cru": 2.60
        }
        tier_factor = tier_multipliers.get(appellation_tier, 1.0)

        # Baseline log price formulation
        ln_base = 2.10  # ~ $8.16 base table wine

        # Quality elasticity: each point above 5.0 brings substantial nonlinear exponential premium
        quality_delta = predicted_quality - 5.0
        q_premium = 0.42 * quality_delta + 0.12 * (quality_delta ** 2 if quality_delta > 0 else -0.05 * (quality_delta ** 2))

        # Alcohol body contribution (optimal 13.2 - 14.2)
        alc_effect = 0.08 * (alcohol - 12.0) if alcohol >= 12.0 else -0.15 * (12.0 - alcohol)

        # Acid balance (pH 3.3 - 3.5 optimal)
        ph_dev = abs(ph - 3.40)
        ph_penalty = -0.35 * ph_dev

        # Volatile acidity discount
        va_penalty = -1.20 * max(0.0, volatile_acidity - 0.40)

        # Aging premium (exponential maturation value)
        aging_premium = 0.10 * math.sqrt(max(0.0, cellar_aging_years))

        # Oak integration
        oak_premium = 0.15 * min(1.0, oak_extract_index)

        ln_price = ln_base + q_premium + alc_effect + ph_penalty + va_penalty + aging_premium + oak_premium
        est_price_usd = math.exp(ln_price) * tier_factor

        # 90% confidence interval (+/- 18%)
        ci_lower = est_price_usd * 0.82
        ci_upper = est_price_usd * 1.22

        return {
            "predicted_quality": round(predicted_quality, 2),
            "appellation_tier": appellation_tier,
            "estimated_market_price_usd": round(est_price_usd, 2),
            "confidence_interval_90_usd": (round(ci_lower, 2), round(ci_upper, 2)),
            "hedonic_components": {
                "quality_premium_contribution_pct": round(float(q_premium * 100.0), 1),
                "aging_premium_pct": round(float(aging_premium * 100.0), 1),
                "tier_multiplier": tier_factor,
                "volatile_acidity_penalty_pct": round(float(va_penalty * 100.0), 1),
            }
        }

    def calculate_regional_palate_alignment(
        self,
        alcohol: float,
        ph: float,
        residual_sugar: float,
        volatile_acidity: float,
        sulphates: float,
        oak_index: float = 0.70
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes hedonic alignment score (0 - 100) and expected consumer satisfaction
        across North America, Continental Europe, East Asia, and Nordic Monopolies.
        """
        results = {}

        for region_key, profile in self.regional_profiles.items():
            ideal_alc = profile["ideal_alcohol"]
            ideal_ph = profile["ideal_ph"]
            ideal_rs = profile["ideal_residual_sugar"]
            ideal_va = profile["ideal_volatile_acidity"]
            ideal_so4 = profile["ideal_sulphates"]

            # Normalized Euclidean deviations
            d_alc = abs(alcohol - ideal_alc) / 1.5
            d_ph = abs(ph - ideal_ph) / 0.25
            d_rs = abs(residual_sugar - ideal_rs) / 2.0
            d_va = max(0.0, volatile_acidity - ideal_va) / 0.20
            d_so4 = abs(sulphates - ideal_so4) / 0.20

            total_penalty = (
                d_alc * 18.0 +
                d_ph * 22.0 * profile["acid_tolerance_penalty"] +
                d_rs * 15.0 +
                d_va * 25.0 +
                d_so4 * 12.0
            )

            # Bonus for matching oak affinity
            oak_bonus = (oak_index - 0.5) * 8.0 * profile["oak_affinity_weight"]

            raw_score = 100.0 - total_penalty + oak_bonus
            palate_score = max(20.0, min(99.0, raw_score))

            if palate_score >= 85.0:
                market_fit = "EXCEPTIONAL_MARKET_FIT"
            elif palate_score >= 70.0:
                market_fit = "STRONG_COMMERCIAL_FIT"
            elif palate_score >= 55.0:
                market_fit = "MODERATE_NICHE_APPEAL"
            else:
                market_fit = "MISALIGNED_HIGH_RESISTANCE"

            results[region_key] = {
                "region_name": profile["name"],
                "palate_alignment_score": round(palate_score, 1),
                "market_fit_verdict": market_fit,
                "price_elasticity": profile["price_elasticity"],
                "projected_retail_usd": round(profile["base_wtp_usd"] * (palate_score / 80.0), 2)
            }

        return results

    def optimize_release_pricing(
        self,
        production_cost_per_bottle_usd: float,
        market_valuation_usd: float,
        price_elasticity: float = -1.25,
        total_bottles_inventory: int = 12000
    ) -> Dict[str, Any]:
        """
        Solves profit-maximizing cellar release price given demand price elasticity.
        Profit(P) = (P - Cost) * Q(P)
        P* = Cost * (elasticity / (elasticity + 1))  if elasticity < -1
        """
        cost = production_cost_per_bottle_usd
        eps = price_elasticity

        # Theoretical Lerner rule optimum
        if eps < -1.05:
            optimal_p = cost * (eps / (eps + 1.0))
        else:
            # Inelastic / luxury pricing: markup based on perceived valuation
            optimal_p = max(market_valuation_usd * 0.95, cost * 2.8)

        # Bound optimal release price within reasonable market corridor
        optimal_p = max(cost * 1.35, min(market_valuation_usd * 1.40, optimal_p))

        # Demand projection
        p_ref = market_valuation_usd
        q_ref = total_bottles_inventory
        # Q = Q_ref * (P / P_ref)^eps
        demand_at_optimal = int(q_ref * ((optimal_p / max(p_ref, 1.0)) ** eps))
        demand_at_optimal = min(total_bottles_inventory, max(500, demand_at_optimal))

        revenue_optimal = demand_at_optimal * optimal_p
        gross_profit_optimal = demand_at_optimal * (optimal_p - cost)
        margin_pct = ((optimal_p - cost) / optimal_p) * 100.0

        return {
            "production_cost_usd": round(cost, 2),
            "market_reference_valuation_usd": round(market_valuation_usd, 2),
            "optimal_release_price_usd": round(optimal_p, 2),
            "projected_sales_bottles": demand_at_optimal,
            "projected_gross_revenue_usd": round(revenue_optimal, 2),
            "projected_gross_profit_usd": round(gross_profit_optimal, 2),
            "gross_margin_pct": round(margin_pct, 1),
            "pricing_recommendation": f"Release at ${optimal_p:.2f} for optimal risk-adjusted cellar gross margin of {margin_pct:.1f}%."
        }


# ============================================================================
# 4. Unified Digital Winery Twin & ESG Life Cycle Platform
# ============================================================================

class UnifiedDigitalWineryTwinPlatform:
    """
    Master digital twin simulating the 15-stage unified winery operation from
    harvest must intake through packaging, cold chain, and ESG carbon lifecycle.
    """

    def __init__(self):
        self.logistics_sim = ColdChainLogisticsSimulator()
        self.terroir_auditor = IsotopicTerroirAndFraudAuditor()
        self.hedonic_opt = ConsumerHedonicAndMarketPricingOptimizer()

    def simulate_end_to_end_winery_lifecycle(
        self,
        cultivar: str = "red",
        harvest_sugar_brix: float = 24.5,
        must_tartaric_acid: float = 6.8,
        must_ph: float = 3.55,
        oak_regime: str = "french_medium_plus",
        oak_duration_months: int = 12,
        stabilization_method: str = "contact_chilling_seeding",
        closure_choice: str = "technical_diam",
        shipping_route: str = "maritime_reefer_rotterdam_singapore",
        appellation_claimed: str = "bordeaux_left_bank",
        production_lot_bottles: int = 25000
    ) -> Dict[str, Any]:
        """
        Executes complete multi-stage winery twin simulation across all transformation gates.
        """
        # Stage 1: Fermentation kinetics prediction (Sugar -> Alcohol)
        potential_alcohol = harvest_sugar_brix * 0.585
        fermented_alcohol = round(min(15.2, potential_alcohol), 2)
        residual_sugar = 1.8 if fermented_alcohol < 14.8 else 3.2

        # Stage 2: Secondary Malolactic Fermentation (MLF)
        # Malic acid converts to lactic acid, raising pH by ~0.15 - 0.20
        post_mlf_ph = round(must_ph + 0.16, 2)
        post_mlf_tartaric = round(must_tartaric_acid - 0.9, 2)

        # Stage 3: Oak Maturation Phenolics
        oak_ellagitannin_mg_l = 28.5 if "french" in oak_regime else 19.0
        oak_vanillin_mg_l = 0.85 if "medium_plus" in oak_regime else 0.50

        # Stage 4: Tartrate & Protein Stabilization
        # Contact seeding reduces tartrate saturation
        final_tartaric = round(post_mlf_tartaric - 0.4, 2)
        final_potassium = 950.0  # stabilized mg/L
        bentonite_dosage_g_hl = 25.0

        # Stage 5: Crossflow Microfiltration
        filtrate_ntu = 0.42  # sterile brilliant clarity
        free_so2_target = 34.0

        # Stage 6: Quality Model Scoring proxy
        # Synthesize quality score from balance of alcohol, pH, tartaric, free SO2
        q_score = 6.85
        if post_mlf_ph <= 3.70 and fermented_alcohol >= 13.0 and final_tartaric >= 5.2:
            q_score += 0.55
        if "french" in oak_regime:
            q_score += 0.35
        q_score = min(9.2, round(q_score, 2))

        # Stage 7: Cold-Chain Shipping
        logistics_eval = self.logistics_sim.evaluate_transit_scenario(
            preset_name=shipping_route,
            closure_type=closure_choice,
            initial_free_so2=free_so2_target
        )

        # Stage 8: Terroir & Authenticity Audit
        claimed_prof = TERROIR_PROFILES.get(appellation_claimed, TERROIR_PROFILES["bordeaux_left_bank"])
        sample_minerals = {
            "lithium": claimed_prof["minerals_ppm"]["lithium"] * 1.02,
            "rubidium": claimed_prof["minerals_ppm"]["rubidium"] * 0.98,
            "strontium": claimed_prof["minerals_ppm"]["strontium"] * 1.01,
            "barium": claimed_prof["minerals_ppm"]["barium"] * 0.99,
            "manganese": claimed_prof["minerals_ppm"]["manganese"] * 1.03,
            "boron": claimed_prof["minerals_ppm"]["boron"] * 1.00,
        }
        terroir_eval = self.terroir_auditor.verify_appellation_terroir(
            claimed_appellation=appellation_claimed,
            sample_minerals=sample_minerals,
            measured_sr_ratio=claimed_prof["sr_87_86_ratio"] + 0.0001,
            measured_delta13c=-27.1,
            measured_delta18o=3.6
        )

        # Stage 9: Consumer Market Valuation & Hedonic Palate Fit
        hedonic_eval = self.hedonic_opt.estimate_bottle_market_value(
            predicted_quality=q_score,
            cultivar=cultivar,
            alcohol=fermented_alcohol,
            ph=post_mlf_ph,
            residual_sugar=residual_sugar,
            cellar_aging_years=oak_duration_months / 12.0
        )
        palate_eval = self.hedonic_opt.calculate_regional_palate_alignment(
            alcohol=fermented_alcohol,
            ph=post_mlf_ph,
            residual_sugar=residual_sugar,
            volatile_acidity=0.42,
            sulphates=0.68
        )
        pricing_opt = self.hedonic_opt.optimize_release_pricing(
            production_cost_per_bottle_usd=9.40,
            market_valuation_usd=hedonic_eval["estimated_market_price_usd"],
            total_bottles_inventory=production_lot_bottles
        )

        # Stage 10: ESG Carbon & Water Life Cycle Assessment (LCA)
        lca_eval = self.calculate_sustainability_lca(
            bottle_glass_weight_g=420.0,  # lightweight eco-glass
            is_organic_certified=True,
            fermentation_co2_captured=True,
            transit_distance_km=11000.0 if "singapore" in shipping_route else 4500.0,
            freight_mode="maritime" if "reefer" in shipping_route else "truck"
        )

        return {
            "simulation_id": f"TWIN_LOT_{cultivar.upper()}_{production_lot_bottles}",
            "cultivar": cultivar,
            "lot_size_bottles": production_lot_bottles,
            "stage_1_fermentation": {
                "brix": harvest_sugar_brix,
                "fermented_alcohol_pct": fermented_alcohol,
                "residual_sugar_g_l": residual_sugar,
            },
            "stage_2_maturation": {
                "oak_profile": oak_regime,
                "oak_months": oak_duration_months,
                "ellagitannins_ppm": oak_ellagitannin_mg_l,
                "vanillin_ppm": oak_vanillin_mg_l,
                "post_mlf_ph": post_mlf_ph,
            },
            "stage_3_stabilization": {
                "method": stabilization_method,
                "bentonite_g_hl": bentonite_dosage_g_hl,
                "final_turbidity_ntu": filtrate_ntu,
                "final_tartaric_g_l": final_tartaric,
            },
            "stage_4_quality_scoring": {
                "predicted_quality_score": q_score,
                "quality_tier": "Grand Vin (Super-Premium)" if q_score >= 7.5 else "Premier Cru (Premium)",
            },
            "stage_5_cold_chain_logistics": logistics_eval,
            "stage_6_terroir_authenticity": terroir_eval,
            "stage_7_hedonic_and_pricing": {
                "market_valuation": hedonic_eval,
                "palate_fit_by_region": palate_eval,
                "optimized_pricing": pricing_opt,
            },
            "stage_8_esg_lca_sustainability": lca_eval
        }

    def calculate_sustainability_lca(
        self,
        bottle_glass_weight_g: float = 450.0,
        is_organic_certified: bool = True,
        fermentation_co2_captured: bool = True,
        transit_distance_km: float = 6500.0,
        freight_mode: str = "maritime"
    ) -> Dict[str, Any]:
        """
        Computes Scope 1, 2, and 3 Greenhouse Gas (GHG) emissions (g CO2e per 750 mL bottle)
        and water footprint metrics according to OIV-CST 2008 and ISO 14044 standards.
        """
        # Scope 1: Direct emissions (tractors, heating, biogenic fermentation CO2)
        # 1 kg sugar generates ~ 0.49 kg CO2. 1 bottle has ~ 180 g sugar fermented -> ~88 g CO2
        biogenic_co2_g = 88.0
        captured_co2_g = 68.0 if fermentation_co2_captured else 0.0
        scope_1_g = (28.0 if is_organic_certified else 52.0) + (biogenic_co2_g - captured_co2_g)

        # Scope 2: Indirect electricity (refrigeration, crossflow filtration, bottling line)
        scope_2_g = 45.0  # g CO2e / bottle

        # Scope 3: Supply chain & packaging (Glass bottle is typically 40-50% of total wine footprint!)
        # Container glass production emits ~ 0.85 kg CO2e per kg glass
        glass_emissions_g = bottle_glass_weight_g * 0.85
        cardboard_and_closure_g = 35.0

        # Transportation emissions
        if freight_mode == "maritime":
            # Container ship: ~0.015 g CO2e / ton-km -> ~0.018 g CO2e per bottle-km
            freight_g = transit_distance_km * 0.018
        elif freight_mode == "truck":
            # Refrigerated truck: ~0.095 g CO2e per bottle-km
            freight_g = transit_distance_km * 0.095
        else: # air cargo
            # Air freight: ~0.650 g CO2e per bottle-km
            freight_g = transit_distance_km * 0.650

        scope_3_g = glass_emissions_g + cardboard_and_closure_g + freight_g
        total_carbon_g_co2e = scope_1_g + scope_2_g + scope_3_g

        # Water footprint (Liters of water / Liter of wine produced)
        # Viticulture + cellar washdown + cooling
        water_footprint_l_per_l = 2.8 if is_organic_certified else 4.5

        # Eco-efficiency rating
        if total_carbon_g_co2e <= 750.0:
            rating = "A+ (Low-Carbon Benchmark)"
        elif total_carbon_g_co2e <= 950.0:
            rating = "A (Eco-Superior)"
        elif total_carbon_g_co2e <= 1300.0:
            rating = "B (Industry Standard)"
        else:
            rating = "C (Carbon Intensive)"

        return {
            "scope_1_direct_g_co2e": round(scope_1_g, 1),
            "scope_2_electricity_g_co2e": round(scope_2_g, 1),
            "scope_3_supply_chain_g_co2e": round(scope_3_g, 1),
            "total_footprint_g_co2e_per_bottle": round(total_carbon_g_co2e, 1),
            "glass_bottle_contribution_pct": round((glass_emissions_g / total_carbon_g_co2e) * 100.0, 1),
            "freight_transport_contribution_pct": round((freight_g / total_carbon_g_co2e) * 100.0, 1),
            "water_footprint_liters_per_liter": round(water_footprint_l_per_l, 2),
            "sustainability_rating": rating,
            "carbon_neutrality_offset_cost_usd": round((total_carbon_g_co2e / 1000000.0) * 35.0, 3) # $35/ton voluntary carbon credit
        }
