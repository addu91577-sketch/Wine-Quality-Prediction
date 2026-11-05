"""
Enological Tartrate & Protein Stabilization Kinetics, Crossflow Membrane Microfiltration,
Dissolved Gas Packaging Dynamics, and Packaging Closure Shelf-Life Forecaster.

Modules & Capabilities:
- TartrateStabilizationSimulator: Thermodynamic bitartrate concentration product (CP),
  saturation temperature (T_sat), minicontact electrical conductivity drop test (Delta chi),
  chilled contact precipitation kinetics, and additive inhibitor comparison (CMC, KPA, Metatartaric).
- ProteinHazeAndBentoniteOptimizer: Pathogenesis-Related (PR) thermolabile protein haze risk,
  heat stability test simulation (80C / 2h Delta NTU), and Langmuir adsorption isotherm for
  optimal bentonite fining dosage (g/hL) balancing clarity vs volatile aroma preservation.
- CrossflowFiltrationSimulator: Hermia's filtration blocking models (complete, intermediate,
  pore constriction, cake layer), Transmembrane Pressure (TMP), flux decline J(t),
  nephelometric turbidity reduction (NTU < 0.6), and Filterability Index (FI).
- DissolvedGasPackagingEngine: Henry's Law solubility equilibria for DO and CO2 vs temperature/ethanol,
  counter-current micro-bubble sparging deoxygenation, Total Packaging Oxygen (TPO) calculation,
  headspace vacuum flushing, and antioxidant Free SO2 consumption stoichiometry.
- ClosureShelfLifeForecaster: Oxygen Transmission Rate (OTR) across closure profiles (Natural Cork,
  Diam micro-agglomerated, Synthetic Nomacorc, Screwcap Saran-Tin, Screwcap Saranex), 0-60 month
  Free SO2 depletion kinetics, browning index (A420) elevation, PremOx risk, and thermal shelf-life.
- BottlingLineQCAuditor: Statistical Process Control (SPC) evaluating fill volume legal metrology
  (MAV 750 mL +/- 1.5%), closure insertion depth, sterile membrane bubble point integrity (> 3.2 bar),
  and process capability metrics (Cp, Cpk).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

from src.feature_engineering import create_engineered_features


# ============================================================================
# Enological Constants & Reference Profiles
# ============================================================================

# Molecular Weights (g/mol)
MW_TARTRIC_ACID = 150.09
MW_POTASSIUM = 39.098
MW_POTASSIUM_BITARTRATE = 188.18   # KHT (KHC4H4O6)
MW_OXYGEN = 31.9988
MW_SO2 = 64.066

# Bitartrate Acid Dissociation Constants at 20C
PKA1_TARTARIC = 3.01
PKA2_TARTARIC = 4.37

# Universal gas constant
GAS_CONSTANT_R = 8.31446  # J/(mol*K)

# Closure Oxygen Transmission Rates (OTR in mg O2 / bottle / year & initial desorption mg)
CLOSURE_PROFILES: Dict[str, Dict[str, Any]] = {
    "natural_cork_fleur": {
        "name": "Natural Cork (Grade Fleur)",
        "steady_otr_mg_year": 0.22,       # ~0.018 mg/month
        "initial_desorption_mg": 0.45,   # Compressed cell gas release in first 3 months
        "otr_variance_pct": 28.0,         # High natural biological variance
        "tca_risk_pct": 0.3,             # Residual 2,4,6-trichloroanisole risk
        "reductive_risk": "very_low",
        "recommended_longevity_years": 15
    },
    "diam_5_microagglomerated": {
        "name": "Diam 5 (Micro-agglomerated TCA-free)",
        "steady_otr_mg_year": 0.10,       # ~0.0083 mg/month
        "initial_desorption_mg": 0.12,
        "otr_variance_pct": 5.0,          # Highly uniform industrial batch control
        "tca_risk_pct": 0.0,
        "reductive_risk": "low",
        "recommended_longevity_years": 8
    },
    "diam_10_microagglomerated": {
        "name": "Diam 10 (Micro-agglomerated Extended Aging)",
        "steady_otr_mg_year": 0.065,      # ~0.0054 mg/month
        "initial_desorption_mg": 0.09,
        "otr_variance_pct": 4.5,
        "tca_risk_pct": 0.0,
        "reductive_risk": "low",
        "recommended_longevity_years": 18
    },
    "synthetic_nomacorc_select": {
        "name": "Nomacorc Select Green (Bio-polymer)",
        "steady_otr_mg_year": 0.16,       # ~0.0133 mg/month
        "initial_desorption_mg": 0.18,
        "otr_variance_pct": 7.0,
        "tca_risk_pct": 0.0,
        "reductive_risk": "very_low",
        "recommended_longevity_years": 5
    },
    "screwcap_saran_tin": {
        "name": "Screwcap (Saran-Tin Liner - Hermetic)",
        "steady_otr_mg_year": 0.004,      # ~0.00033 mg/month (quasi-hermetic)
        "initial_desorption_mg": 0.02,
        "otr_variance_pct": 3.0,
        "tca_risk_pct": 0.0,
        "reductive_risk": "high",        # Risk of H2S / mercaptan reduction if DO too low
        "recommended_longevity_years": 20
    },
    "screwcap_saranex": {
        "name": "Screwcap (Saranex Liner - Controlled Permeability)",
        "steady_otr_mg_year": 0.045,      # ~0.00375 mg/month
        "initial_desorption_mg": 0.04,
        "otr_variance_pct": 4.0,
        "tca_risk_pct": 0.0,
        "reductive_risk": "medium",
        "recommended_longevity_years": 10
    }
}

# Additive Tartrate Stabilizers Comparison Data
STABILIZATION_ADDITIVES: Dict[str, Dict[str, Any]] = {
    "cold_stabilization_seeding": {
        "name": "Chilled Contact Stabilization (-4C + 4 g/L KHT seed)",
        "mode": "Physical precipitation",
        "durability_months": 60,
        "red_wine_suitable": True,
        "white_wine_suitable": True,
        "typical_cost_eur_hl": 1.45,
        "organoleptic_impact": "None (neutral)",
        "energy_kwh_hl": 3.8
    },
    "carboxymethylcellulose_cmc": {
        "name": "Carboxymethylcellulose (CMC / E466)",
        "mode": "Crystal nucleation & growth inhibition",
        "durability_months": 36,
        "red_wine_suitable": False,      # Precipitates coloring anthocyanins/tannins
        "white_wine_suitable": True,
        "typical_cost_eur_hl": 0.42,
        "organoleptic_impact": "Slight colloidal rounding",
        "energy_kwh_hl": 0.1
    },
    "potassium_polyaspartate_kpa": {
        "name": "Potassium Polyaspartate (KPA / E456 / Zenith)",
        "mode": "Crystal lattice binding inhibitor",
        "durability_months": 48,
        "red_wine_suitable": True,
        "white_wine_suitable": True,
        "typical_cost_eur_hl": 0.85,
        "organoleptic_impact": "Zero stripping, stable colour",
        "energy_kwh_hl": 0.1
    },
    "metatartaric_acid": {
        "name": "Metatartaric Acid (E353)",
        "mode": "Short-term crystal coating inhibitor",
        "durability_months": 9,          # Hydrolyzes rapidly at > 18C
        "red_wine_suitable": True,
        "white_wine_suitable": True,
        "typical_cost_eur_hl": 0.28,
        "organoleptic_impact": "None initially, acid release upon hydrolysis",
        "energy_kwh_hl": 0.05
    },
    "electrodialysis_membrane": {
        "name": "Electrodialysis (ED Membrane Extraction)",
        "mode": "Direct K+ and HT- ion removal",
        "durability_months": 72,
        "red_wine_suitable": True,
        "white_wine_suitable": True,
        "typical_cost_eur_hl": 1.95,
        "organoleptic_impact": "Preserves native aromas, slight acidity reduction",
        "energy_kwh_hl": 1.2
    }
}


# ============================================================================
# 1. Tartrate Stabilization Simulator
# ============================================================================

class TartrateStabilizationSimulator:
    """
    Thermodynamic and kinetic simulator of potassium bitartrate (KHT) stability.
    Models bitartrate ion equilibrium, Concentration Product (CP),
    Saturation Temperature (T_sat), minicontact electrical conductivity drop (Delta chi),
    and chilled contact precipitation kinetics with crystal seeding.
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_bitartrate_fraction(ph: float) -> float:
        """
        Calculates the molar fraction of total tartaric acid present as the bitartrate mono-anion (HT-).
        [HT-] / [H2T]_total = 1 / (10^(pKa1 - pH) + 1 + 10^(pH - pKa2))
        """
        denom = (
            10.0 ** (PKA1_TARTARIC - ph)
            + 1.0
            + 10.0 ** (ph - PKA2_TARTARIC)
        )
        return 1.0 / denom

    @staticmethod
    def calculate_solubility_product(temperature_c: float, alcohol_pct: float) -> float:
        """
        Calculates the thermodynamic solubility product K_sp of KHT ((mol/L)^2)
        as a function of absolute temperature (Kelvin) and alcohol percentage.
        Based on Berg & Keefer enological solubility equations.
        """
        temp_k = temperature_c + 273.15
        # Apparent K_sp in wine matrix accounting for ionic strength activity coefficients
        # Drops as temperature drops and as ethanol increases (lower dielectric constant)
        ln_ksp = (
            -1850.0 / temp_k
            - 0.065 * alcohol_pct
            - 1.35
        )
        return float(math.exp(ln_ksp))

    def evaluate_tartrate_stability(
        self,
        tartaric_acid_g_l: float,
        potassium_mg_l: float,
        ph: float,
        alcohol_pct: float,
        storage_temp_c: float = 14.0
    ) -> Dict[str, Any]:
        """
        Evaluates potassium bitartrate stability parameters:
        - Molar potassium [K+]
        - Molar bitartrate [HT-]
        - Concentration Product (CP)
        - Solubility Product (K_sp) at storage temperature
        - Saturation Temperature (T_sat)
        - Minicontact conductivity drop estimation (Delta chi in uS/cm)
        - Stability classification
        """
        # Molar potassium: mg/L to mol/L
        k_molar = (potassium_mg_l / 1000.0) / MW_POTASSIUM

        # Total tartaric acid in mol/L
        tartaric_total_molar = tartaric_acid_g_l / MW_TARTRIC_ACID

        # Bitartrate fraction
        f_ht = self.calculate_bitartrate_fraction(ph)
        ht_molar = tartaric_total_molar * f_ht

        # Concentration Product: CP = [K+] * [HT-] in (mol/L)^2
        cp = k_molar * ht_molar

        # K_sp at storage temperature
        ksp_storage = self.calculate_solubility_product(storage_temp_c, alcohol_pct)

        # Saturation index: SI = CP / K_sp
        saturation_ratio = cp / max(ksp_storage, 1e-9)

        # Solve for Saturation Temperature (T_sat where CP = K_sp(T_sat))
        # ln(CP) = -1850 / T_sat_k - 0.065 * alcohol - 1.35
        # T_sat_k = -1850 / (ln(CP) + 0.065 * alcohol + 1.35)
        try:
            target_ln = math.log(max(cp, 1e-9)) + 0.065 * alcohol_pct + 1.35
            if target_ln < 0:
                t_sat_k = -1850.0 / target_ln
                t_sat_c = t_sat_k - 273.15
            else:
                t_sat_c = 45.0
        except Exception:
            t_sat_c = 25.0

        t_sat_c = float(np.clip(t_sat_c, -10.0, 45.0))

        # Mini-contact conductivity drop estimation (Delta chi in uS/cm at 0C contact test)
        # Proportional to supersaturation at 0C
        ksp_0c = self.calculate_solubility_product(0.0, alcohol_pct)
        delta_molar_excess = max(0.0, cp - ksp_0c)
        delta_chi_us_cm = float(delta_molar_excess * 2.5e5)

        # Stability classification:
        # Standard enological threshold: Delta chi < 30 uS/cm is commercially stable
        if delta_chi_us_cm < 30.0:
            stability_status = "STABLE"
            haze_crystal_risk = "LOW"
            action_required = False
        elif delta_chi_us_cm <= 55.0:
            stability_status = "MODERATE_RISK"
            haze_crystal_risk = "MEDIUM"
            action_required = True
        else:
            stability_status = "UNSTABLE"
            haze_crystal_risk = "HIGH"
            action_required = True

        return {
            "potassium_molar": float(k_molar),
            "bitartrate_molar": float(ht_molar),
            "bitartrate_fraction": float(f_ht),
            "concentration_product_cp": float(cp),
            "solubility_product_ksp": float(ksp_storage),
            "saturation_ratio": float(saturation_ratio),
            "saturation_temperature_c": float(round(t_sat_c, 2)),
            "storage_temp_c": float(storage_temp_c),
            "minicontact_delta_chi_us_cm": float(round(delta_chi_us_cm, 1)),
            "stability_status": stability_status,
            "haze_crystal_risk": haze_crystal_risk,
            "action_required": action_required
        }

    def simulate_cold_stabilization_kinetics(
        self,
        tartaric_acid_g_l: float,
        potassium_mg_l: float,
        ph: float,
        alcohol_pct: float,
        chilling_temp_c: float = -4.0,
        seed_crystal_g_l: float = 4.0,
        duration_hours: int = 12,
        time_step_hours: float = 0.25
    ) -> Dict[str, Any]:
        """
        Simulates the dynamic contact crystallization of potassium bitartrate over time.
        Rate: d[KHT]/dt = -k_c * (1 + beta * seed) * (CP(t) - K_sp)^alpha
        """
        # Equilibrium K_sp at chilling temp
        ksp_eq = self.calculate_solubility_product(chilling_temp_c, alcohol_pct)

        f_ht = self.calculate_bitartrate_fraction(ph)
        current_k_mg_l = potassium_mg_l
        current_tartaric_g_l = tartaric_acid_g_l

        # Kinetic rate constant: higher seeding increases crystal growth rate
        k_rate = 0.65 * (1.0 + 0.45 * seed_crystal_g_l)

        times = []
        k_trajectory = []
        tartaric_trajectory = []
        cp_trajectory = []
        kht_precipitated_g_l = []

        total_precip_g_l = 0.0

        n_steps = int(duration_hours / time_step_hours) + 1

        for step in range(n_steps):
            t = step * time_step_hours

            k_molar = (current_k_mg_l / 1000.0) / MW_POTASSIUM
            t_molar = (current_tartaric_g_l / MW_TARTRIC_ACID) * f_ht
            cp = k_molar * t_molar

            times.append(round(t, 2))
            k_trajectory.append(round(current_k_mg_l, 1))
            tartaric_trajectory.append(round(current_tartaric_g_l, 2))
            cp_trajectory.append(float(cp))
            kht_precipitated_g_l.append(round(total_precip_g_l, 3))

            if cp > ksp_eq:
                # Driving force: supersaturation (CP - K_sp)
                driving_force = cp - ksp_eq
                # Moles of KHT precipitated per liter per hour
                rate_moles_l_h = k_rate * (driving_force ** 0.85)
                moles_step = rate_moles_l_h * time_step_hours

                # Convert moles KHT to mg K+ and g Tartaric
                delta_k_mg = moles_step * MW_POTASSIUM * 1000.0
                delta_tartaric_g = moles_step * MW_TARTRIC_ACID
                delta_kht_g = moles_step * MW_POTASSIUM_BITARTRATE

                current_k_mg_l = max(50.0, current_k_mg_l - delta_k_mg)
                current_tartaric_g_l = max(1.0, current_tartaric_g_l - delta_tartaric_g)
                total_precip_g_l += delta_kht_g

        # Final stability check
        final_eval = self.evaluate_tartrate_stability(
            tartaric_acid_g_l=current_tartaric_g_l,
            potassium_mg_l=current_k_mg_l,
            ph=ph,
            alcohol_pct=alcohol_pct,
            storage_temp_c=14.0
        )

        return {
            "parameters": {
                "chilling_temp_c": chilling_temp_c,
                "seed_crystal_g_l": seed_crystal_g_l,
                "duration_hours": duration_hours
            },
            "initial_state": {
                "tartaric_acid_g_l": tartaric_acid_g_l,
                "potassium_mg_l": potassium_mg_l,
                "ph": ph
            },
            "final_state": {
                "tartaric_acid_g_l": current_tartaric_g_l,
                "potassium_mg_l": current_k_mg_l,
                "total_kht_precipitated_g_l": round(total_precip_g_l, 3),
                "saturation_temp_c": final_eval["saturation_temperature_c"],
                "minicontact_delta_chi_us_cm": final_eval["minicontact_delta_chi_us_cm"],
                "stability_status": final_eval["stability_status"]
            },
            "trajectory": {
                "hour": times,
                "potassium_mg_l": k_trajectory,
                "tartaric_acid_g_l": tartaric_trajectory,
                "kht_precipitated_g_l": kht_precipitated_g_l
            }
        }


# ============================================================================
# 2. Protein Haze & Bentonite Fining Optimizer
# ============================================================================

class ProteinHazeAndBentoniteOptimizer:
    """
    Evaluates Pathogenesis-Related (PR) thermolabile protein instability (thaumatin-like, chitinases).
    Simulates heat-stability test (80C / 2h Delta NTU) and computes Langmuir adsorption isotherm
    to determine optimal bentonite fining dosage (g/hL) balancing clarity vs aroma stripping.
    """

    def __init__(
        self,
        q_max_adsorption_mg_g: float = 85.0,  # Max protein adsorbed per g bentonite
        langmuir_kl: float = 0.045,            # Affinity constant (L/mg)
        target_max_delta_ntu: float = 1.8     # Commercial heat-stability threshold
    ):
        self.q_max = q_max_adsorption_mg_g
        self.kl = langmuir_kl
        self.target_delta_ntu = target_max_delta_ntu

    def evaluate_protein_haze_risk(
        self,
        protein_content_mg_l: float,
        ph: float,
        phenolics_total_mg_l: float = 250.0,
        cultivar: str = "white"
    ) -> Dict[str, Any]:
        """
        Simulates the heat test: 80C for 2 hours, cooling to 20C, and measuring Delta NTU.
        Protein haze is amplified at lower pH (further below isoelectric point pI ~ 4.5-5.5)
        and modulated by phenolic protective condensation.
        """
        # Electrostatic charge factor: PR proteins have net positive charge at wine pH (3.0-3.6)
        charge_factor = 1.0 + max(0.0, (3.8 - ph) * 0.8)

        # Phenolic tannin-protein interaction factor (reds have tannins that naturally precipitate proteins)
        if cultivar.lower() == "red":
            phenolic_masking = max(0.1, 1.0 - (phenolics_total_mg_l / 1200.0))
        else:
            phenolic_masking = 1.0

        # Estimated heat-induced Delta NTU
        # Baseline: ~ 10 mg/L unstable PR protein yields ~ 1.5 - 2.5 NTU haze
        delta_ntu = (protein_content_mg_l * 0.16 * charge_factor * phenolic_masking)
        delta_ntu = float(round(delta_ntu, 2))

        is_stable = delta_ntu <= self.target_delta_ntu
        if is_stable:
            risk_category = "STABLE"
        elif delta_ntu <= 6.0:
            risk_category = "MODERATE_HAZE_RISK"
        else:
            risk_category = "SEVERE_HAZE_RISK"

        return {
            "protein_content_mg_l": float(protein_content_mg_l),
            "ph": float(ph),
            "estimated_heat_delta_ntu": delta_ntu,
            "target_threshold_ntu": self.target_delta_ntu,
            "is_stable": is_stable,
            "risk_category": risk_category
        }

    def optimize_bentonite_dosage(
        self,
        protein_content_mg_l: float,
        ph: float,
        cultivar: str = "white",
        bentonite_type: str = "sodium_bentonite"
    ) -> Dict[str, Any]:
        """
        Calculates optimal bentonite dosage (g/hL) using Langmuir adsorption isotherm.
        Balances haze prevention against volatile terpene/ester stripping penalty.
        """
        # Type modifier: Sodium bentonite has higher swelling & adsorption capacity than Calcium bentonite
        capacity_mult = 1.15 if bentonite_type == "sodium_bentonite" else 0.82
        q_eff = self.q_max * capacity_mult

        haze_eval = self.evaluate_protein_haze_risk(protein_content_mg_l, ph, cultivar=cultivar)

        if haze_eval["is_stable"]:
            return {
                "protein_initial_mg_l": float(protein_content_mg_l),
                "heat_delta_ntu_initial": haze_eval["estimated_heat_delta_ntu"],
                "recommended_bentonite_g_hl": 0.0,
                "residual_protein_mg_l": float(protein_content_mg_l),
                "heat_delta_ntu_final": haze_eval["estimated_heat_delta_ntu"],
                "aroma_stripping_penalty_pct": 0.0,
                "status": "ALREADY_STABLE"
            }

        # Target residual protein that corresponds to target_delta_ntu
        # delta_ntu = residual_protein * 0.16 * charge_factor * phenolic_masking
        ratio = haze_eval["estimated_heat_delta_ntu"] / max(protein_content_mg_l, 1e-3)
        target_residual_protein = self.target_delta_ntu / max(ratio, 1e-4)

        # Protein to remove (mg/L)
        protein_to_remove = max(0.0, protein_content_mg_l - target_residual_protein)

        # Langmuir equilibrium: q = q_eff * Kl * C_eq / (1 + Kl * C_eq)
        # where C_eq = target_residual_protein
        c_eq = target_residual_protein
        q_adsorbed_mg_g = (q_eff * self.kl * c_eq) / (1.0 + self.kl * c_eq)
        q_adsorbed_mg_g = max(q_adsorbed_mg_g, 5.0)

        # Bentonite required in g/L: protein_to_remove (mg/L) / q_adsorbed (mg/g) / 1000? No:
        # (mg protein / L) / (mg protein / g bentonite) = g bentonite / L
        bentonite_g_l = protein_to_remove / q_adsorbed_mg_g
        # Convert g/L to g/hL (1 hL = 100 L) -> multiply by 100
        bentonite_g_hl = bentonite_g_l * 100.0

        # Safety margin of 10%
        bentonite_g_hl = round(float(bentonite_g_hl * 1.10), 1)
        bentonite_g_hl = float(np.clip(bentonite_g_hl, 10.0, 180.0))

        # Aroma stripping estimation: higher bentonite strips monoterpenes and fatty acid esters
        # Stripping penalty (%) ~ 100 * (1 - exp(-0.008 * dosage_g_hl))
        aroma_stripping_pct = round(float(100.0 * (1.0 - math.exp(-0.0085 * bentonite_g_hl))), 1)

        # Final post-fining haze estimate
        residual_p = target_residual_protein
        final_delta_ntu = round(float(self.target_delta_ntu * 0.92), 2)

        return {
            "protein_initial_mg_l": float(protein_content_mg_l),
            "heat_delta_ntu_initial": haze_eval["estimated_heat_delta_ntu"],
            "bentonite_type": bentonite_type,
            "recommended_bentonite_g_hl": bentonite_g_hl,
            "residual_protein_mg_l": float(round(residual_p, 1)),
            "heat_delta_ntu_final": final_delta_ntu,
            "aroma_stripping_penalty_pct": aroma_stripping_pct,
            "status": "OPTIMIZATION_SUCCESSFUL"
        }


# ============================================================================
# 3. Crossflow Membrane Microfiltration Simulator
# ============================================================================

class CrossflowFiltrationSimulator:
    """
    Simulates crossflow hollow fiber membrane microfiltration (0.45 um polyethersulfone PES).
    Implements Hermia's blocking models (pore constriction, intermediate, cake layer),
    transmembrane pressure (TMP), permeate flux decay J(t), nephelometric turbidity
    clarification (NTU < 0.6), and Filterability Index (FI).
    """

    def __init__(
        self,
        clean_membrane_resistance: float = 1.2e11,  # R_m (1/m)
        dynamic_viscosity_pa_s: float = 1.45e-3,   # Wine viscosity at 18C (Pa*s)
        membrane_surface_area_m2: float = 40.0     # Commercial skid area
    ):
        self.rm = clean_membrane_resistance
        self.viscosity = dynamic_viscosity_pa_s
        self.area_m2 = membrane_surface_area_m2

    def simulate_crossflow_run(
        self,
        initial_turbidity_ntu: float = 35.0,
        transmembrane_pressure_bar: float = 1.6,
        target_volume_liters: float = 10000.0,
        colloidal_index: float = 1.2,              # Fouling load factor
        backwash_interval_hours: float = 2.0,
        run_duration_hours: float = 6.0,
        time_step_hours: float = 0.1
    ) -> Dict[str, Any]:
        """
        Simulates flux decline J(t) (L/(m^2*h)) and cumulative permeate volume.
        Incorporates periodic backwash hydrodynamic cleaning to restore flux.
        """
        tmp_pa = transmembrane_pressure_bar * 1e5
        initial_flux = (tmp_pa / (self.viscosity * self.rm)) * 3600.0 * 1000.0  # L/(m^2*h)
        # Scale to typical enological flux: 80 - 150 L/(m^2*h)
        flux_scale = 120.0 / max(initial_flux, 1e-3)

        current_rcake = 0.0
        current_rpore = 0.0

        times = []
        fluxes = []
        cumulative_volumes = []
        ntu_permeate = []
        tmps = []

        total_volume = 0.0
        n_steps = int(run_duration_hours / time_step_hours) + 1

        for step in range(n_steps):
            t = step * time_step_hours

            # Periodic backwash reset
            if t > 0 and (t % backwash_interval_hours) < time_step_hours:
                # Backwash removes ~75% of reversible surface cake resistance
                current_rcake *= 0.25

            # Total resistance
            r_total = self.rm + current_rpore + current_rcake
            # Instantaneous flux in L/(m^2 * h)
            j_t = (tmp_pa / (self.viscosity * r_total)) * 3600.0 * 1000.0 * flux_scale
            j_t = max(25.0, j_t)

            # Cake and pore fouling accumulation
            d_rcake = 3.5e10 * colloidal_index * (j_t / 100.0) * time_step_hours
            d_rpore = 8.0e9 * colloidal_index * (j_t / 100.0) * time_step_hours

            current_rcake += d_rcake
            current_rpore += d_rpore

            # Volume produced in this step
            step_vol = j_t * self.area_m2 * time_step_hours
            total_volume += step_vol

            # Permeate turbidity: crossflow membrane retains 99%+ of particulate colloids
            permeate_ntu = max(0.25, 0.35 + 0.005 * initial_turbidity_ntu)

            times.append(round(t, 2))
            fluxes.append(round(j_t, 1))
            cumulative_volumes.append(round(total_volume, 1))
            ntu_permeate.append(round(permeate_ntu, 2))
            tmps.append(round(transmembrane_pressure_bar, 2))

            if total_volume >= target_volume_liters:
                break

        # Filterability Index calculation (FI = t400 - 2*t200 in seconds)
        # Empirically predicted from colloidal index and initial turbidity
        filterability_index_s = round(float(8.0 + 12.0 * colloidal_index + 0.15 * initial_turbidity_ntu), 1)

        clarity_status = "BRILLIANT_STERILE" if ntu_permeate[-1] < 0.6 else "COMMERCIALLY_CLEAR"

        return {
            "parameters": {
                "transmembrane_pressure_bar": transmembrane_pressure_bar,
                "membrane_area_m2": self.area_m2,
                "initial_turbidity_ntu": initial_turbidity_ntu,
                "target_volume_liters": target_volume_liters
            },
            "summary": {
                "total_volume_filtered_l": round(total_volume, 1),
                "duration_hours": times[-1],
                "average_flux_lmh": round(float(np.mean(fluxes)), 1),
                "final_flux_lmh": fluxes[-1],
                "final_permeate_turbidity_ntu": ntu_permeate[-1],
                "clarity_status": clarity_status,
                "filterability_index_s": filterability_index_s,
                "clogging_risk": "LOW" if filterability_index_s < 25.0 else ("MEDIUM" if filterability_index_s < 45.0 else "HIGH")
            },
            "trajectory": {
                "hour": times,
                "flux_lmh": fluxes,
                "cumulative_volume_l": cumulative_volumes,
                "permeate_ntu": ntu_permeate
            }
        }


# ============================================================================
# 4. Dissolved Gas Packaging & Sparging Simulator
# ============================================================================

class DissolvedGasPackagingEngine:
    """
    Simulates dissolved oxygen (DO) and carbon dioxide (CO2) equilibria in wine packaging.
    Models Henry's Law solubility vs temperature and ethanol %, counter-current micro-bubble
    nitrogen sparging deoxygenation, Total Packaging Oxygen (TPO), and antioxidant Free SO2
    consumption stoichiometry (1.0 mg/L O2 consumes ~4.0 mg/L Free SO2).
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_henry_oxygen_solubility(temperature_c: float, alcohol_pct: float) -> float:
        """
        Calculates saturated dissolved oxygen (DO) solubility in mg/L in equilibrium
        with atmospheric air (20.95% O2) at 1 atm.
        Solubility decreases with increasing temperature and slightly decreases with ethanol.
        """
        # Water air-saturation at 1 atm: ~ 9.09 mg/L at 20C, 11.28 mg/L at 10C
        temp_k = temperature_c + 273.15
        base_solubility = 14.16 - 0.394 * temperature_c + 0.0078 * (temperature_c ** 2) - 0.000064 * (temperature_c ** 3)
        # Ethanol matrix correction
        ethanol_factor = 1.0 - 0.008 * (alcohol_pct - 12.0)
        return float(max(1.0, round(base_solubility * ethanol_factor, 2)))

    @staticmethod
    def calculate_henry_co2_solubility(temperature_c: float, pressure_bar: float = 1.0) -> float:
        """
        Calculates equilibrium dissolved CO2 solubility in g/L at given temperature and partial pressure.
        Henry's constant for CO2 in wine: K_H ~ 1.5 - 2.0 g/(L*bar) at 20C.
        """
        kh_20c = 1.72  # g/(L*bar)
        # van 't Hoff temperature dependency
        delta_h_r = 2400.0  # K
        temp_k = temperature_c + 273.15
        kh_t = kh_20c * math.exp(delta_h_r * (1.0 / temp_k - 1.0 / 293.15))
        return float(round(kh_t * pressure_bar, 2))

    def simulate_nitrogen_sparging(
        self,
        initial_do_mg_l: float,
        target_do_mg_l: float = 0.40,
        nitrogen_flow_ratio_nl_l: float = 0.5,     # Norm-liters N2 per liter of wine
        temperature_c: float = 18.0,
        contactor_efficiency: float = 0.88
    ) -> Dict[str, Any]:
        """
        Simulates inline counter-current porous sintered sparger deoxygenation.
        Stripping rate: ln(DO_final / DO_initial) = - Kla_eff * (V_N2 / V_wine)
        """
        kla_eff = 3.2 * contactor_efficiency

        # Final DO achieved
        ratio = math.exp(-kla_eff * nitrogen_flow_ratio_nl_l)
        final_do = max(0.12, initial_do_mg_l * ratio)

        oxygen_removed_mg_l = initial_do_mg_l - final_do
        target_reached = final_do <= target_do_mg_l

        return {
            "initial_do_mg_l": float(initial_do_mg_l),
            "final_do_mg_l": float(round(final_do, 2)),
            "target_do_mg_l": float(target_do_mg_l),
            "nitrogen_ratio_nl_l": float(nitrogen_flow_ratio_nl_l),
            "oxygen_removed_mg_l": float(round(oxygen_removed_mg_l, 2)),
            "deoxygenation_efficiency_pct": float(round((oxygen_removed_mg_l / initial_do_mg_l) * 100.0, 1)),
            "target_reached": target_reached
        }

    def calculate_total_packaging_oxygen(
        self,
        dissolved_oxygen_wine_mg_l: float,
        headspace_volume_ml: float = 6.0,
        bottle_volume_ml: float = 750.0,
        headspace_o2_pct: float = 1.5,       # 1.5% with inert gas flushing, 20.9% without flushing
        wine_temperature_c: float = 18.0
    ) -> Dict[str, Any]:
        """
        Calculates Total Packaging Oxygen (TPO):
        TPO = DO_wine + Headspace_Oxygen (mg/L of wine)
        Headspace Oxygen (mg) = V_hs (L) * (O2_pct / 100) * (MW_O2 / V_molar) * 1000
        """
        # Molar volume of gas at 1 atm and wine temperature
        temp_k = wine_temperature_c + 273.15
        v_molar = (0.08206 * temp_k)  # L/mol

        # Mass of O2 in headspace (mg)
        v_hs_liters = headspace_volume_ml / 1000.0
        o2_volume_l = v_hs_liters * (headspace_o2_pct / 100.0)
        moles_o2 = o2_volume_l / v_molar
        headspace_o2_mg = moles_o2 * MW_OXYGEN * 1000.0

        # Distributed over bottle volume (mg O2 / L wine)
        headspace_o2_mg_l_wine = headspace_o2_mg / (bottle_volume_ml / 1000.0)

        tpo_mg_l = dissolved_oxygen_wine_mg_l + headspace_o2_mg_l_wine

        # Antioxidant consumption: 1 mg O2 consumes ~ 4.0 mg Free SO2
        free_so2_depletion_mg_l = round(tpo_mg_l * 4.0, 2)

        # Quality tier threshold evaluation
        if tpo_mg_l < 0.8:
            tpo_grade = "EXCELLENT_PREMIUM"
        elif tpo_mg_l < 1.5:
            tpo_grade = "GOOD_COMMERCIAL"
        elif tpo_mg_l < 2.5:
            tpo_grade = "ELEVATED_RISK"
        else:
            tpo_grade = "SEVERE_OXIDATION_HAZARD"

        return {
            "dissolved_oxygen_mg_l": float(round(dissolved_oxygen_wine_mg_l, 2)),
            "headspace_oxygen_mg_l_wine": float(round(headspace_o2_mg_l_wine, 2)),
            "headspace_o2_pct": float(headspace_o2_pct),
            "total_packaging_oxygen_tpo_mg_l": float(round(tpo_mg_l, 2)),
            "stoichiometric_free_so2_depletion_mg_l": free_so2_depletion_mg_l,
            "tpo_grade": tpo_grade,
            "is_within_premium_spec": tpo_mg_l < 1.0
        }


# ============================================================================
# 5. Packaging Closure Shelf-Life Forecaster
# ============================================================================

class ClosureShelfLifeForecaster:
    """
    Predicts multi-year bottle shelf-life and oxidation kinetics across closure types.
    Models Oxygen Transmission Rate (OTR), initial gas desorption, Free SO2 depletion,
    spectrophotometric browning elevation (A420), and PremOx (premature oxidation) boundaries.
    """

    def __init__(self):
        pass

    def simulate_bottle_shelf_life(
        self,
        closure_type: str,
        initial_free_so2: float,
        initial_ph: float,
        initial_tpo_mg_l: float,
        storage_temperature_c: float = 14.0,
        initial_a420_browning: float = 0.12,
        duration_months: int = 48,
        bottle_volume_l: float = 0.75
    ) -> Dict[str, Any]:
        """
        Simulates 0-48+ months temporal bottle aging:
        - Free SO2 decay from TPO consumption and steady OTR ingress
        - Browning index (A420) elevation
        - Sensory freshness score (0-100)
        - PremOx threshold identification (Free SO2 < 10 mg/L and Molecular SO2 < 0.4 mg/L)
        """
        profile = CLOSURE_PROFILES.get(closure_type, CLOSURE_PROFILES["natural_cork_fleur"])

        monthly_steady_otr_mg = profile["steady_otr_mg_year"] / 12.0
        initial_desorption_mg = profile["initial_desorption_mg"]

        # Molecular SO2 calculation factor
        # Molecular SO2 = Free SO2 / (1 + 10^(pH - 1.81))
        pka_so2 = 1.81

        # Temperature acceleration factor (Arrhenius with Ea ~ 70 kJ/mol)
        temp_k = storage_temperature_c + 273.15
        ref_temp_k = 287.15  # 14C
        arrhenius_mult = math.exp((70000.0 / GAS_CONSTANT_R) * (1.0 / ref_temp_k - 1.0 / temp_k))

        # Initial Free SO2 hit from TPO: each 1 mg TPO burns ~ 4 mg Free SO2
        tpo_hit = initial_tpo_mg_l * 4.0
        current_free_so2 = max(2.0, initial_free_so2 - tpo_hit)
        current_a420 = initial_a420_browning
        current_freshness = 95.0

        months = []
        free_so2_traj = []
        molecular_so2_traj = []
        a420_traj = []
        freshness_traj = []

        premox_month = None
        peak_maturity_month = None
        end_of_life_month = None

        for m in range(duration_months + 1):
            # Compute molecular SO2
            denom = 1.0 + (10.0 ** (initial_ph - pka_so2))
            mol_so2 = current_free_so2 / denom

            months.append(m)
            free_so2_traj.append(round(current_free_so2, 2))
            molecular_so2_traj.append(round(mol_so2, 3))
            a420_traj.append(round(current_a420, 3))
            freshness_traj.append(round(current_freshness, 1))

            # Milestone checks
            if premox_month is None and (current_free_so2 < 10.0 or mol_so2 < 0.35):
                premox_month = m

            if peak_maturity_month is None and m >= 12 and current_freshness <= 82.0:
                peak_maturity_month = m

            if end_of_life_month is None and (current_free_so2 < 4.0 or current_freshness < 45.0 or current_a420 > 0.45):
                end_of_life_month = m

            # Dynamic updates for next month
            # OTR for this month: first 3 months include desorption fraction
            desorption_fraction = (initial_desorption_mg / 3.0) if m < 3 else 0.0
            month_otr_mg = monthly_steady_otr_mg + desorption_fraction
            month_otr_mg_l = month_otr_mg / bottle_volume_l

            # Free SO2 consumption by OTR (4 mg SO2 per mg O2) + slow native dissociation
            so2_consumed = (month_otr_mg_l * 4.0 + 0.15) * arrhenius_mult
            current_free_so2 = max(0.5, current_free_so2 - so2_consumed)

            # Browning rate: inversely related to protective Free SO2
            browning_rate = 0.0035 * arrhenius_mult * (1.0 / (1.0 + current_free_so2 / 10.0))
            current_a420 += browning_rate

            # Freshness score decay
            freshness_decay = (0.75 + (0.5 if current_free_so2 < 10.0 else 0.0)) * arrhenius_mult
            current_freshness = max(10.0, current_freshness - freshness_decay)

        return {
            "closure_info": profile,
            "parameters": {
                "initial_free_so2": initial_free_so2,
                "initial_ph": initial_ph,
                "initial_tpo_mg_l": initial_tpo_mg_l,
                "storage_temperature_c": storage_temperature_c,
                "duration_months": duration_months
            },
            "milestones": {
                "premox_breach_month": premox_month if premox_month is not None else -1,
                "peak_maturity_month": peak_maturity_month if peak_maturity_month is not None else 18,
                "commercial_shelf_life_months": end_of_life_month if end_of_life_month is not None else duration_months
            },
            "final_state": {
                "free_so2_mg_l": free_so2_traj[-1],
                "molecular_so2_mg_l": molecular_so2_traj[-1],
                "a420_browning": a420_traj[-1],
                "freshness_score": freshness_traj[-1]
            },
            "trajectory": {
                "month": months,
                "free_so2_mg_l": free_so2_traj,
                "molecular_so2_mg_l": molecular_so2_traj,
                "a420_browning": a420_traj,
                "freshness_score": freshness_traj
            }
        }


# ============================================================================
# 6. Bottling Line QA/QC & Statistical Process Control (SPC) Auditor
# ============================================================================

class BottlingLineQCAuditor:
    """
    Performs real-time quality assurance and Statistical Process Control (SPC)
    for high-speed commercial bottling lines.
    Audits net fill volume (750 mL +/- 1.5% MAV legal metrology), closure insertion depth,
    headspace vacuum, dissolved O2, and sterile membrane bubble point integrity (> 3.2 bar).
    Computes Process Capability indices (Cp, Cpk).
    """

    def __init__(self):
        pass

    def audit_bottling_batch(
        self,
        lot_id: str,
        fill_volumes_ml: List[float],
        closure_depths_mm: List[float],
        headspace_vacuums_bar: List[float],
        filter_bubble_point_bar: float = 3.4
    ) -> Dict[str, Any]:
        """
        Audits a commercial bottling run batch against international packaging specifications.
        """
        n_samples = len(fill_volumes_ml)
        if n_samples < 5:
            raise ValueError("SPC audit requires at least 5 batch inspection samples.")

        # 1. Fill Volume Metrology: Target 750.0 mL, Legal MAV = 750 +/- 11.25 mL (1.5%)
        lsl_vol = 738.75  # Lower Specification Limit
        usl_vol = 761.25  # Upper Specification Limit
        nominal_vol = 750.0

        mean_vol = float(np.mean(fill_volumes_ml))
        std_vol = float(np.std(fill_volumes_ml, ddof=1)) if n_samples > 1 else 0.5
        std_vol = max(std_vol, 0.05)

        # Process Capability (Cp and Cpk)
        cp_vol = (usl_vol - lsl_vol) / (6.0 * std_vol)
        cpk_vol = min((usl_vol - mean_vol) / (3.0 * std_vol), (mean_vol - lsl_vol) / (3.0 * std_vol))

        vol_defects = sum(1 for v in fill_volumes_ml if v < lsl_vol or v > usl_vol)

        # 2. Closure Insertion Depth: Nominal 1.0 mm below rim, LSL = 0.5 mm, USL = 1.8 mm
        mean_depth = float(np.mean(closure_depths_mm))
        depth_defects = sum(1 for d in closure_depths_mm if d < 0.5 or d > 2.0)

        # 3. Headspace Vacuum: Target -0.2 to -0.4 bar, LSL = -0.5 bar, USL = -0.05 bar
        mean_vac = float(np.mean(headspace_vacuums_bar))
        vac_defects = sum(1 for vac in headspace_vacuums_bar if vac > -0.05)

        # 4. Filter Membrane Integrity Bubble Point Test
        # Commercial 0.45 um polyethersulfone membrane requires bubble point >= 3.2 bar
        bubble_point_pass = filter_bubble_point_bar >= 3.20

        # Overall Batch Compliance
        is_batch_approved = (
            vol_defects == 0
            and depth_defects == 0
            and vac_defects == 0
            and bubble_point_pass
            and cpk_vol >= 1.0
        )

        audit_tier = (
            "SIX_SIGMA_COMPLIANT" if (is_batch_approved and cpk_vol >= 1.33)
            else ("CONDITIONALLY_APPROVED" if is_batch_approved else "HOLD_CONTAINMENT_REQUIRED")
        )

        return {
            "lot_id": lot_id,
            "sample_count": n_samples,
            "audit_tier": audit_tier,
            "is_batch_approved": is_batch_approved,
            "fill_volume_spc": {
                "nominal_ml": nominal_vol,
                "mean_ml": round(mean_vol, 2),
                "std_ml": round(std_vol, 2),
                "lsl_ml": lsl_vol,
                "usl_ml": usl_vol,
                "cp": round(float(cp_vol), 2),
                "cpk": round(float(cpk_vol), 2),
                "defect_count": vol_defects,
                "defect_rate_pct": round((vol_defects / n_samples) * 100.0, 2)
            },
            "closure_insertion_spc": {
                "mean_depth_mm": round(mean_depth, 2),
                "defect_count": depth_defects
            },
            "headspace_vacuum_spc": {
                "mean_vacuum_bar": round(mean_vac, 3),
                "defect_count": vac_defects
            },
            "membrane_integrity": {
                "bubble_point_bar": round(filter_bubble_point_bar, 2),
                "specification_min_bar": 3.20,
                "passed": bubble_point_pass
            }
        }
