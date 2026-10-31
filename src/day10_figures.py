"""
Day 10 Publication Figures Generator.
Creates publication-quality figures for:
1. day10_coupage_optimization_surface.png: Enological coupage (wine blending) trajectory,
   non-linear pH equilibrium, and constraint satisfaction envelope.
2. day10_conformal_prediction_coverage.png: Split conformal coverage guarantee,
   prediction set efficiency, and sommelier arbitration zone.
3. day10_winery_economic_threshold_utility.png: Commercial payoff matrix, Expected Monetary
   Value (EMV) curve, and revenue uplift across decision thresholds.
4. day10_stacking_ensemble_calibration.png: Multi-model stacking meta-learner probability
   calibration (reliability diagram), Brier score, and ROC-AUC benchmarking.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import joblib

from src.data_loader import load_wine_data
from src.feature_engineering import create_binned_targets, create_engineered_features, get_stratified_train_test_split
from src.assemblage_and_conformal import (
    calculate_blended_chemistry,
    WineAssemblageOptimizer,
    EnologicalConformalPredictor,
    WineryEconomicEngine,
    WineStackingEnsemble,
    ALL_FEATURE_NAMES,
    DEFAULT_SCREENING_THRESHOLD
)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_coupage_optimization_surface(
    save_path: Path = FIGURES_DIR / "day10_coupage_optimization_surface.png"
) -> None:
    """
    Plots the non-linear enological coupage trajectory, volatile acidity boundary,
    and multi-tank chemical balance before and after assemblage.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # Define two contrasting commercial harvest lots
    # Lot A: Sub-optimal table wine lot (high volatile acidity, lower alcohol)
    lot_a = {
        "fixed acidity": 7.2, "volatile acidity": 0.72, "citric acid": 0.05,
        "residual sugar": 2.2, "chlorides": 0.082, "free sulfur dioxide": 12.0,
        "total sulfur dioxide": 38.0, "density": 0.9975, "pH": 3.48,
        "sulphates": 0.52, "alcohol": 9.6
    }
    # Lot B: High-structure reserve lot (low volatile acidity, high alcohol, ripe acid)
    lot_b = {
        "fixed acidity": 10.4, "volatile acidity": 0.32, "citric acid": 0.48,
        "residual sugar": 2.6, "chlorides": 0.064, "free sulfur dioxide": 24.0,
        "total sulfur dioxide": 58.0, "density": 0.9958, "pH": 3.22,
        "sulphates": 0.82, "alcohol": 12.8
    }

    optimizer = WineAssemblageOptimizer()
    df_grid = optimizer.simulate_blend_grid_2lots(lot_a, lot_b, steps=80)
    opt_res = optimizer.optimize_blend([lot_a, lot_b], lot_names=["Tank A (Sub-optimal)", "Tank B (Reserve)"])

    # Panel 1: Trajectory Curve
    ax1 = axes[0]
    ax1.plot(
        df_grid["weight_lot_b"] * 100.0,
        df_grid["predicted_probability"],
        color="#800020",
        linewidth=3.0,
        label="Predicted P(Premium Quality ≥ 7)"
    )

    # Highlight screening threshold
    ax1.axhline(
        DEFAULT_SCREENING_THRESHOLD,
        color="#1b4d3e",
        linestyle="--",
        linewidth=1.8,
        label=f"Screening Threshold (τ* = {DEFAULT_SCREENING_THRESHOLD})"
    )

    # Mark optimal blend point
    opt_w_b = opt_res["weights_breakdown"][1]["percentage"]
    opt_p = opt_res["predicted_premium_probability"]
    ax1.scatter(
        [opt_w_b], [opt_p],
        color="#d4af37",
        s=180,
        zorder=5,
        edgecolor="black",
        linewidth=1.5,
        label=f"Optimal Coupage: {opt_w_b:.1f}% Tank B (P = {opt_p:.3f})"
    )

    # Shade viable Reserve bottling zone
    viable_mask = df_grid["is_premium"] & df_grid["is_enologically_compliant"]
    if viable_mask.any():
        min_b = df_grid.loc[viable_mask, "weight_lot_b"].min() * 100.0
        max_b = df_grid.loc[viable_mask, "weight_lot_b"].max() * 100.0
        ax1.axvspan(min_b, max_b, color="#2e7d32", alpha=0.12, label="Viable Reserve Bottling Zone")

    ax1.set_title("Enological Coupage Trajectory (Tank A + Tank B Assemblage)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Blend Proportion of Tank B (% Lot B)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Probability P(Quality ≥ 7)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    # Panel 2: Physicochemical Balance Comparison
    ax2 = axes[1]
    compare_features = ["volatile acidity", "alcohol", "pH", "sulphates", "citric acid"]
    labels = ["Volatile Acidity (g/L)", "Alcohol (% vol)", "pH Balance", "Sulphates (g/L)", "Citric Acid (g/L)"]

    val_a = [lot_a[f] for f in compare_features]
    val_b = [lot_b[f] for f in compare_features]
    val_opt = [opt_res["blended_chemistry"][f] for f in compare_features]

    x = np.arange(len(compare_features))
    width = 0.25

    ax2.bar(x - width, val_a, width, label="Tank A (Single Lot)", color="#c0392b", alpha=0.85)
    ax2.bar(x, val_opt, width, label="Optimal Coupage Blend", color="#8e44ad", alpha=0.95)
    ax2.bar(x + width, val_b, width, label="Tank B (Single Lot)", color="#27ae60", alpha=0.85)

    # Annotate values
    for i in range(len(compare_features)):
        ax2.text(x[i] - width, val_a[i] + 0.15, f"{val_a[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
        ax2.text(x[i], val_opt[i] + 0.15, f"{val_opt[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#8e44ad")
        ax2.text(x[i] + width, val_b[i] + 0.15, f"{val_b[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax2.set_title("Physicochemical Rectification via Assemblage", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Chemical Concentration / Unit", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(val_b) * 1.25)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_conformal_prediction_coverage(
    save_path: Path = FIGURES_DIR / "day10_conformal_prediction_coverage.png"
) -> None:
    """
    Plots conformal prediction empirical coverage calibration and set size breakdown.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Load data
    df_red = load_wine_data("red")
    df_targets = create_binned_targets(df_red)
    df_feat = create_engineered_features(df_targets)

    X = df_feat[ALL_FEATURE_NAMES]
    y = df_feat["binary_quality"]
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_cal, y_tr, y_cal = get_stratified_train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    rf_model = joblib.load("models/champion_rf_pipeline.joblib")
    conformal = EnologicalConformalPredictor(model=rf_model)
    conformal.fit_calibration(X_cal, y_cal)

    alphas = np.linspace(0.01, 0.30, 30)
    df_cov = conformal.evaluate_coverage(X_test, y_test, alpha_levels=alphas)

    # Panel 1: Calibration Coverage Curve
    ax1 = axes[0]
    ax1.plot(df_cov["nominal_confidence"] * 100.0, df_cov["empirical_coverage"] * 100.0, color="#1565c0", linewidth=2.8, label="Empirical Coverage")
    ax1.plot([70, 100], [70, 100], color="#d32f2f", linestyle="--", linewidth=1.8, label="Perfect Validity Line (Coverage ≥ 1-α)")
    ax1.fill_between(df_cov["nominal_confidence"] * 100.0, df_cov["empirical_coverage"] * 100.0, df_cov["nominal_confidence"] * 100.0, color="#bbdefb", alpha=0.5)

    ax1.set_title("Conformal Calibration Reliability", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Target Nominal Confidence 1 - α (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Empirical Test Coverage (%)", fontsize=10, fontweight="bold")
    ax1.set_xlim(70, 100)
    ax1.set_ylim(70, 101)
    ax1.legend(loc="lower right", frameon=True)

    # Panel 2: Prediction Set Cardinality
    ax2 = axes[1]
    eval_alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
    df_discrete = conformal.evaluate_coverage(X_test, y_test, alpha_levels=eval_alphas)

    x = np.arange(len(eval_alphas))
    w = 0.55
    singletons = df_discrete["singleton_ratio"] * 100.0
    ambiguous = df_discrete["ambiguous_ratio"] * 100.0
    empty = df_discrete["empty_ratio"] * 100.0

    ax2.bar(x, singletons, w, label="Singleton (Confident Decision)", color="#2e7d32", alpha=0.85)
    ax2.bar(x, ambiguous, w, bottom=singletons, label="Ambiguous (Sommelier Panel Req.)", color="#f57c00", alpha=0.85)
    ax2.bar(x, empty, w, bottom=singletons + ambiguous, label="Empty (Chemical Anomaly)", color="#c62828", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"α={a}\n({int((1-a)*100)}% Conf)" for a in eval_alphas], fontsize=9, fontweight="bold")
    ax2.set_title("Conformal Set Efficiency vs Confidence", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Proportion of Test Samples (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower left", frameon=True)

    # Panel 3: Ambiguity Density
    ax3 = axes[2]
    probs = rf_model.predict_proba(X_test)[:, 1]
    q_90 = conformal.get_conformal_quantile(0.10)
    cutoff = max(0.0, 1.0 - q_90)
    amb_mask = (probs >= cutoff) & ((1.0 - probs) >= cutoff)

    sns.kdeplot(probs[~amb_mask], ax=ax3, color="#2e7d32", label="Confident Predictions (Size=1)", fill=True, alpha=0.3)
    if amb_mask.any():
        sns.kdeplot(probs[amb_mask], ax=ax3, color="#e65100", label="Ambiguous Specimens (Size=2)", fill=True, alpha=0.5)

    ax3.axvline(DEFAULT_SCREENING_THRESHOLD, color="#b71c1c", linestyle="--", label=f"Decision Boundary (τ* = {DEFAULT_SCREENING_THRESHOLD})")
    ax3.set_title("Density of Ambiguous Lots across Probability Space", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("Predicted Premium Probability P(Quality ≥ 7)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_winery_economic_threshold_utility(
    save_path: Path = FIGURES_DIR / "day10_winery_economic_threshold_utility.png"
) -> None:
    """
    Plots commercial Expected Monetary Value (EMV) profit curve and harvest financial impact.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), dpi=300)

    # Load data
    df_red = load_wine_data("red")
    df_targets = create_binned_targets(df_red)
    df_feat = create_engineered_features(df_targets)

    X = df_feat[ALL_FEATURE_NAMES]
    y = df_feat["binary_quality"]
    _, X_test, _, y_test = get_stratified_train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = joblib.load("models/champion_rf_pipeline.joblib")
    probs = rf_model.predict_proba(X_test)[:, 1]

    econ = WineryEconomicEngine()
    econ_opt = econ.optimize_financial_threshold(y_test.values, probs, lot_size_bottles=25000)
    df_curve = econ_opt["profit_curve"]

    # Panel 1: EMV Curve
    ax1 = axes[0]
    ax1.plot(df_curve["threshold"], df_curve["profit_per_bottle"], color="#004d40", linewidth=3.0, label="Expected Profit ($ / Bottle)")
    ax1.set_ylabel("Expected Monetary Value ($ / Bottle)", fontsize=11, fontweight="bold", color="#004d40")

    # Secondary axis for F1 Score
    ax1_f1 = ax1.twinx()
    ax1_f1.plot(df_curve["threshold"], df_curve["f1_score"], color="#e65100", linestyle=":", linewidth=2.0, label="F1 Classification Score")
    ax1_f1.set_ylabel("F1 Score", fontsize=11, fontweight="bold", color="#e65100")
    ax1_f1.grid(False)

    # Mark Optimal Points
    opt_th = econ_opt["optimal_financial_threshold"]
    opt_p = econ_opt["max_profit_per_bottle"]
    ax1.scatter([opt_th], [opt_p], color="#d50000", s=150, zorder=6, edgecolor="black", label=f"Profit Optimum: τ* = {opt_th:.2f} (${opt_p:.2f}/btl)")

    ax1.axvline(0.50, color="#616161", linestyle="--", alpha=0.7, label="Default Threshold: τ = 0.50")
    ax1.axvline(DEFAULT_SCREENING_THRESHOLD, color="#1b5e20", linestyle="-.", alpha=0.7, label=f"Screening Threshold: τ = {DEFAULT_SCREENING_THRESHOLD}")

    ax1.set_title("Winery Expected Monetary Value (EMV) vs Operating Threshold", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Operating Decision Threshold (τ)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True)

    # Panel 2: Harvest Lot Revenue Comparison (25,000 bottles)
    ax2 = axes[1]
    regimes = ["Default Standard\n(τ = 0.50)", f"F1 / Screening\n(τ = {DEFAULT_SCREENING_THRESHOLD})", f"Economic Optimum\n(τ* = {opt_th:.2f})"]
    def_rev = econ_opt["default_threshold_evaluation"]["harvest_lot_profit"]
    f1_rev = econ_opt["f1_threshold_evaluation"]["harvest_lot_profit"]
    opt_rev = econ_opt["max_harvest_profit"]

    revenues = [def_rev, f1_rev, opt_rev]
    colors = ["#78909c", "#43a047", "#1e88e5"]

    bars = ax2.bar(regimes, revenues, color=colors, width=0.55, edgecolor="black", linewidth=1.2)

    # Annotate dollar values
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 1500, f"${h:,.0f}", ha="center", fontsize=10, fontweight="bold")

    uplift = opt_rev - def_rev
    ax2.annotate(
        f"+${uplift:,.0f} Net Gain\n(+{(uplift/def_rev)*100:.1f}%)",
        xy=(2, opt_rev),
        xytext=(1.6, opt_rev * 1.08),
        arrowprops=dict(facecolor="#d50000", shrink=0.08, width=2, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#d50000"
    )

    ax2.set_title("Harvest Vintage Net Return (25,000 Commercial Bottles)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Commercial Harvest Net Profit ($)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(revenues) * 1.25)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_stacking_ensemble_calibration(
    save_path: Path = FIGURES_DIR / "day10_stacking_ensemble_calibration.png"
) -> None:
    """
    Plots probability reliability calibration curve and Brier score benchmark
    comparing Level-0 base models against the Level-1 Stacking Meta-Learner.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), dpi=300)

    # Load data
    df_red = load_wine_data("red")
    df_targets = create_binned_targets(df_red)
    df_feat = create_engineered_features(df_targets)

    X = df_feat[ALL_FEATURE_NAMES]
    y = df_feat["binary_quality"]
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or fit stacking ensemble
    stacker_path = Path("models/stacking_ensemble.joblib")
    if stacker_path.exists():
        stacker = WineStackingEnsemble.load(stacker_path)
    else:
        stacker = WineStackingEnsemble(random_state=42)
        X_tr, _, y_tr, _ = get_stratified_train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        stacker.fit(X_tr, y_tr)
        stacker.save(stacker_path, "models/stacking_ensemble.json")

    y_arr = y_test.values
    model_predictions = {}
    for name, m in stacker.base_models.items():
        model_predictions[name] = m.predict_proba(X_test)[:, 1]
    model_predictions["stacking_meta_learner"] = stacker.predict_proba(X_test)[:, 1]

    palette = {
        "random_forest": "#2e7d32",
        "gradient_boosting": "#f57c00",
        "extra_trees": "#6a1b9a",
        "logistic_regression": "#0288d1",
        "stacking_meta_learner": "#d50000"
    }
    display_names = {
        "random_forest": "Random Forest (Champion)",
        "gradient_boosting": "Gradient Boosted Trees (GBDT)",
        "extra_trees": "Extra Trees Classifier",
        "logistic_regression": "Logistic Regression (ElasticNet)",
        "stacking_meta_learner": "★ Stacking Meta-Learner (Day 10)"
    }

    # Panel 1: Reliability Calibration Curves
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration Line", linewidth=1.5)

    brier_scores = {}
    roc_scores = {}

    for name, p in model_predictions.items():
        frac_pos, mean_pred = calibration_curve(y_arr, p, n_bins=8, strategy="uniform")
        brier = brier_score_loss(y_arr, p)
        roc = roc_auc_score(y_arr, p)
        brier_scores[name] = brier
        roc_scores[name] = roc

        lw = 3.2 if name == "stacking_meta_learner" else 1.8
        ls = "-" if name == "stacking_meta_learner" else "-."
        ax1.plot(mean_pred, frac_pos, marker="o", linewidth=lw, linestyle=ls, color=palette[name], label=f"{display_names[name]} (Brier={brier:.4f})")

    ax1.set_title("Probability Calibration Diagram (Reliability Curves)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Empirical Fraction of Positives", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    # Panel 2: Comparative ROC-AUC and Brier Score
    ax2 = axes[1]
    names_list = list(model_predictions.keys())
    x = np.arange(len(names_list))
    width = 0.35

    rocs = [roc_scores[n] for n in names_list]
    briers = [brier_scores[n] for n in names_list]

    bars1 = ax2.bar(x - width/2, rocs, width, label="ROC-AUC Score (Higher is Better)", color="#1565c0", alpha=0.85)
    bars2 = ax2.bar(x + width/2, [b * 5.0 for b in briers], width, label="Brier Loss x 5 (Lower is Better)", color="#e53935", alpha=0.85)

    for b in bars1:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    for i, b in enumerate(bars2):
        raw_b = briers[i]
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{raw_b:.4f}", ha="center", fontsize=8.5, fontweight="bold", color="#b71c1c")

    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("_", " ").title() for n in names_list], rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax2.set_title("Model Discriminative Power vs Calibration Error", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Normalized Metric Score", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_all_day10_figures() -> None:
    """Generate all 4 publication figures for Day 10."""
    print("Generating Day 10 Figure 1: Coupage Optimization Surface...")
    plot_coupage_optimization_surface()

    print("Generating Day 10 Figure 2: Conformal Prediction Coverage...")
    plot_conformal_prediction_coverage()

    print("Generating Day 10 Figure 3: Winery Economic Threshold Utility...")
    plot_winery_economic_threshold_utility()

    print("Generating Day 10 Figure 4: Stacking Ensemble Calibration...")
    plot_stacking_ensemble_calibration()
    print("All Day 10 figures successfully created in reports/figures/!")


if __name__ == "__main__":
    generate_all_day10_figures()
