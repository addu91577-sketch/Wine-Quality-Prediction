"""
Day 9 Publication Figures Generator.
Creates publication-quality figures for:
1. day9_cross_cultivar_chemical_divergence.png: Comparative KDE distributions across Red vs White cultivars.
2. day9_domain_transfer_gap.png: In-domain vs zero-shot cross-cultivar model degradation benchmark.
3. day9_white_champion_roc_pr_curves.png: White Wine Champion ROC and PR curves with operating threshold tau*=0.35.
4. day9_counterfactual_recourse_waterfall.png: Enological recourse intervention waterfall elevating standard wine to reserve.
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
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
import joblib

from src.data_loader import load_wine_data
from src.feature_engineering import create_engineered_features
from src.cross_cultivar_and_recourse import (
    CultivarDomainAnalyzer,
    WhiteWineChampionTrainer,
    CellarRecourseOptimizer,
    WHITE_CHAMPION_MODEL_PATH,
    DEFAULT_WHITE_THRESHOLD
)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_cross_cultivar_chemical_divergence(
    save_path: Path = FIGURES_DIR / "day9_cross_cultivar_chemical_divergence.png"
) -> None:
    """
    Plots multi-panel KDE distributions showing fundamental chemical divergences
    between Red and White wine populations.
    """
    red_df = load_wine_data("red")
    white_df = load_wine_data("white")

    key_features = [
        ("total sulfur dioxide", "Total SO₂ (mg/dm³)", "Antioxidant / Preservative Capacity"),
        ("volatile acidity", "Volatile Acidity (g/dm³)", "Acetic Spoilage / Freshness Index"),
        ("chlorides", "Chlorides (g/dm³)", "Mineral Salinity / Soil Ion Balance"),
        ("residual sugar", "Residual Sugar (g/dm³)", "Sweetness / Fermentation Completeness"),
        ("fixed acidity", "Fixed Acidity (g/dm³)", "Tartaric / Malic Acid Skeleton"),
        ("sulphates", "Sulphates (g/dm³)", "Potassium Sulphate Preservative Structure")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for idx, (feat, xlabel, title) in enumerate(key_features):
        ax = axes[idx]
        sns.kdeplot(red_df[feat], ax=ax, color="#8B0000", fill=True, alpha=0.35, linewidth=2, label="Red Wine (n=1,599)")
        sns.kdeplot(white_df[feat], ax=ax, color="#DAA520", fill=True, alpha=0.35, linewidth=2, label="White Wine (n=4,898)")

        red_mean = red_df[feat].mean()
        white_mean = white_df[feat].mean()

        ax.axvline(red_mean, color="#8B0000", linestyle="--", linewidth=1.5, alpha=0.9, label=f"Red μ: {red_mean:.2f}")
        ax.axvline(white_mean, color="#B8860B", linestyle="--", linewidth=1.5, alpha=0.9, label=f"White μ: {white_mean:.2f}")

        ax.set_title(f"{feat.title()}\n({title})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(
        "Day 9: Cross-Cultivar Physicochemical Divergence (Red vs White Viticulture Chemistry)",
        fontsize=15, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_domain_transfer_gap(
    save_path: Path = FIGURES_DIR / "day9_domain_transfer_gap.png"
) -> None:
    """
    Plots side-by-side grouped bar chart illustrating the domain transfer fragility
    when the Red Champion model is applied zero-shot to White wine vs dedicated White Champion.
    """
    metrics = ["ROC-AUC", "Balanced Accuracy", "Precision (tau=0.35)", "Recall (tau=0.35)"]
    red_champion_on_red = [0.9368, 0.8520, 0.6380, 0.7910]
    red_champion_zero_shot_white = [0.7593, 0.6759, 0.4728, 0.5085]
    white_champion_on_white = [0.9243, 0.8433, 0.5413, 0.8962]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    b1 = ax.bar(x - width, red_champion_on_red, width, label="Red Champion (In-Domain: Red)", color="#8B0000", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x, red_champion_zero_shot_white, width, label="Red Champion (Zero-Shot Transfer: White)", color="#708090", edgecolor="black", alpha=0.85)
    b3 = ax.bar(x + width, white_champion_on_white, width, label="White Champion (Adapted Domain: White)", color="#DAA520", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
    ax.set_title("Day 9: Domain Adaptation Benchmark — In-Domain vs Zero-Shot vs Adapted Champion", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)

    # Value labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Add alert callout
    ax.annotate(
        "Zero-Shot Degradation:\nROC-AUC drops -17.8%\nRecall drops -28.3%",
        xy=(x[0], 0.76), xytext=(x[0] - 0.2, 0.48),
        arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", edgecolor="#D32F2F", alpha=0.95),
        fontsize=9, fontweight="bold", color="#B71C1C"
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_white_champion_roc_pr_curves(
    save_path: Path = FIGURES_DIR / "day9_white_champion_roc_pr_curves.png"
) -> None:
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves
    for the White Wine Champion Random Forest model.
    """
    trainer = WhiteWineChampionTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(random_state=42)

    model = joblib.load(WHITE_CHAMPION_MODEL_PATH)
    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, roc_thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    baseline_pr = float(np.mean(y_test))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. ROC Curve
    ax1.plot(fpr, tpr, color="#DAA520", lw=2.5, label=f"White Champion RF (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    # Mark operating threshold tau = 0.35
    idx_35 = np.argmin(np.abs(roc_thresholds - DEFAULT_WHITE_THRESHOLD))
    ax1.plot(fpr[idx_35], tpr[idx_35], marker="o", markersize=9, color="#D32F2F", label=f"Operating τ* = 0.35 (TPR={tpr[idx_35]:.3f}, FPR={fpr[idx_35]:.3f})")

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("White Wine Champion: ROC Curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Precision-Recall Curve
    ax2.plot(recall, precision, color="#2E7D32", lw=2.5, label=f"White Champion RF (PR-AUC = {pr_auc:.4f})")
    ax2.axhline(baseline_pr, color="gray", lw=1.5, linestyle="--", label=f"Prevalence Baseline ({baseline_pr:.1%})")

    idx_pr_35 = np.argmin(np.abs(pr_thresholds - DEFAULT_WHITE_THRESHOLD))
    ax2.plot(recall[idx_pr_35], precision[idx_pr_35], marker="o", markersize=9, color="#D32F2F",
             label=f"Operating τ* = 0.35 (Prec={precision[idx_pr_35]:.3f}, Rec={recall[idx_pr_35]:.3f})")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Coverage of Premium White Wines)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax2.set_title("White Wine Champion: Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Day 9: White Wine Champion Model Discrimination Performance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_counterfactual_recourse_waterfall(
    save_path: Path = FIGURES_DIR / "day9_counterfactual_recourse_waterfall.png"
) -> None:
    """
    Plots a waterfall chart illustrating how sequential cellar recourse interventions
    elevate a sub-premium wine specimen's posterior probability above the screening threshold.
    """
    sample = {
        "fixed acidity": 7.4, "volatile acidity": 0.68, "citric acid": 0.05,
        "residual sugar": 2.1, "chlorides": 0.082, "free sulfur dioxide": 10.0,
        "total sulfur dioxide": 32.0, "density": 0.9972, "pH": 3.48,
        "sulphates": 0.54, "alcohol": 9.5
    }

    opt = CellarRecourseOptimizer()
    recourse = opt.optimize_recourse(sample, cultivar="red", target_prob=0.45)

    steps = ["Baseline (Table Wine)"]
    probs = [recourse["initial_prob"]]

    # Intermediate progressive impacts
    cur = sample.copy()
    for inv in recourse["interventions"]:
        feat = inv["feature"]
        cur[feat] = inv["target_value"]
        p_step = opt._predict_prob(cur, "red")
        steps.append(f"Adj: {feat.title()}")
        probs.append(p_step)

    # Ensure projected is final
    steps.append("Final Recourse (Reserve)")
    probs.append(recourse["projected_prob"])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(steps))

    colors = ["#708090"] + ["#1E88E5"] * (len(steps) - 2) + ["#2E7D32"]
    bars = ax.bar(x, probs, color=colors, edgecolor="black", width=0.6, alpha=0.88)

    ax.axhline(0.35, color="#D32F2F", linestyle="--", linewidth=2.0, label="Reserve Screening Threshold (τ* = 0.35)")

    ax.set_ylabel("Posterior Probability P(Good >= 7)", fontsize=11, fontweight="bold")
    ax.set_title("Day 9: Prescriptive Cellar Recourse Pathway — From Table Wine to Reserve Grade", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=18, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(probs) * 1.35)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper left")

    # Annotations
    for bar, p in zip(bars, probs):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_day9_figures():
    """Generates all 4 publication figures for Day 9."""
    print("Generating Day 9 figures...")
    plot_cross_cultivar_chemical_divergence()
    plot_domain_transfer_gap()
    plot_white_champion_roc_pr_curves()
    plot_counterfactual_recourse_waterfall()
    print("All Day 9 figures successfully generated!")


if __name__ == "__main__":
    generate_all_day9_figures()
