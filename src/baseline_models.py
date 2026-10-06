"""
Baseline classification models, evaluation pipelines, and diagnostic visualizations
for Wine Quality Prediction.

Implements:
- Scaler-wrapped scikit-learn pipelines for Random Forest, SGDClassifier, and SVC.
- Unified model training, cross-fold metrics, and performance evaluation.
- Diagnostic visualization routines: target distribution comparisons, confusion matrices,
  model benchmark comparisons, and ROC curves.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


def get_baseline_classifiers(
    scaler_type: str = "standard",
    random_state: int = 42
) -> Dict[str, Pipeline]:
    """
    Construct dictionary of baseline classifier pipelines.

    Classifiers included:
    1. Random Forest (Ensemble Tree Classifier)
    2. Stochastic Gradient Descent (SGD) with log_loss
    3. Support Vector Classifier (SVC) with RBF kernel and probability calibration

    Parameters
    ----------
    scaler_type : str
        Type of scaler for feature normalization: 'standard' or 'robust'.
    random_state : int
        Seed for deterministic execution.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of initialized pipeline objects.
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()

    models = {
        "Random Forest": Pipeline([
            # Tree models are scale-invariant, but scaler included for uniformity
            ("scaler", scaler),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "SGD Classifier": Pipeline([
            ("scaler", scaler),
            ("clf", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                random_state=random_state
            ))
        ]),
        "Support Vector Classifier": Pipeline([
            ("scaler", scaler),
            ("clf", CalibratedClassifierCV(
                SVC(
                    kernel="rbf",
                    C=1.0,
                    random_state=random_state
                ),
                ensemble=False
            ))
        ])
    }
    return models


def train_and_evaluate_baselines(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Fit each baseline model on the training set and compute comprehensive test diagnostics.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Dictionary of baseline model pipelines.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing target labels.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing trained pipelines, predictions, probabilities, and evaluation metrics.
    """
    results = {}

    for name, pipeline in models.items():
        # Fit pipeline (ensures no data leakage)
        pipeline.fit(X_train, y_train)

        # Predict labels and class probabilities
        y_pred = pipeline.predict(X_test)
        
        # Determine probability scores for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = None

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec_wt = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_wt = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Class 1 (Good Wine) specific metrics
        prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, zero_division=0)

        results[name] = {
            "model": pipeline,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": {
                "Accuracy": acc,
                "Balanced Accuracy": bal_acc,
                "Weighted Precision": prec_wt,
                "Weighted Recall": rec_wt,
                "Weighted F1": f1_wt,
                "Good Wine Precision": prec_pos,
                "Good Wine Recall": rec_pos,
                "Good Wine F1": f1_pos,
                "ROC-AUC": roc_auc
            },
            "confusion_matrix": cm,
            "classification_report_dict": report_dict,
            "classification_report_text": report_text
        }

    return results


def compile_comparison_dataframe(
    eval_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compile side-by-side performance comparison table for all evaluated models.
    """
    rows = []
    for model_name, data in eval_results.items():
        m = data["metrics"]
        rows.append({
            "Model": model_name,
            "Accuracy": round(m["Accuracy"], 4),
            "Balanced Acc": round(m["Balanced Accuracy"], 4),
            "Weighted Precision": round(m["Weighted Precision"], 4),
            "Weighted Recall": round(m["Weighted Recall"], 4),
            "Weighted F1": round(m["Weighted F1"], 4),
            "Good Wine Precision": round(m["Good Wine Precision"], 4),
            "Good Wine Recall": round(m["Good Wine Recall"], 4),
            "Good Wine F1": round(m["Good Wine F1"], 4),
            "ROC-AUC": round(m["ROC-AUC"], 4) if not np.isnan(m["ROC-AUC"]) else "N/A"
        })

    return pd.DataFrame(rows).sort_values(by="Weighted F1", ascending=False).reset_index(drop=True)


def plot_target_distribution_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create a 3-panel visual comparison illustrating Raw Quality Scores, 3-Class Binning,
    and Binary Target Formulations to justify class formulation choices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    total = len(df)

    # 1. Raw Scores
    raw_counts = df["quality"].value_counts().sort_index()
    palette_raw = sns.color_palette("mako", n_colors=len(raw_counts))
    bars1 = axes[0].bar(raw_counts.index.astype(str), raw_counts.values, color=palette_raw, edgecolor="black", alpha=0.85)
    axes[0].set_title("1. Raw Discrete Quality Scores (3–8)\nSevere Class Imbalance (Scores 5 & 6 = 82.5%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Quality Score", fontsize=10)
    axes[0].set_ylabel("Sample Count", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        pct = (yval / total) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 12, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(raw_counts.values) * 1.18)

    # 2. 3-Class Binning
    three_order = ["Low (3-4)", "Medium (5-6)", "High (7-8)"]
    three_counts = df["three_class_quality"].value_counts().reindex(three_order)
    colors_three = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars2 = axes[1].bar(three_order, three_counts.values, color=colors_three, edgecolor="black", alpha=0.85)
    axes[1].set_title("2. 3-Class Formulation\n(Low: 3.9% | Medium: 82.5% | High: 13.6%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Class Category", fontsize=10)
    axes[1].set_ylabel("Sample Count", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        pct = (yval / total) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 20, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(three_counts.values) * 1.15)

    # 3. Binary Formulation
    bin_labels = ["Normal / Low (< 7)", "Good / Premium (>= 7)"]
    bin_counts = df["binary_quality"].value_counts().sort_index()
    colors_bin = ["#3498db", "#27ae60"]
    bars3 = axes[2].bar(bin_labels, bin_counts.values, color=colors_bin, edgecolor="black", alpha=0.85)
    axes[2].set_title("3. Binary Target Formulation (Selected)\n(Normal/Low: 86.4% | Good Wine: 13.6%)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Target Class", fontsize=10)
    axes[2].set_ylabel("Sample Count", fontsize=10)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        pct = (yval / total) * 100
        axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 25, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].set_ylim(0, max(bin_counts.values) * 1.15)

    plt.suptitle("Target Formulation Comparison: Raw vs. 3-Class vs. Binary Formulation", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Target distribution comparison plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrices(
    eval_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot side-by-side annotated confusion matrix heatmaps for all baseline models.
    """
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    labels = ["Normal (<7)", "Good (>=7)"]

    for idx, (model_name, data) in enumerate(eval_results.items()):
        ax = axes[idx]
        cm = data["confusion_matrix"]
        acc = data["metrics"]["Accuracy"]
        f1_good = data["metrics"]["Good Wine F1"]

        # Percentages per row (recall perspective)
        cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Annotations showing both count and recall %
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1%})"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 11, "fontweight": "bold"}
        )

        ax.set_title(f"{model_name}\nAcc: {acc:.3f} | Good F1: {f1_good:.3f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)

    fig.suptitle("Baseline Models: Confusion Matrix Comparison (Test Set)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrices plot saved to: {save_path}")
    plt.close()


def plot_baseline_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot grouped bar chart comparing performance metrics across baseline models.
    """
    metrics = ["Accuracy", "Balanced Acc", "Weighted F1", "Good Wine F1", "ROC-AUC"]
    
    # Filter to only numeric metrics available
    plot_df = comparison_df.set_index("Model")[metrics].copy()
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    ax = plot_df.plot(kind="bar", figsize=(12, 6), colormap="viridis", width=0.75, edgecolor="black", alpha=0.85)

    plt.title("Baseline Model Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Classifier Pipeline", fontsize=11, fontweight="medium")
    plt.ylabel("Metric Score (0.0 to 1.0)", fontsize=11, fontweight="medium")
    plt.xticks(rotation=0, fontsize=10)
    plt.ylim(0, 1.08)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Evaluation Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # Label heights of bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Baseline comparison plot saved to: {save_path}")
    plt.close()


def plot_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: Optional[str] = None
) -> None:
    """
    Plot overlaid Receiver Operating Characteristic (ROC) curves for baseline models.
    """
    plt.figure(figsize=(9, 7))

    colors = ["#2ecc71", "#e67e22", "#9b59b6", "#34495e"]

    for idx, (name, data) in enumerate(eval_results.items()):
        y_prob = data["y_prob"]
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = data["metrics"]["ROC-AUC"]
            color = colors[idx % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="medium")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="medium")
    plt.title("Receiver Operating Characteristic (ROC) Curves - Baseline Models", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves plot saved to: {save_path}")
    plt.close()
