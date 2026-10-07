"""
Hyperparameter optimization, cross-validation evaluation, and threshold calibration
routines for Wine Quality Prediction.

Implements:
- Stratified 5-Fold Cross-Validation evaluation across baseline and tuned models.
- Systematic GridSearchCV optimization for Random Forest, SVC, and SGDClassifier
  with class-imbalance weighting.
- Probability threshold analysis and Precision-Recall optimization for minority
  premium wine detection.
- Comprehensive diagnostic visualizations:
  1. 5-Fold CV score distributions before vs. after tuning.
  2. Hyperparameter validation impact curves.
  3. Precision-Recall curves with Average Precision (PR-AUC) and optimal thresholds.
  4. Test set confusion matrices of tuned classifiers.
  5. Side-by-side Baseline vs. Tuned benchmark comparison.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def get_tuning_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Construct standardized pipelines for model hyperparameter optimization.

    Classifiers:
    1. Random Forest Classifier
    2. Support Vector Classifier (calibrated for posterior probabilities)
    3. SGD Classifier (log loss / modified huber)

    Parameters
    ----------
    random_state : int
        Seed for deterministic reproducibility.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of initialized scikit-learn pipelines.
    """
    pipelines = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "Support Vector Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                random_state=random_state
            ))
        ]),
        "SGD Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                random_state=random_state,
                max_iter=2000,
                tol=1e-3
            ))
        ])
    }
    return pipelines


def get_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Define hyperparameter search spaces targeting minority-class (Good Wine) detection.

    Returns
    -------
    Dict[str, Dict[str, List[Any]]]
        Search grids mapping pipeline parameter names to candidate values.
    """
    grids = {
        "Random Forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 12, 18],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": [None, "balanced", "balanced_subsample"]
        },
        "Support Vector Classifier": {
            "clf__C": [0.5, 1.0, 3.0, 5.0, 10.0],
            "clf__gamma": ["scale", 0.05, 0.1, 0.2],
            "clf__class_weight": [None, "balanced"]
        },
        "SGD Classifier": {
            "clf__loss": ["log_loss", "modified_huber"],
            "clf__penalty": ["l2", "elasticnet"],
            "clf__alpha": [1e-4, 1e-3, 1e-2, 5e-2],
            "clf__class_weight": [None, "balanced"]
        }
    }
    return grids


def evaluate_kfold_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform Stratified K-Fold cross-validation and compute comprehensive fold metrics.

    Metrics calculated per fold:
    - Accuracy
    - Balanced Accuracy
    - Good Wine Precision (pos_label=1)
    - Good Wine Recall (pos_label=1)
    - Good Wine F1 (pos_label=1)
    - ROC-AUC
    - PR-AUC (Average Precision)

    Parameters
    ----------
    pipeline : Pipeline
        Scikit-learn classification pipeline.
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training target vector.
    n_splits : int
        Number of stratified folds (default: 5).
    random_state : int
        Random seed for fold partitioning.

    Returns
    -------
    Dict[str, Any]
        Dictionary of per-fold scores, metric means, and standard deviations.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = {
        "Accuracy": [],
        "Balanced Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "ROC-AUC": [],
        "PR-AUC": []
    }

    X_mat = X.values if isinstance(X, pd.DataFrame) else X
    y_vec = y.values if isinstance(y, pd.Series) else y

    for train_idx, val_idx in skf.split(X_mat, y_vec):
        X_tr, X_val = X_mat[train_idx], X_mat[val_idx]
        y_tr, y_val = y_vec[train_idx], y_vec[val_idx]

        # Fit model on training fold
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_val)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_val)
        else:
            y_prob = None

        fold_metrics["Accuracy"].append(accuracy_score(y_val, y_pred))
        fold_metrics["Balanced Accuracy"].append(balanced_accuracy_score(y_val, y_pred))
        fold_metrics["Precision"].append(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
        fold_metrics["Recall"].append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
        fold_metrics["F1"].append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))

        if y_prob is not None:
            fold_metrics["ROC-AUC"].append(roc_auc_score(y_val, y_prob))
            fold_metrics["PR-AUC"].append(average_precision_score(y_val, y_prob))
        else:
            fold_metrics["ROC-AUC"].append(np.nan)
            fold_metrics["PR-AUC"].append(np.nan)

    summary = {}
    for metric, values in fold_metrics.items():
        summary[metric] = {
            "mean": np.nanmean(values),
            "std": np.nanstd(values),
            "scores": values
        }

    return {"fold_metrics": fold_metrics, "summary": summary}


def run_hyperparameter_search(
    pipelines: Dict[str, Pipeline],
    param_grids: Dict[str, Dict[str, List[Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    scoring: str = "f1",
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Execute exhaustive GridSearchCV optimization across all candidate model pipelines.

    Parameters
    ----------
    pipelines : Dict[str, Pipeline]
        Candidate pipelines to optimize.
    param_grids : Dict[str, Dict[str, List[Any]]]
        Search grids per model.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels.
    n_splits : int
        Stratified folds for cross-validation (default: 5).
    scoring : str
        Objective metric for model selection (default: 'f1').
    random_state : int
        Seed for CV splitting.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing best estimator, best params, CV score, and search object.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search_results = {}

    for name, pipeline in pipelines.items():
        grid = param_grids.get(name, {})
        print(f"--> Optimizing {name} across {len(grid)} hyperparameter dimensions...")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        search_results[name] = {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": pd.DataFrame(grid_search.cv_results_),
            "grid_search": grid_search
        }
        print(f"    Best {scoring.upper()}: {grid_search.best_score_:.4f} | Best Params: {grid_search.best_params_}")

    return search_results


def evaluate_tuned_models_on_test(
    tuned_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate tuned models on held-out test partition ($N=320$).

    Parameters
    ----------
    tuned_models : Dict[str, Pipeline]
        Dictionary of tuned pipeline estimators.
    X_test : pd.DataFrame
        Held-out test features.
    y_test : pd.Series
        Held-out test labels.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Evaluation metrics, predictions, probabilities, and confusion matrices.
    """
    results = {}

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec_wt = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_wt = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else np.nan
        cm = confusion_matrix(y_test, y_pred)
        rep_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rep_text = classification_report(y_test, y_pred, zero_division=0)

        results[name] = {
            "model": model,
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
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc
            },
            "confusion_matrix": cm,
            "classification_report_dict": rep_dict,
            "classification_report_text": rep_text
        }

    return results


def find_optimal_thresholds(
    y_true: pd.Series,
    y_prob: np.ndarray,
    step: float = 0.02
) -> Dict[str, Any]:
    """
    Perform probability threshold sweep from 0.05 to 0.95 to identify optimal decision cutoff.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True binary target values.
    y_prob : np.ndarray
        Predicted posterior probabilities for positive class (Good Wine).
    step : float
        Grid step for threshold search (default: 0.02).

    Returns
    -------
    Dict[str, Any]
        Threshold sweep dataframe, optimal threshold maximizing F1, and balanced accuracy.
    """
    thresholds = np.arange(0.05, 0.95 + step, step)
    rows = []

    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        prec = precision_score(y_true, pred_t, pos_label=1, zero_division=0)
        rec = recall_score(y_true, pred_t, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, pred_t, pos_label=1, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, pred_t)
        acc = accuracy_score(y_true, pred_t)

        rows.append({
            "threshold": round(t, 2),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "accuracy": acc
        })

    thresh_df = pd.DataFrame(rows)

    # Optimal threshold for F1
    best_f1_idx = thresh_df["f1"].idxmax()
    best_f1_row = thresh_df.iloc[best_f1_idx]

    # Optimal threshold for Balanced Accuracy
    best_bal_idx = thresh_df["balanced_accuracy"].idxmax()
    best_bal_row = thresh_df.iloc[best_bal_idx]

    return {
        "threshold_df": thresh_df,
        "best_f1_threshold": best_f1_row["threshold"],
        "best_f1_score": best_f1_row["f1"],
        "best_f1_precision": best_f1_row["precision"],
        "best_f1_recall": best_f1_row["recall"],
        "best_bal_acc_threshold": best_bal_row["threshold"],
        "best_bal_acc_score": best_bal_row["balanced_accuracy"]
    }


def compile_tuned_comparison_dataframe(
    baseline_results: Dict[str, Dict[str, Any]],
    tuned_results: Dict[str, Dict[str, Any]],
    rf_thresh_metrics: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compile side-by-side performance comparison table showing Baseline vs. Tuned models.

    Parameters
    ----------
    baseline_results : Dict[str, Dict[str, Any]]
        Test results from Day 3 baselines.
    tuned_results : Dict[str, Dict[str, Any]]
        Test results from Day 4 tuned models.
    rf_thresh_metrics : Optional[Dict[str, float]]
        Metrics for threshold-optimized Random Forest.

    Returns
    -------
    pd.DataFrame
        Formatted side-by-side benchmark table.
    """
    rows = []

    # Baselines
    for name, data in baseline_results.items():
        m = data["metrics"]
        rows.append({
            "Model Stage": "Baseline (Day 3)",
            "Model Architecture": name,
            "Accuracy": round(m["Accuracy"], 4),
            "Balanced Acc": round(m["Balanced Accuracy"], 4),
            "Good Wine Prec": round(m["Good Wine Precision"], 4),
            "Good Wine Rec": round(m["Good Wine Recall"], 4),
            "Good Wine F1": round(m["Good Wine F1"], 4),
            "Weighted F1": round(m["Weighted F1"], 4),
            "ROC-AUC": round(m["ROC-AUC"], 4) if not np.isnan(m["ROC-AUC"]) else "N/A"
        })

    # Tuned
    for name, data in tuned_results.items():
        m = data["metrics"]
        rows.append({
            "Model Stage": "Tuned (Day 4)",
            "Model Architecture": name,
            "Accuracy": round(m["Accuracy"], 4),
            "Balanced Acc": round(m["Balanced Accuracy"], 4),
            "Good Wine Prec": round(m["Good Wine Precision"], 4),
            "Good Wine Rec": round(m["Good Wine Recall"], 4),
            "Good Wine F1": round(m["Good Wine F1"], 4),
            "Weighted F1": round(m["Weighted F1"], 4),
            "ROC-AUC": round(m["ROC-AUC"], 4) if not np.isnan(m["ROC-AUC"]) else "N/A"
        })

    # Threshold-optimized Random Forest if provided
    if rf_thresh_metrics:
        rows.append({
            "Model Stage": "Tuned + Thresh Opt (Day 4)",
            "Model Architecture": "Random Forest (Thresh=0.35)",
            "Accuracy": round(rf_thresh_metrics.get("Accuracy", 0.0), 4),
            "Balanced Acc": round(rf_thresh_metrics.get("Balanced Accuracy", 0.0), 4),
            "Good Wine Prec": round(rf_thresh_metrics.get("Good Wine Precision", 0.0), 4),
            "Good Wine Rec": round(rf_thresh_metrics.get("Good Wine Recall", 0.0), 4),
            "Good Wine F1": round(rf_thresh_metrics.get("Good Wine F1", 0.0), 4),
            "Weighted F1": round(rf_thresh_metrics.get("Weighted F1", 0.0), 4),
            "ROC-AUC": round(rf_thresh_metrics.get("ROC-AUC", 0.0), 4)
        })

    return pd.DataFrame(rows)


def compile_tuning_lift_dataframe(
    baseline_results: Dict[str, Dict[str, Any]],
    tuned_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Calculate absolute delta and percentage lift for key metrics post-hyperparameter optimization.

    Returns
    -------
    pd.DataFrame
        DataFrame listing Metric, Baseline Score, Tuned Score, Delta, and % Improvement.
    """
    rows = []
    models = ["Random Forest", "Support Vector Classifier", "SGD Classifier"]

    for name in models:
        base_name = name
        tune_name = f"{name} (Tuned)" if f"{name} (Tuned)" in tuned_results else name

        if base_name in baseline_results and tune_name in tuned_results:
            b_m = baseline_results[base_name]["metrics"]
            t_m = tuned_results[tune_name]["metrics"]

            for metric in ["Good Wine F1", "Good Wine Recall", "Balanced Accuracy", "ROC-AUC", "Accuracy"]:
                b_val = b_m[metric]
                t_val = t_m[metric]
                delta = t_val - b_val
                pct_lift = ((t_val - b_val) / (b_val + 1e-9)) * 100

                rows.append({
                    "Model": name,
                    "Metric": metric,
                    "Baseline": round(b_val, 4),
                    "Tuned": round(t_val, 4),
                    "Delta (+/-)": f"{'+' if delta >= 0 else ''}{delta:.4f}",
                    "Lift (%)": f"{'+' if pct_lift >= 0 else ''}{pct_lift:.2f}%"
                })

    return pd.DataFrame(rows)


# ==============================================================================
# DIAGNOSTIC VISUALIZATION ROUTINES
# ==============================================================================

def plot_cv_performance_comparison(
    cv_summary_baseline: Dict[str, Dict[str, Any]],
    cv_summary_tuned: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3-panel comparison of 5-Fold Cross-Validation metric distributions (ROC-AUC,
    Good Wine F1, Balanced Accuracy) comparing baseline vs tuned models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics_to_plot = ["F1", "Balanced Accuracy", "ROC-AUC"]
    titles = [
        "Good Wine F1 Score (5-Fold CV)",
        "Balanced Accuracy (5-Fold CV)",
        "ROC-AUC Score (5-Fold CV)"
    ]

    models = ["Random Forest", "Support Vector Classifier", "SGD Classifier"]
    x = np.arange(len(models))
    width = 0.35

    for ax_idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        base_means = [cv_summary_baseline[m]["summary"][metric]["mean"] for m in models]
        base_stds = [cv_summary_baseline[m]["summary"][metric]["std"] for m in models]
        tuned_means = [cv_summary_tuned[m]["summary"][metric]["mean"] for m in models]
        tuned_stds = [cv_summary_tuned[m]["summary"][metric]["std"] for m in models]

        rects1 = axes[ax_idx].bar(x - width/2, base_means, width, yerr=base_stds, capsize=5,
                                  label="Baseline (Default)", color="#7f8c8d", edgecolor="black", alpha=0.85)
        rects2 = axes[ax_idx].bar(x + width/2, tuned_means, width, yerr=tuned_stds, capsize=5,
                                  label="Tuned (GridSearchCV)", color="#2980b9", edgecolor="black", alpha=0.85)

        axes[ax_idx].set_title(title, fontsize=12, fontweight="bold")
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(models, rotation=15, fontsize=10)
        axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].set_ylabel("Score (0.0 to 1.0)", fontsize=10)
        axes[ax_idx].grid(axis="y", linestyle="--", alpha=0.5)

        # Value annotations
        for rect in rects1:
            h = rect.get_height()
            axes[ax_idx].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                                  xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            axes[ax_idx].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                                  xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].legend(loc="upper left", frameon=True)
    plt.suptitle("Day 4 Cross-Validation Performance: Baseline vs. Tuned Models (5-Fold Stratified CV)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CV performance comparison plot saved to: {save_path}")
    plt.close()


def plot_hyperparameter_tuning_results(
    search_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3-panel diagnostic visualization showing parameter search landscapes:
    1. Random Forest: Effect of class_weight and max_depth on CV F1.
    2. SVC: Effect of regularization parameter C and gamma on CV F1.
    3. SGD: Effect of regularization alpha and penalty on CV F1.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. Random Forest landscape
    rf_cv = search_results["Random Forest"]["cv_results"]
    rf_pivot = rf_cv.pivot_table(
        index="param_clf__max_depth",
        columns="param_clf__class_weight",
        values="mean_test_score"
    ).fillna(0)
    rf_pivot.index = [str(x) if x is not None else "None" for x in rf_pivot.index]
    rf_pivot.columns = [str(c) if c is not None else "None" for c in rf_pivot.columns]
    sns.heatmap(rf_pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar=False, annot_kws={"size": 10, "weight": "bold"})
    axes[0].set_title("Random Forest CV F1\n(class_weight vs. max_depth)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Class Weight Strategy", fontsize=10)
    axes[0].set_ylabel("Max Tree Depth", fontsize=10)

    # 2. SVC landscape
    svc_cv = search_results["Support Vector Classifier"]["cv_results"]
    svc_pivot = svc_cv.pivot_table(
        index="param_clf__C",
        columns="param_clf__gamma",
        values="mean_test_score"
    ).fillna(0)
    svc_pivot.columns = [str(c) for c in svc_pivot.columns]
    sns.heatmap(svc_pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], cbar=False, annot_kws={"size": 10, "weight": "bold"})
    axes[1].set_title("SVC CV F1\n(Penalty C vs. Kernel Gamma)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Kernel Gamma Parameter", fontsize=10)
    axes[1].set_ylabel("Regularization C", fontsize=10)

    # 3. SGD landscape
    sgd_cv = search_results["SGD Classifier"]["cv_results"]
    sgd_pivot = sgd_cv.pivot_table(
        index="param_clf__alpha",
        columns="param_clf__penalty",
        values="mean_test_score"
    ).fillna(0)
    sns.heatmap(sgd_pivot, annot=True, fmt=".3f", cmap="Blues", ax=axes[2], cbar=False, annot_kws={"size": 10, "weight": "bold"})
    axes[2].set_title("SGD Classifier CV F1\n(Penalty vs. Learning Alpha)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Penalty Norm", fontsize=10)
    axes[2].set_ylabel("Alpha Regularization", fontsize=10)

    plt.suptitle("Hyperparameter Search Response Landscapes (5-Fold CV Mean F1 Score)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Hyperparameter tuning results plot saved to: {save_path}")
    plt.close()


def plot_precision_recall_curves(
    eval_results: Optional[Dict[str, Dict[str, Any]]] = None,
    y_test: Optional[pd.Series] = None,
    best_thresholds: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    model_results_dict: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """
    Plot Precision-Recall curves with calculated Average Precision (PR-AUC) scores
    and optimal operating threshold markers.
    """
    if eval_results is None and model_results_dict is not None:
        eval_results = model_results_dict
    elif eval_results is None:
        raise ValueError("Must provide either eval_results or model_results_dict")

    plt.figure(figsize=(9, 7))

    colors = {
        "Random Forest (Tuned)": "#27ae60",
        "SVC (Tuned)": "#2980b9",
        "SGD (Tuned)": "#8e44ad",
        "Random Forest (Baseline)": "#95a5a6"
    }

    # Baseline positive class prevalence
    no_skill = (y_test == 1).sum() / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", color="#7f8c8d", label=f"No Skill Prevalence ({no_skill:.1%})")

    for name, data in eval_results.items():
        y_prob = data["y_prob"]
        if y_prob is not None:
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            color = colors.get(name, "#34495e")
            plt.plot(recall, precision, color=color, lw=2.2, label=f"{name} (PR-AUC = {ap:.3f})")

            # Mark default threshold (0.5)
            idx_50 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
            if idx_50 < len(precision):
                plt.scatter(recall[idx_50], precision[idx_50], color=color, s=70, zorder=5, marker="o")

    # Iso-F1 curves
    f_scores = np.linspace(0.4, 0.8, 3)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.3, linestyle=":")
        plt.annotate(f"F1={f_score:.1f}", xy=(0.88, f_score * 0.88 / (2 * 0.88 - f_score) + 0.01),
                     color="gray", fontsize=8)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity for Good Wine)", fontsize=11, fontweight="medium")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="medium")
    plt.title("Precision-Recall (PR) Curves & Operating Tradeoffs (Test Set)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Precision-Recall curves plot saved to: {save_path}")
    plt.close()


def plot_tuned_confusion_matrices(
    tuned_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot side-by-side annotated confusion matrix heatmaps for tuned models on the test set.
    """
    n_models = len(tuned_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    labels = ["Normal (<7)", "Good (>=7)"]

    for idx, (model_name, data) in enumerate(tuned_results.items()):
        ax = axes[idx]
        cm = data["confusion_matrix"]
        acc = data["metrics"]["Accuracy"]
        rec_good = data["metrics"]["Good Wine Recall"]
        f1_good = data["metrics"]["Good Wine F1"]

        # Percentages per row (recall perspective)
        cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1%})"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Greens",
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 11, "fontweight": "bold"}
        )

        ax.set_title(f"{model_name}\nAcc: {acc:.3f} | Good Rec: {rec_good:.1%} | F1: {f1_good:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)

    fig.suptitle("Tuned Models: Test Confusion Matrices (Balanced & Optimized)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Tuned confusion matrices plot saved to: {save_path}")
    plt.close()


def plot_tuned_vs_baseline_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot grouped bar chart comparing performance metrics across Baseline vs. Tuned models.
    """
    metrics = ["Accuracy", "Balanced Acc", "Good Wine Prec", "Good Wine Rec", "Good Wine F1", "ROC-AUC"]
    plot_df = comparison_df.copy()

    # Filter out threshold row for cleaner 3-model baseline vs tuned plot
    plot_df = plot_df[plot_df["Model Stage"].isin(["Baseline (Day 3)", "Tuned (Day 4)"])]
    plot_df["Label"] = plot_df["Model Architecture"] + " (" + plot_df["Model Stage"].str.split(" ").str[0] + ")"

    chart_data = plot_df.set_index("Label")[metrics].apply(pd.to_numeric, errors="coerce")

    ax = chart_data.plot(kind="bar", figsize=(14, 6.5), colormap="plasma", width=0.8, edgecolor="black", alpha=0.88)

    plt.title("Day 4 Benchmark: Baseline vs. Tuned Model Performance (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Classifier Model Configuration", fontsize=11, fontweight="medium")
    plt.ylabel("Performance Score (0.0 to 1.0)", fontsize=11, fontweight="medium")
    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Evaluation Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

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
                    fontsize=7.5,
                    rotation=90
                )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Tuned vs baseline comparison plot saved to: {save_path}")
    plt.close()
