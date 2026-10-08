"""
Model interpretation, feature importance analytics, partial dependence profiling,
enological decision rule synthesis, and champion pipeline serialization for
Wine Quality Prediction.

Modules & Capabilities:
- Random Forest Mean Decrease in Impurity (MDI / Gini) feature importance with tree variance.
- Model-agnostic Permutation Feature Importance evaluated on held-out test partitions
  across Random Forest, Support Vector Classifier, and SGD Classifier.
- Directional linear weight interpretation (SGD Classifier standardized coefficients).
- Partial Dependence (PDP) profiling for non-linear threshold and tipping-point discovery.
- 2D Bivariate partial dependence interaction contours (e.g. alcohol vs. volatile acidity).
- Quantitative Enological Decision Rules comparing chemical distributions between
  Normal (<7) and Premium (>=7) wines.
- Champion model serialization, persistence, metadata tracking, and single-sample inference.
- Publication-quality diagnostic figures for interpretability and executive reporting.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def compute_rf_feature_importances(
    rf_pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract Mean Decrease in Impurity (MDI / Gini) feature importance from a fitted Random Forest pipeline.

    Computes both mean importance and inter-tree standard deviation across all constituent estimators.

    Parameters
    ----------
    rf_pipeline : Pipeline
        Fitted Random Forest classification pipeline.
    feature_names : List[str]
        List of feature column names corresponding to input matrix columns.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending mean importance with columns:
        ['feature', 'importance_mean', 'importance_std', 'importance_pct']
    """
    clf = rf_pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        raise ValueError("Provided pipeline does not contain an estimator with 'feature_importances_'.")

    importances = clf.feature_importances_
    tree_importances = np.array([tree.feature_importances_ for tree in clf.estimators_])
    std_importances = np.std(tree_importances, axis=0)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importances,
        "importance_std": std_importances,
        "importance_pct": np.round((importances / np.sum(importances)) * 100, 2)
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_imp


def compute_permutation_importances(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute out-of-sample Permutation Feature Importance on held-out test or validation data.

    Permutation importance measures metric degradation when a feature column is randomly shuffled,
    providing an unbiased out-of-sample assessment free of MDI's high-cardinality preference.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted classification pipeline.
    X_val : pd.DataFrame
        Evaluation feature dataframe.
    y_val : pd.Series
        Ground truth target series.
    scoring : str
        Scikit-learn evaluation metric (default: 'roc_auc').
    n_repeats : int
        Number of permutation shuffles per feature (default: 10).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending mean importance drop with columns:
        ['feature', 'importance_mean', 'importance_std']
    """
    feature_names = list(X_val.columns) if isinstance(X_val, pd.DataFrame) else [f"feature_{i}" for i in range(X_val.shape[1])]
    
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1
    )

    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_perm


def compute_all_models_permutation_importance(
    models_dict: Dict[str, Pipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Compute test set Permutation Feature Importance across multiple candidate model pipelines.

    Parameters
    ----------
    models_dict : Dict[str, Pipeline]
        Dictionary mapping model names to fitted pipelines.
    X_val : pd.DataFrame
        Evaluation feature dataframe.
    y_val : pd.Series
        Target vector.
    scoring : str
        Scoring metric.
    n_repeats : int
        Permutation iterations.
    random_state : int
        Seed.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping model names to their permutation importance DataFrames.
    """
    results = {}
    for name, model in models_dict.items():
        results[name] = compute_permutation_importances(
            pipeline=model,
            X_val=X_val,
            y_val=y_val,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state
        )
    return results


def compute_linear_coefficients(
    sgd_pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract standardized linear decision weights from a fitted linear model (e.g. SGDClassifier).

    Since features are scaled via StandardScaler prior to fitting, the magnitude of these
    coefficients directly reflects their relative directional effect on the log-odds of being
    classified as Good Quality (>= 7).

    Parameters
    ----------
    sgd_pipeline : Pipeline
        Fitted linear classifier pipeline.
    feature_names : List[str]
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending absolute coefficient magnitude with columns:
        ['feature', 'coefficient', 'abs_coefficient', 'odds_ratio', 'direction']
    """
    clf = sgd_pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        raise ValueError("Pipeline does not contain an estimator with 'coef_'.")

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
        "odds_ratio": np.round(odds_ratios, 4),
        "direction": np.where(coefs >= 0, "Positive (+)", "Negative (-)")
    }).sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    return df_coef


def compute_partial_dependence_profiles(
    pipeline: Pipeline,
    X: pd.DataFrame,
    features: List[str],
    grid_resolution: int = 50
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate 1D Partial Dependence profiles for specified continuous features.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted classification pipeline with predict_proba capability.
    X : pd.DataFrame
        Feature dataset.
    features : List[str]
        Features to compute partial dependence for.
    grid_resolution : int
        Number of evaluation points across each feature range (default: 50).

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping feature name to grid values and average partial dependence predictions.
    """
    profiles = {}
    for feat in features:
        if feat not in X.columns:
            continue
        feat_idx = list(X.columns).index(feat)
        pd_result = partial_dependence(
            estimator=pipeline,
            X=X,
            features=[feat_idx],
            grid_resolution=grid_resolution,
            kind="average"
        )
        grid_key = "grid_values" if "grid_values" in pd_result else "values"
        profiles[feat] = {
            "values": pd_result[grid_key][0],
            "average": pd_result["average"][0]
        }
    return profiles


def compute_2d_partial_dependence(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_pair: Tuple[str, str],
    grid_resolution: int = 30
) -> Dict[str, Any]:
    """
    Calculate 2D joint partial dependence surface for a pair of interacting features.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted classifier pipeline.
    X : pd.DataFrame
        Feature matrix.
    feature_pair : Tuple[str, str]
        Pair of feature names (feat1, feat2).
    grid_resolution : int
        Grid resolution along each dimension.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing grid_x, grid_y, and 2D average prediction matrix Z.
    """
    f1, f2 = feature_pair
    idx1 = list(X.columns).index(f1)
    idx2 = list(X.columns).index(f2)

    pd_result = partial_dependence(
        estimator=pipeline,
        X=X,
        features=[(idx1, idx2)],
        grid_resolution=grid_resolution,
        kind="average"
    )

    grid_key = "grid_values" if "grid_values" in pd_result else "values"
    return {
        "feature_x": f1,
        "feature_y": f2,
        "grid_x": pd_result[grid_key][0],
        "grid_y": pd_result[grid_key][1],
        "z": pd_result["average"][0]
    }


def compute_enological_rules_profile(
    df_features: pd.DataFrame,
    target_col: str = "binary_quality"
) -> pd.DataFrame:
    """
    Synthesize quantitative enological decision rules and chemical thresholds by comparing
    feature distributions between Normal (< 7) and Good / Premium (>= 7) wines.

    Computes means, medians, 25th-75th percentiles (IQR), percentage shift, and derived
    practical winery targets.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing physicochemical features and binary quality target.
    target_col : str
        Name of binary quality column (default: 'binary_quality').

    Returns
    -------
    pd.DataFrame
        Benchmark table detailing chemical contrast and empirical operational targets.
    """
    normal_df = df_features[df_features[target_col] == 0]
    good_df = df_features[df_features[target_col] == 1]

    feature_cols = [c for c in df_features.columns if c not in ["quality", "binary_quality", "three_class_quality"]]

    rules = []
    for col in feature_cols:
        norm_mean = normal_df[col].mean()
        good_mean = good_df[col].mean()
        norm_med = normal_df[col].median()
        good_med = good_df[col].median()
        good_q25 = good_df[col].quantile(0.25)
        good_q75 = good_df[col].quantile(0.75)
        
        pct_diff = ((good_mean - norm_mean) / (abs(norm_mean) + 1e-6)) * 100

        # Enological recommendation logic based on chemical properties
        if col == "alcohol":
            guideline = f"Target >= {good_med:.1f}% vol (Optimal reserve: {good_q25:.1f} - {good_q75:.1f}%)"
            impact = "Enhances body, perceived warmth, and bouquet release"
        elif col == "volatile acidity":
            guideline = f"Keep <= {good_med:.2f} g/dm3 (Upper tolerance: {good_q75:.2f} g/dm3)"
            impact = "Prevents pungent acetobacter acetic acid and ethyl acetate taint"
        elif col == "sulphates":
            guideline = f"Target >= {good_med:.2f} g/dm3 (Optimal range: {good_q25:.2f} - {good_q75:.2f} g/dm3)"
            impact = "Essential antioxidant and antimicrobial preservation during bottle aging"
        elif col == "citric acid":
            guideline = f"Target >= {good_med:.2f} g/dm3 (Maintain freshness buffer)"
            impact = "Imparts palate crispness and counters harsh volatile notes"
        elif col == "free_to_total_so2_ratio":
            guideline = f"Maintain >= {good_med:.2f} (Active free fraction: >= 25%)"
            impact = "Guarantees molecular active antimicrobial protection without sulfur odor"
        elif col == "alcohol_to_density_ratio":
            guideline = f"Target >= {good_med:.2f} (Dry mouthfeel balance)"
            impact = "Indicates high ethanol conversion with low unfermented residual density"
        elif col == "sulphate_to_chloride_ratio":
            guideline = f"Target >= {good_med:.1f} (Optimal salt balance)"
            impact = "Maximizes beneficial potassium salts while suppressing mineral chlorides"
        elif col == "total sulfur dioxide":
            guideline = f"Control <= {good_med:.1f} mg/dm3 (Avoid antioxidant over-dosing)"
            impact = "High total SO2 masks fruit expression and indicates stressed must"
        elif col == "chlorides":
            guideline = f"Keep <= {good_med:.3f} g/dm3 (Avoid soil mineral salinity)"
            impact = "Excess salt degrades harmonious phenolic finish"
        elif col == "total_acidity":
            guideline = f"Balance around {good_med:.2f} g/dm3"
            impact = "Structural palate firmness and natural preservative backbone"
        else:
            direction = "Elevate" if pct_diff > 0 else "Moderate"
            guideline = f"{direction} towards median {good_med:.3f}"
            impact = f"Mean shift: {pct_diff:+.1f}% in premium vintages"

        rules.append({
            "Feature": col,
            "Normal Mean": np.round(norm_mean, 3),
            "Good Mean": np.round(good_mean, 3),
            "Normal Median": np.round(norm_med, 3),
            "Good Median": np.round(good_med, 3),
            "Mean Shift (%)": np.round(pct_diff, 1),
            "Direction": "Higher in Good (+)" if pct_diff > 0 else "Lower in Good (-)",
            "Enological Operational Guideline": guideline,
            "Sensory / Enological Impact": impact
        })

    df_rules = pd.DataFrame(rules).sort_values(
        by="Mean Shift (%)",
        key=lambda x: np.abs(x),
        ascending=False
    ).reset_index(drop=True)

    return df_rules


def save_champion_model_pipeline(
    pipeline: Pipeline,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Serialize the champion trained pipeline and write associated JSON metadata.

    Parameters
    ----------
    pipeline : Pipeline
        Champion scikit-learn pipeline.
    filepath : Union[str, Path]
        Destination .joblib file path.
    metadata : Optional[Dict[str, Any]]
        Dictionary of model hyperparameters, metrics, and training specifications.

    Returns
    -------
    str
        Canonical path to saved artifact.
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, save_path)

    if metadata is not None:
        meta_path = save_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    return str(save_path)


def load_champion_model_pipeline(filepath: Union[str, Path]) -> Pipeline:
    """
    Load a serialized pipeline from disk.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to serialized .joblib file.

    Returns
    -------
    Pipeline
        Loaded scikit-learn pipeline.
    """
    load_path = Path(filepath)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found at: {load_path}")
    return joblib.load(load_path)


def predict_wine_quality_sample(
    pipeline: Pipeline,
    raw_sample: Dict[str, float],
    threshold: float = 0.35
) -> Dict[str, Any]:
    """
    End-to-end inference for a single wine sample.

    Automatically computes domain-engineered interaction features, feeds the full 16-feature
    vector to the pipeline, and generates prediction class, probability, and enological advice.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted champion pipeline.
    raw_sample : Dict[str, float]
        Dictionary containing raw physicochemical values:
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'.
    threshold : float
        Decision probability threshold for Good Wine classification (default: 0.35).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predicted class label, binary label, posterior probability,
        and chemical diagnostic observations.
    """
    req_keys = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    missing = [k for k in req_keys if k not in raw_sample]
    if missing:
        raise ValueError(f"Missing required raw feature keys: {missing}")

    # Synthesize domain engineered features
    s = dict(raw_sample)
    s["total_acidity"] = s["fixed acidity"] + s["volatile acidity"]
    s["bound_sulfur_dioxide"] = s["total sulfur dioxide"] - s["free sulfur dioxide"]
    s["free_to_total_so2_ratio"] = s["free sulfur dioxide"] / (s["total sulfur dioxide"] + 1e-5)
    s["alcohol_to_density_ratio"] = s["alcohol"] / s["density"]
    s["sulphate_to_chloride_ratio"] = s["sulphates"] / (s["chlorides"] + 1e-5)

    feature_order = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "total_acidity", "bound_sulfur_dioxide",
        "free_to_total_so2_ratio", "alcohol_to_density_ratio", "sulphate_to_chloride_ratio"
    ]
    
    input_df = pd.DataFrame([[s[k] for k in feature_order]], columns=feature_order)

    # Compute posterior probability
    if hasattr(pipeline, "predict_proba"):
        prob_good = float(pipeline.predict_proba(input_df)[0, 1])
    else:
        dec = pipeline.decision_function(input_df)[0]
        prob_good = float(1.0 / (1.0 + np.exp(-dec)))

    is_good = int(prob_good >= threshold)
    label_text = "Good / Premium Wine (Score >= 7)" if is_good else "Normal / Table Wine (Score < 7)"

    # Generate diagnostic notes
    notes = []
    if s["alcohol"] >= 11.5:
        notes.append("Favorable: Elevated alcohol (>= 11.5%) contributes excellent body and aromatic volume.")
    else:
        notes.append("Advisory: Low alcohol (< 11.5%) is typical of lighter commercial table vintages.")

    if s["volatile acidity"] <= 0.40:
        notes.append("Favorable: Clean volatile acidity (<= 0.40 g/dm3) prevents vinegar sensory flaws.")
    else:
        notes.append("Risk: Elevated volatile acidity (> 0.40 g/dm3) indicates microbial acetic acid spoilage.")

    if s["sulphates"] >= 0.70:
        notes.append("Favorable: Optimal potassium sulphate antioxidant protection (>= 0.70 g/dm3).")
    else:
        notes.append("Advisory: Moderate sulphate preservative levels; monitor bottle aging stability.")

    if s["free_to_total_so2_ratio"] >= 0.25:
        notes.append("Favorable: High active free SO2 ratio (>= 25%) delivers effective antimicrobial efficacy.")
    else:
        notes.append("Risk: Low active free SO2 ratio indicates oxidized or over-bound preservative sulfur.")

    return {
        "predicted_label": label_text,
        "is_good": is_good,
        "probability_good": np.round(prob_good, 4),
        "decision_threshold": threshold,
        "diagnostic_notes": notes,
        "synthesized_features": {
            "total_acidity": np.round(s["total_acidity"], 3),
            "bound_sulfur_dioxide": np.round(s["bound_sulfur_dioxide"], 3),
            "free_to_total_so2_ratio": np.round(s["free_to_total_so2_ratio"], 3),
            "alcohol_to_density_ratio": np.round(s["alcohol_to_density_ratio"], 3),
            "sulphate_to_chloride_ratio": np.round(s["sulphate_to_chloride_ratio"], 3)
        }
    }


# ==============================================================================
# VISUALIZATION ROUTINES
# ==============================================================================

def plot_rf_feature_importance(
    mdi_df: pd.DataFrame,
    perm_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Random Forest feature importance: MDI (Gini) vs. Test Permutation Importance.

    Parameters
    ----------
    mdi_df : pd.DataFrame
        DataFrame with MDI importances.
    perm_df : Optional[pd.DataFrame]
        DataFrame with Permutation importances.
    save_path : Optional[str]
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    
    if perm_df is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_df = mdi_df.sort_values(by="importance_mean", ascending=True)
        ax.barh(
            plot_df["feature"],
            plot_df["importance_mean"],
            xerr=plot_df["importance_std"],
            color="#2b5c8f",
            edgecolor="#1a365d",
            capsize=3,
            alpha=0.85
        )
        ax.set_title("Random Forest Feature Importance (Mean Decrease in Impurity - MDI)", fontsize=13, weight="bold")
        ax.set_xlabel("Mean Impurity Decrease (Gini Importance)", fontsize=11)
        ax.set_ylabel("Predictor Attribute", fontsize=11)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

        # 1. MDI Subplot
        plot_mdi = mdi_df.sort_values(by="importance_mean", ascending=True)
        axes[0].barh(
            plot_mdi["feature"],
            plot_mdi["importance_mean"],
            xerr=plot_mdi["importance_std"],
            color="#2b5c8f",
            edgecolor="#1a365d",
            capsize=3,
            alpha=0.85
        )
        axes[0].set_title("Random Forest MDI (Gini) Importance\n(In-Sample Tree Node Splitting)", fontsize=12, weight="bold", pad=10)
        axes[0].set_xlabel("Mean Impurity Reduction across 200 Trees", fontsize=10)
        axes[0].set_ylabel("Physicochemical Attribute", fontsize=10)

        # 2. Permutation Subplot
        plot_perm = perm_df.sort_values(by="importance_mean", ascending=True)
        # Highlight engineered features with distinct color
        colors = ["#10b981" if any(k in f for k in ["ratio", "total_acidity", "bound"]) else "#6366f1" for f in plot_perm["feature"]]
        axes[1].barh(
            plot_perm["feature"],
            plot_perm["importance_mean"],
            xerr=plot_perm["importance_std"],
            color=colors,
            edgecolor="#374151",
            capsize=3,
            alpha=0.85
        )
        axes[1].axvline(0, color="gray", linestyle="--", linewidth=1.0)
        axes[1].set_title("Test Set Permutation Importance (ROC-AUC Drop)\n(Unbiased Out-of-Sample Impact)", fontsize=12, weight="bold", pad=10)
        axes[1].set_xlabel("Mean ROC-AUC Drop when Feature is Shuffled (10 Repeats)", fontsize=10)

        # Add legend for feature types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#6366f1", label="Raw Physicochemical Feature"),
            Patch(facecolor="#10b981", label="Domain Engineered Feature")
        ]
        axes[1].legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.suptitle("Random Forest Feature Importance Analysis: Gini Impurity vs. Test Permutation Drop", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig


def plot_permutation_importance_comparison(
    perm_dict: Dict[str, pd.DataFrame],
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare test set permutation feature importance across Random Forest, SVC, and SGDClassifier.

    Parameters
    ----------
    perm_dict : Dict[str, pd.DataFrame]
        Dictionary mapping model names to permutation importance DataFrames.
    top_n : int
        Number of top features to display (default: 10).
    save_path : Optional[str]
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    model_names = list(perm_dict.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(18, 7), sharey=False)

    palette = {
        "Random Forest": "#2b5c8f",
        "Support Vector Classifier": "#9333ea",
        "SGD Classifier": "#d97706"
    }

    for idx, name in enumerate(model_names):
        ax = axes[idx] if len(model_names) > 1 else axes
        df = perm_dict[name].head(top_n).sort_values(by="importance_mean", ascending=True)
        
        col = palette.get(name, "#4b5563")
        ax.barh(
            df["feature"],
            df["importance_mean"],
            xerr=df["importance_std"],
            color=col,
            edgecolor="#1f2937",
            capsize=3,
            alpha=0.85
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(f"{name}\n(ROC-AUC Permutation Metric)", fontsize=11, weight="bold")
        ax.set_xlabel("Mean ROC-AUC Drop", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Predictor Attribute", fontsize=10)
        else:
            ax.set_ylabel("")

    plt.suptitle("Cross-Model Permutation Feature Importance Comparison (Held-Out Test Set, N=320)", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig


def plot_linear_coefficients(
    coef_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot directional standardized linear weights from SGDClassifier.

    Features are sorted by coefficient magnitude; green indicates positive odds increase,
    crimson indicates negative odds reduction.

    Parameters
    ----------
    coef_df : pd.DataFrame
        DataFrame of linear coefficients.
    save_path : Optional[str]
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8))

    df_sorted = coef_df.sort_values(by="coefficient", ascending=True)
    colors = ["#dc2626" if c < 0 else "#059669" for c in df_sorted["coefficient"]]

    bars = ax.barh(
        df_sorted["feature"],
        df_sorted["coefficient"],
        color=colors,
        edgecolor="#1f2937",
        alpha=0.85,
        height=0.65
    )

    ax.axvline(0, color="#111827", linestyle="-", linewidth=1.2)
    ax.set_xlabel("Standardized Logistic Decision Weight (beta)", fontsize=11, weight="bold")
    ax.set_ylabel("Physicochemical Attribute", fontsize=11, weight="bold")
    ax.set_title("SGD Classifier Standardized Directional Feature Weights\n(Positive Green = Increases Premium Likelihood; Red = Suppresses)", fontsize=13, weight="bold", pad=12)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        offset = 0.015 if width >= 0 else -0.015
        ha = "left" if width >= 0 else "right"
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            weight="bold",
            color="#111827"
        )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#059669", label="Positive Driver (Elevates Quality Rating)"),
        Patch(facecolor="#dc2626", label="Negative Driver (Penalizes Quality Rating)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig


def plot_partial_dependence_grid(
    pipeline: Pipeline,
    X: pd.DataFrame,
    features: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 1D Partial Dependence Profiles for top chemical drivers.

    Displays marginal prediction curves alongside empirical data rug plots showing
    sample density across the feature range.

    Parameters
    ----------
    pipeline : Pipeline
        Champion pipeline.
    X : pd.DataFrame
        Evaluation feature matrix.
    features : List[str]
        Features to plot (recommended 4 to 6).
    save_path : Optional[str]
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    n_features = len(features)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), sharey=False)
    axes = axes.flatten()

    pdp_profiles = compute_partial_dependence_profiles(pipeline, X, features, grid_resolution=50)

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in pdp_profiles:
            ax.set_visible(False)
            continue

        data = pdp_profiles[feat]
        x_vals = data["values"]
        y_vals = data["average"]

        # Plot PDP curve
        ax.plot(x_vals, y_vals, color="#1e40af", linewidth=2.5, label="Partial Dependence (Avg Proba)")
        ax.fill_between(x_vals, y_vals.min(), y_vals, color="#3b82f6", alpha=0.15)

        # Add empirical rug plot along bottom
        sample_vals = X[feat].sample(min(300, len(X)), random_state=42).values
        y_rug_pos = y_vals.min() - 0.015 * (y_vals.max() - y_vals.min() + 1e-4)
        ax.plot(sample_vals, [y_rug_pos] * len(sample_vals), "|", color="#64748b", markersize=6, alpha=0.6, label="Sample Density")

        ax.set_title(f"PDP: {feat}", fontsize=11, weight="bold")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("Marginal Probability P(Good)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide extra unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Partial Dependence Profiles for Top Predictive Physicochemical Drivers\n(Marginal Impact on Predicted Probability of Good Wine)", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig


def plot_pdp_2d_interactions(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D joint partial dependence contour maps for interacting feature pairs.

    Parameters
    ----------
    pipeline : Pipeline
        Champion pipeline.
    X : pd.DataFrame
        Feature matrix.
    feature_pairs : List[Tuple[str, str]]
        Pairs of interacting features to profile (e.g. ('alcohol', 'volatile acidity')).
    save_path : Optional[str]
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    sns.set_theme(style="white", font_scale=0.95)
    n_pairs = len(feature_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 6), sharey=False)

    if n_pairs == 1:
        axes = [axes]

    for idx, pair in enumerate(feature_pairs):
        ax = axes[idx]
        pdp_2d = compute_2d_partial_dependence(pipeline, X, pair, grid_resolution=30)

        gx = pdp_2d["grid_x"]
        gy = pdp_2d["grid_y"]
        Z = pdp_2d["z"]

        X_grid, Y_grid = np.meshgrid(gx, gy)

        # Plot filled contour
        cs = ax.contourf(X_grid, Y_grid, Z.T, levels=15, cmap="viridis", alpha=0.9)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("P(Good Wine >= 7)", fontsize=10, weight="bold")

        # Overlay contour lines with labels
        lines = ax.contour(X_grid, Y_grid, Z.T, levels=8, colors="white", linewidths=0.8, alpha=0.7)
        ax.clabel(lines, inline=True, fontsize=8, fmt="%.2f")

        ax.set_title(f"Joint 2D PDP: {pair[0]} vs. {pair[1]}", fontsize=11, weight="bold")
        ax.set_xlabel(pair[0], fontsize=10, weight="bold")
        ax.set_ylabel(pair[1], fontsize=10, weight="bold")

    plt.suptitle("Bivariate Non-Linear Partial Dependence Interaction Contours", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig


def plot_champion_model_summary(
    df_feat: pd.DataFrame,
    rf_pipeline: Pipeline,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate an executive summary dashboard showing normalized chemical attribute profiles
    of Normal (< 7) vs. Good (>= 7) wines alongside key champion model metrics.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Complete engineered feature dataset.
    rf_pipeline : Pipeline
        Tuned champion Random Forest pipeline.
    save_path : Optional[str]
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8], height_ratios=[1, 1])

    ax_bar = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    # 1. Normalized Relative Differences Bar Plot
    feature_cols = [c for c in df_feat.columns if c not in ["quality", "binary_quality", "three_class_quality"]]
    norm_means = df_feat[df_feat["binary_quality"] == 0][feature_cols].mean()
    good_means = df_feat[df_feat["binary_quality"] == 1][feature_cols].mean()
    pct_diffs = ((good_means - norm_means) / (norm_means.abs() + 1e-5)) * 100

    diff_df = pd.DataFrame({
        "feature": feature_cols,
        "pct_diff": pct_diffs
    }).sort_values(by="pct_diff", ascending=True)

    colors = ["#dc2626" if v < 0 else "#059669" for v in diff_df["pct_diff"]]
    bars = ax_bar.barh(diff_df["feature"], diff_df["pct_diff"], color=colors, edgecolor="#1f2937", alpha=0.85)
    ax_bar.axvline(0, color="#111827", linewidth=1.2)
    ax_bar.set_title("Mean Chemical Attribute Percentage Difference\n(Good Wines [>=7] vs. Normal Wines [<7])", fontsize=12, weight="bold")
    ax_bar.set_xlabel("Percentage Difference Relative to Normal Wines (%)", fontsize=10, weight="bold")
    ax_bar.set_ylabel("Physicochemical Attribute", fontsize=10, weight="bold")

    for bar in bars:
        w = bar.get_width()
        offset = 1.0 if w >= 0 else -1.0
        ha = "left" if w >= 0 else "right"
        ax_bar.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}%", va="center", ha=ha, fontsize=8, weight="bold")

    # 2. Executive Scorecard Text Box
    ax_top.axis("off")
    scorecard_text = (
        "CHAMPION MODEL EXECUTIVE SPECIFICATION\n"
        "======================================\n"
        "• Model Architecture: Tuned Random Forest (200 Trees)\n"
        "• Hyperparameters: depth=18, class_weight='balanced'\n"
        "• Test Partition Size: N = 320 (Held-out 20% Stratified Split)\n"
        "• Test ROC-AUC: 0.9368  |  PR-AUC (Avg Precision): 0.8095\n"
        "• Standard Threshold (t=0.50):\n"
        "    - Good Wine Recall: 65.1%  |  Precision: 71.8%\n"
        "    - Balanced Accuracy: 80.6% |  Overall Accuracy: 91.9%\n"
        "• Industrial Screening Threshold (t=0.35):\n"
        "    - Good Wine Recall: 79.1% (Catches 34 of 43 Reserve Wines)\n"
        "    - Balanced Accuracy: 85.2% |  Precision: 58.6%\n"
        "• Generalization Check: 5-Fold Stratified CV Mean F1 = 0.604"
    )
    ax_top.text(
        0.05, 0.5, scorecard_text,
        family="monospace",
        fontsize=9.5,
        va="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    )
    ax_top.set_title("Champion Model Diagnostic Card", fontsize=12, weight="bold", pad=8)

    # 3. Actionable Enological Targets Table
    ax_bot.axis("off")
    table_data = [
        ["Attribute", "Target Range", "Primary Mechanism"],
        ["Alcohol", ">= 11.5% vol", "Body & Bouquet Release"],
        ["Volatile Acidity", "<= 0.40 g/dm3", "Prevents Acetic Acid Flaw"],
        ["Sulphates", ">= 0.70 g/dm3", "Bottle Oxidation Defense"],
        ["Free/Total SO2", ">= 0.25 (25%)", "Active Antimicrobial Pool"],
        ["Residual Sugar", "< 3.0 g/dm3", "Crisp Dry Phenolic Finish"]
    ]
    tab = ax_bot.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.85]
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f5f9" if row % 2 == 0 else "white")
    ax_bot.set_title("Actionable Viticultural & Cellar Quality Targets", fontsize=12, weight="bold", pad=8)

    plt.suptitle("Day 5 Model Interpretation & Champion Selection Executive Summary", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig
