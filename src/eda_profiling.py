import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_feature_columns(df: pd.DataFrame, target_col: str = "quality") -> List[str]:
    """Return all numerical predictor column names excluding the target column."""
    return [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]


def compute_univariate_stats(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute comprehensive univariate descriptive statistics including skewness and kurtosis.

    Parameters:
    -----------
    df : pd.DataFrame
        Wine quality dataset.
    features : list of str, optional
        Subset of features to profile. Defaults to all predictor features.

    Returns:
    --------
    pd.DataFrame
        Summary statistics table indexed by feature name.
    """
    if features is None:
        features = get_feature_columns(df)

    stats_list = []
    for col in features:
        series = df[col]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        stats_list.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 3),
            "Std": round(float(series.std()), 3),
            "Min": round(float(series.min()), 3),
            "25%": round(float(q25), 3),
            "Median (50%)": round(float(series.median()), 3),
            "75%": round(float(q75), 3),
            "Max": round(float(series.max()), 3),
            "IQR": round(float(iqr), 3),
            "Skewness": round(float(series.skew()), 3),
            "Kurtosis": round(float(series.kurtosis()), 3)
        })

    return pd.DataFrame(stats_list).set_index("Feature")


def detect_outliers_iqr(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Identify outliers using Tukey's Interquartile Range (IQR) method.

    Parameters:
    -----------
    df : pd.DataFrame
        Wine quality dataset.
    features : list of str, optional
        Features to inspect for outliers.
    factor : float, default=1.5
        IQR multiplier for fence thresholds.

    Returns:
    --------
    pd.DataFrame
        Summary table detailing outlier thresholds, counts, and percentages.
    """
    if features is None:
        features = get_feature_columns(df)

    outlier_records = []
    total_rows = len(df)

    for col in features:
        series = df[col]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_fence = q25 - factor * iqr
        upper_fence = q75 + factor * iqr

        lower_outliers = (series < lower_fence).sum()
        upper_outliers = (series > upper_fence).sum()
        total_outliers = lower_outliers + upper_outliers

        outlier_records.append({
            "Feature": col,
            "Q1 (25%)": round(float(q25), 3),
            "Q3 (75%)": round(float(q75), 3),
            "IQR": round(float(iqr), 3),
            "Lower Fence": round(float(lower_fence), 3),
            "Upper Fence": round(float(upper_fence), 3),
            "Low Outliers": int(lower_outliers),
            "High Outliers": int(upper_outliers),
            "Total Outliers": int(total_outliers),
            "Outlier Pct (%)": round(float(total_outliers / total_rows * 100), 2)
        })

    return pd.DataFrame(outlier_records).set_index("Feature")


def compute_target_correlations(
    df: pd.DataFrame,
    target_col: str = "quality"
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman rank correlations between all features and the target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Wine quality dataset.
    target_col : str, default='quality'
        Target column.

    Returns:
    --------
    pd.DataFrame
        Table of correlations sorted by descending absolute Pearson correlation.
    """
    features = get_feature_columns(df, target_col=target_col)
    records = []

    for col in features:
        pearson_val = df[col].corr(df[target_col], method="pearson")
        spearman_val = df[col].corr(df[target_col], method="spearman")
        records.append({
            "Feature": col,
            "Pearson Correlation": round(float(pearson_val), 4),
            "Spearman Correlation": round(float(spearman_val), 4),
            "Abs Pearson": round(abs(float(pearson_val)), 4),
            "Direction": "Positive" if pearson_val > 0 else "Negative"
        })

    corr_df = pd.DataFrame(records).sort_values(by="Abs Pearson", ascending=False).reset_index(drop=True)
    return corr_df


def plot_univariate_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot grid of histograms with KDE curves for all physicochemical features.
    """
    if features is None:
        features = get_feature_columns(df)

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        ax = axes[idx]
        skew_val = df[col].skew()
        sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", edgecolor="#0e4b75", alpha=0.6)
        
        # Add mean and median vertical indicators
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.2f}")

        ax.set_title(f"{col}\n(Skew: {skew_val:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Univariate Feature Distributions & Skewness Analysis (Red Wine)", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Distribution plot saved to: {save_path}")
    plt.close()


def plot_feature_boxplots(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot grid of standalone boxplots to identify spread and IQR outliers.
    """
    if features is None:
        features = get_feature_columns(df)

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.8 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        ax = axes[idx]
        sns.boxplot(
            x=df[col],
            ax=ax,
            color="#4a90e2",
            flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#e74c3c", "alpha": 0.6}
        )
        ax.set_title(f"Spread & Outliers: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boxplots for Physicochemical Features (IQR Outlier Assessment)", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Boxplot saved to: {save_path}")
    plt.close()


def plot_features_vs_quality(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "quality",
    save_path: Optional[str] = None
) -> None:
    """
    Plot boxplots of each physicochemical feature grouped across quality ratings (3 to 8).
    """
    if features is None:
        features = get_feature_columns(df, target_col=target_col)

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    palette = sns.color_palette("coolwarm", n_colors=df[target_col].nunique())

    for idx, col in enumerate(features):
        ax = axes[idx]
        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            ax=ax,
            palette=palette,
            hue=target_col,
            legend=False,
            fliersize=3
        )
        ax.set_title(f"{col} vs Quality", fontsize=11, fontweight="bold")
        ax.set_xlabel("Quality Score", fontsize=10)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Bivariate Analysis: Physicochemical Features Across Quality Scores", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bivariate plot saved to: {save_path}")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    method: str = "pearson"
) -> None:
    """
    Generate an annotated correlation heatmap for all numeric features and quality.
    """
    plt.figure(figsize=(11, 9))
    corr_matrix = df.corr(method=method)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation Coefficient"},
        annot_kws={"size": 9}
    )

    plt.title(f"Correlation Heatmap ({method.capitalize()}) - Red Wine Dataset", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")
    plt.close()


def plot_top_correlations_bar(
    df: pd.DataFrame,
    target_col: str = "quality",
    save_path: Optional[str] = None
) -> None:
    """
    Generate a horizontal bar plot showing features ranked by their correlation with the target.
    """
    corr_series = df.corr(numeric_only=True)[target_col].drop(target_col).sort_values()

    plt.figure(figsize=(10, 6))
    colors = ["#e74c3c" if val < 0 else "#2ecc71" for val in corr_series.values]

    y_positions = np.arange(len(corr_series))
    bars = plt.barh(y_positions, corr_series.values, color=colors, edgecolor="black", alpha=0.85, height=0.65)
    plt.yticks(y_positions, corr_series.index, fontsize=11)
    plt.xlabel(f"Correlation with {target_col.capitalize()}", fontsize=12, fontweight="medium")
    plt.title(f"Feature Correlations with Wine Quality (Red Wine)", fontsize=14, fontweight="bold", pad=15)
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Label bars with exact correlation value
    for bar, val in zip(bars, corr_series.values):
        offset = 0.015 if val >= 0 else -0.015
        ha = "left" if val >= 0 else "right"
        plt.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold"
        )

    plt.xlim(min(corr_series.values) - 0.08, max(corr_series.values) + 0.08)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar correlation plot saved to: {save_path}")
    plt.close()
