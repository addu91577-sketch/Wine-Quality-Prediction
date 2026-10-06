"""
Feature engineering and target transformation routines for Wine Quality Prediction.

Provides methods for:
- Quality score binning (Binary and 3-Class) with distribution analytics.
- Enological domain-specific interaction feature generation.
- Stratified train-test dataset splitting with distribution validation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_binned_targets(
    df: pd.DataFrame,
    target_col: str = "quality",
    binary_threshold: int = 7
) -> pd.DataFrame:
    """
    Generate binary and 3-class target columns from discrete sensory quality scores.

    Binary Formulation:
        - 1 ('Good' / Premium): quality >= binary_threshold (typically >= 7)
        - 0 ('Normal' / Low): quality < binary_threshold (3 to 6)

    3-Class Formulation:
        - 'Low' (Poor): quality <= 4
        - 'Medium' (Average): quality in [5, 6]
        - 'High' (Good): quality >= 7

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the raw quality column.
    target_col : str
        Name of the sensory quality target column (default: 'quality').
    binary_threshold : int
        Score threshold for the binary 'good' designation (default: 7).

    Returns
    -------
    pd.DataFrame
        Dataframe copy with added 'binary_quality' and 'three_class_quality' columns.
    """
    df_out = df.copy()

    # Binary target: 1 = Good (7-8), 0 = Normal/Bad (3-6)
    df_out["binary_quality"] = (df_out[target_col] >= binary_threshold).astype(int)

    # 3-Class target
    def categorize_3class(score: int) -> str:
        if score <= 4:
            return "Low (3-4)"
        elif score <= 6:
            return "Medium (5-6)"
        else:
            return "High (7-8)"

    df_out["three_class_quality"] = df_out[target_col].apply(categorize_3class)
    return df_out


def compute_target_distribution_summary(
    df: pd.DataFrame,
    target_col: str = "quality"
) -> Dict[str, pd.DataFrame]:
    """
    Compute distribution frequencies and percentages for raw, binary, and 3-class targets.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'raw', 'binary', and 'three_class'.
    """
    total = len(df)
    
    # Raw distribution
    raw_counts = df[target_col].value_counts().sort_index()
    raw_df = pd.DataFrame({
        "Quality Score": raw_counts.index,
        "Count": raw_counts.values,
        "Percentage (%)": np.round((raw_counts.values / total) * 100, 2),
        "Cumulative (%)": np.round((raw_counts.values / total).cumsum() * 100, 2)
    })

    # Binary distribution
    bin_counts = df["binary_quality"].value_counts().sort_index()
    bin_labels = ["Normal/Low (< 7)", "Good / Premium (>= 7)"]
    bin_df = pd.DataFrame({
        "Binary Class": [bin_labels[i] for i in bin_counts.index],
        "Label Value": bin_counts.index,
        "Count": bin_counts.values,
        "Percentage (%)": np.round((bin_counts.values / total) * 100, 2)
    })

    # 3-Class distribution
    order = ["Low (3-4)", "Medium (5-6)", "High (7-8)"]
    three_counts = df["three_class_quality"].value_counts().reindex(order)
    three_df = pd.DataFrame({
        "Class Category": order,
        "Count": three_counts.values,
        "Percentage (%)": np.round((three_counts.values / total) * 100, 2)
    })

    return {
        "raw": raw_df,
        "binary": bin_df,
        "three_class": three_df
    }


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-informed enological features based on chemical relationships.

    Engineered features:
    1. total_acidity: fixed acidity + volatile acidity (overall acidic impact)
    2. bound_sulfur_dioxide: total sulfur dioxide - free sulfur dioxide (bound antioxidant fraction)
    3. free_to_total_so2_ratio: free SO2 / (total SO2 + 1e-5) (active preservation fraction)
    4. alcohol_to_density_ratio: alcohol / density (body and mass balance indicator)
    5. sulphate_to_chloride_ratio: sulphates / (chlorides + 1e-5) (preservative vs mineral salinity ratio)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with standard physicochemical columns.

    Returns
    -------
    pd.DataFrame
        Dataframe copy with added engineered features.
    """
    df_out = df.copy()

    # Total Acidity (Fixed + Volatile)
    if "fixed acidity" in df_out.columns and "volatile acidity" in df_out.columns:
        df_out["total_acidity"] = df_out["fixed acidity"] + df_out["volatile acidity"]

    # Bound Sulfur Dioxide
    if "total sulfur dioxide" in df_out.columns and "free sulfur dioxide" in df_out.columns:
        df_out["bound_sulfur_dioxide"] = df_out["total sulfur dioxide"] - df_out["free sulfur dioxide"]
        df_out["free_to_total_so2_ratio"] = df_out["free sulfur dioxide"] / (df_out["total sulfur dioxide"] + 1e-5)

    # Alcohol to Density Ratio
    if "alcohol" in df_out.columns and "density" in df_out.columns:
        df_out["alcohol_to_density_ratio"] = df_out["alcohol"] / df_out["density"]

    # Sulphate to Chloride Ratio
    if "sulphates" in df_out.columns and "chlorides" in df_out.columns:
        df_out["sulphate_to_chloride_ratio"] = df_out["sulphates"] / (df_out["chlorides"] + 1e-5)

    return df_out


def get_stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition feature matrix and target series into stratified train and test partitions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target array or series.
    test_size : float
        Proportion of dataset to allocate to the test partition (default: 0.2).
    random_state : int
        Seed for deterministic reproducibility (default: 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def verify_stratification_balance(
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Generate validation comparison showing class percentages in train and test splits.
    """
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    comp_df = pd.DataFrame({
        "Class": train_dist.index,
        "Train Count": train_counts.values,
        "Train Ratio (%)": np.round(train_dist.values, 2),
        "Test Count": test_counts.values,
        "Test Ratio (%)": np.round(test_dist.values, 2),
        "Absolute Difference (%)": np.round(np.abs(train_dist.values - test_dist.values), 4)
    })
    return comp_df
