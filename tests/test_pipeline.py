"""
Automated Pytest Suite: ML Pipeline Integrity & Feature Engineering Verification
Validates raw UCI dataset structure, stratified partitioning, target binarization,
and scikit-learn champion estimator consistency.
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import load_wine_data, inspect_data, get_class_distribution
from src.feature_engineering import (
    create_binned_targets,
    create_engineered_features,
    get_stratified_train_test_split
)
from src.model_interpretation import load_champion_model_pipeline
from src.production_service import DEFAULT_MODEL_PATH, FULL_FEATURE_ORDER


DATA_PATH = Path("Data/winequality-red.csv")


def test_uci_dataset_file_exists():
    """Verify raw winequality-red.csv dataset is present in Data directory."""
    assert DATA_PATH.exists(), f"UCI dataset missing at {DATA_PATH}"


def test_data_loader_structure():
    """Verify UCI dataset dimensions (1599x12) and zero missing values."""
    df = load_wine_data(wine_type="red")
    assert len(df) == 1599
    assert df.shape[1] == 12
    assert df.isnull().sum().sum() == 0
    assert "quality" in df.columns


def test_class_distribution_integrity():
    """Verify quality score distribution matches Cortez et al. standard."""
    df = load_wine_data(wine_type="red")
    dist = get_class_distribution(df)
    assert len(dist) == 6  # Ratings 3, 4, 5, 6, 7, 8
    # Scores 5 and 6 comprise over 80% of samples
    counts = df["quality"].value_counts()
    dominant_pct = (counts[5] + counts[6]) / len(df)
    assert dominant_pct > 0.80


def test_target_binning():
    """Verify binary quality threshold: quality >= 7 is Good (1), else Normal (0)."""
    df = load_wine_data(wine_type="red")
    df_binned = create_binned_targets(df)
    assert "binary_quality" in df_binned.columns
    # Good wines count (scores 7 and 8) = 199 + 18 = 217
    good_wines = (df_binned["quality"] >= 7).sum()
    assert good_wines == 217
    assert (df_binned["binary_quality"] == 1).sum() == 217
    assert (df_binned["binary_quality"] == 0).sum() == 1599 - 217


def test_engineered_features_creation():
    """Verify all 5 chemical interaction features are added correctly."""
    df = load_wine_data(wine_type="red")
    df_eng = create_engineered_features(df)
    for col in [
        "total_acidity", "bound_sulfur_dioxide", "free_to_total_so2_ratio",
        "alcohol_to_density_ratio", "sulphate_to_chloride_ratio"
    ]:
        assert col in df_eng.columns
        assert not df_eng[col].isnull().any()
        assert not np.isinf(df_eng[col]).any()


def test_stratified_train_test_split():
    """Verify 80/20 train/test split preserves class ratio."""
    df = load_wine_data(wine_type="red")
    df_binned = create_binned_targets(df)
    df_eng = create_engineered_features(df_binned)
    feature_cols = [c for c in df_eng.columns if c not in ["quality", "binary_quality", "three_class_quality"]]
    X = df_eng[feature_cols]
    y = df_eng["binary_quality"]
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    assert len(X_train) == 1279
    assert len(X_test) == 320
    # Minority ratio ~13.57%
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.01


def test_champion_model_scoring_on_test_slice():
    """Verify champion pipeline produces expected output shape on feature DataFrame."""
    pipeline = load_champion_model_pipeline(DEFAULT_MODEL_PATH)
    df = load_wine_data(wine_type="red")
    df_eng = create_engineered_features(df)[FULL_FEATURE_ORDER]
    sample_batch = df_eng.iloc[:10]
    preds = pipeline.predict(sample_batch)
    probs = pipeline.predict_proba(sample_batch)
    assert preds.shape == (10,)
    assert probs.shape == (10, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
