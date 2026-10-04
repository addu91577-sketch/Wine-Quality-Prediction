import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

def load_wine_data(
    wine_type: Literal["red", "white", "both"] = "red",
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load the UCI Wine Quality dataset.

    Parameters:
    -----------
    wine_type : {'red', 'white', 'both'}, default='red'
        Type of wine data to load.
    data_dir : Path, optional
        Path to data folder containing CSV files.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing wine physicochemical attributes and quality scores.
    """
    base_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    red_path = base_dir / "winequality-red.csv"
    white_path = base_dir / "winequality-white.csv"

    if wine_type == "red":
        if not red_path.exists():
            raise FileNotFoundError(f"Red wine file not found at: {red_path}")
        df = pd.read_csv(red_path, sep=";")
        return df

    elif wine_type == "white":
        if not white_path.exists():
            raise FileNotFoundError(f"White wine file not found at: {white_path}")
        df = pd.read_csv(white_path, sep=";")
        return df

    elif wine_type == "both":
        if not red_path.exists() or not white_path.exists():
            raise FileNotFoundError(f"Missing wine files in: {base_dir}")
        red_df = pd.read_csv(red_path, sep=";")
        white_df = pd.read_csv(white_path, sep=";")
        red_df["wine_type"] = "red"
        white_df["wine_type"] = "white"
        combined_df = pd.concat([red_df, white_df], ignore_index=True)
        return combined_df

    else:
        raise ValueError(f"Invalid wine_type '{wine_type}'. Expected 'red', 'white', or 'both'.")


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect dataset structure: dimensions, null values, duplicates, and column types.

    Parameters:
    -----------
    df : pd.DataFrame
        Input wine dataset.

    Returns:
    --------
    dict
        Inspection metrics dictionary.
    """
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }
    return info


def get_class_distribution(
    df: pd.DataFrame,
    target_col: str = "quality"
) -> pd.DataFrame:
    """
    Calculate class frequency and percentage distribution of quality scores.

    Parameters:
    -----------
    df : pd.DataFrame
        Wine dataframe.
    target_col : str, default='quality'
        Target column name.

    Returns:
    --------
    pd.DataFrame
        Summary table sorted by quality score.
    """
    counts = df[target_col].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(2)
    dist_df = pd.DataFrame({
        "Quality Score": counts.index,
        "Sample Count": counts.values,
        "Percentage (%)": percentages.values,
        "Cumulative (%)": percentages.cumsum().round(2).values
    }).reset_index(drop=True)
    return dist_df


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "quality",
    save_path: Optional[str] = None,
    title: str = "Distribution of Wine Quality Scores"
) -> None:
    """
    Generate and display/save a bar plot of quality score distribution.

    Parameters:
    -----------
    df : pd.DataFrame
        Wine dataframe.
    target_col : str, default='quality'
        Target column name.
    save_path : str, optional
        Path to save figure.
    title : str
        Plot title.
    """
    plt.figure(figsize=(9, 5))
    palette = sns.color_palette("mako", n_colors=df[target_col].nunique())
    
    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Quality Score", fontsize=12, fontweight="medium")
    plt.ylabel("Number of Samples", fontsize=12, fontweight="medium")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with counts and percentages
    for p, count in zip(ax.patches, counts.values):
        percentage = f"{(count / total) * 100:.1f}%"
        ax.annotate(
            f"{count}\n({percentage})",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 4),
            textcoords="offset points"
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.close()
