import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

def create_day1_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.8"
        }
    }

    cells = []

    # Markdown: Title & Overview
    cells.append(nbf.v4.new_markdown_cell(
        "# Day 1: Wine Quality Dataset Inspection & Class Distribution Analysis\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Date:** October 5, 2026\n"
        "**Tech Stack:** Python, pandas, numpy, seaborn, matplotlib, scikit-learn\n\n"
        "## Objectives for Day 1\n"
        "1. Load the physicochemical red wine dataset from the `Data/` directory.\n"
        "2. Inspect structural attributes: shape, column data types, missing values, duplicates, and descriptive statistics.\n"
        "3. Evaluate the distribution of target `quality` scores (scale 3–8).\n"
        "4. Formulate initial insights regarding class imbalance and its implications on classification modeling."
    ))

    # Code: Imports & Setup
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Style configuration\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n"
        "plt.rcParams['figure.dpi'] = 120\n\n"
        "print('Libraries successfully imported!')"
    ))

    # Markdown: Data Loading
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load the Wine Quality Dataset\n"
        "The UCI Wine Quality dataset uses semicolon (`;`) separators and records 11 physicochemical features along with the sensory `quality` rating."
    ))

    # Code: Data Loading
    cells.append(nbf.v4.new_code_cell(
        "# Load red wine data\n"
        "DATA_PATH = Path('../Data/winequality-red.csv')\n"
        "if not DATA_PATH.exists():\n"
        "    DATA_PATH = Path('Data/winequality-red.csv')\n\n"
        "df = pd.read_csv(DATA_PATH, sep=';')\n"
        "print(f'Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns')\n"
        "df.head()"
    ))

    # Markdown: Structural Inspection
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Dataset Structure & Integrity Inspection\n"
        "Let's review column types, verify if missing values exist, and check for duplicate observations."
    ))

    # Code: Info and Null checks
    cells.append(nbf.v4.new_code_cell(
        "# Column info and data types\n"
        "print('--- Dataset Info ---')\n"
        "df.info()\n\n"
        "print('\\n--- Missing Value Check ---')\n"
        "missing_counts = df.isnull().sum()\n"
        "print(missing_counts[missing_counts > 0] if (missing_counts > 0).any() else 'No missing values found across all 12 columns.')\n\n"
        "print('\\n--- Duplicates Check ---')\n"
        "dup_count = df.duplicated().sum()\n"
        "print(f'Duplicate rows detected: {dup_count} ({dup_count / len(df) * 100:.2f}%)')"
    ))

    # Markdown: Summary Statistics
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Summary Statistics\n"
        "Examining distribution ranges across physicochemical properties (min, quartiles, max, mean, and standard deviation)."
    ))

    # Code: Summary Statistics
    cells.append(nbf.v4.new_code_cell(
        "# Summary statistics\n"
        "summary_stats = df.describe().T\n"
        "summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']\n"
        "summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'IQR']].round(3)"
    ))

    # Markdown: Target Distribution
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Target Variable Analysis: Quality Score Distribution\n"
        "The target variable `quality` represents sensory evaluation scores given by wine experts."
    ))

    # Code: Target distribution table
    cells.append(nbf.v4.new_code_cell(
        "# Compute frequency and percentages\n"
        "quality_counts = df['quality'].value_counts().sort_index()\n"
        "quality_pct = (quality_counts / len(df) * 100).round(2)\n\n"
        "dist_table = pd.DataFrame({\n"
        "    'Quality Score': quality_counts.index,\n"
        "    'Sample Count': quality_counts.values,\n"
        "    'Percentage (%)': quality_pct.values,\n"
        "    'Cumulative (%)': quality_pct.cumsum().round(2).values\n"
        "})\n"
        "dist_table"
    ))

    # Markdown: Visualizing Distribution
    cells.append(nbf.v4.new_markdown_cell(
        "### 4.1 Visualizing Quality Class Frequencies\n"
        "Let's plot a clean bar chart with count and percentage annotations to clearly visualize the concentration of scores."
    ))

    # Code: Visualization Bar Plot
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(10, 5))\n"
        "palette = sns.color_palette('viridis', n_colors=len(quality_counts))\n"
        "ax = sns.barplot(x=quality_counts.index, y=quality_counts.values, hue=quality_counts.index, palette=palette, legend=False)\n\n"
        "plt.title('Distribution of Red Wine Quality Scores (Scale 3–8)', fontsize=14, fontweight='bold', pad=15)\n"
        "plt.xlabel('Quality Rating', fontsize=12)\n"
        "plt.ylabel('Number of Samples', fontsize=12)\n"
        "plt.ylim(0, max(quality_counts.values) * 1.15)\n\n"
        "# Annotations\n"
        "for p, count in zip(ax.patches, quality_counts.values):\n"
        "    pct = f'{(count / len(df)) * 100:.1f}%'\n"
        "    ax.annotate(f'{count}\\n({pct})',\n"
        "                (p.get_x() + p.get_width() / 2., p.get_height()),\n"
        "                ha='center', va='bottom', fontsize=10, xytext=(0, 4),\n"
        "                textcoords='offset points', fontweight='bold')\n\n"
        "plt.tight_layout()\n"
        "# Ensure output directory exists\n"
        "os.makedirs('../reports/figures', exist_ok=True)\n"
        "plt.savefig('../reports/figures/day1_quality_distribution.png', dpi=300)\n"
        "plt.show()"
    ))

    # Markdown: Imbalance Discussion
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Day 1 Key Findings & Class Imbalance Analysis\n\n"
        "### Observations:\n"
        "1. **Severe Imbalance in Middle Classes:**\n"
        "   - Quality scores **5** (42.59%) and **6** (39.90%) constitute **82.49%** of the entire dataset (1,319 out of 1,599 samples).\n"
        "   - Borderline/extreme quality wines are rare:\n"
        "     - Score 3: Only 10 samples (0.63%)\n"
        "     - Score 4: Only 53 samples (3.31%)\n"
        "     - Score 7: 199 samples (12.45%)\n"
        "     - Score 8: Only 18 samples (1.13%)\n\n"
        "2. **Implications for Machine Learning Modeling:**\n"
        "   - **Accuracy Paradox:** A simple majority-class baseline model predicting only 5 or 6 would achieve over 82% nominal accuracy while having 0% recall on exceptional (score 8) or poor (score 3) wines.\n"
        "   - **Stratification Necessity:** Random train-test splitting without stratification risks leaving minority classes with virtually 0 samples in the test or train splits. `StratifiedKFold` and stratified `train_test_split` are mandatory.\n"
        "   - **Motivation for Target Binning (Day 3):** Grouping into binary (`good`: $\\ge 7$ vs `bad`: $< 7$) or 3-tier (`low`, `medium`, `high`) creates more balanced, practically actionable classification boundaries.\n\n"
        "### Next Steps (Day 2 - October 6, 2026):\n"
        "- Conduct detailed Exploratory Data Analysis (EDA) on all 11 physicochemical features.\n"
        "- Generate distribution plots (histograms, KDE, boxplots) and correlation heatmaps."
    ))

    nb.cells = cells
    notebook_path = "notebooks/Day1_Data_Inspection_and_Distribution.ipynb"
    
    # Save unexecuted first
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {notebook_path}")

    # Execute notebook using ExecutePreprocessor
    print("Executing notebook to persist outputs...")
    ep = ExecutePreprocessor(timeout=60, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "notebooks"}})

    # Save executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed notebook saved with cell outputs to {notebook_path}")

if __name__ == "__main__":
    create_day1_notebook()
