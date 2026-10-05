import os
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def create_day2_notebook():
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

    # Markdown: Title & Objectives
    cells.append(nbf.v4.new_markdown_cell(
        "# Day 2: Exploratory Data Analysis & Physicochemical Feature Profiling\n"
        "**Project:** Wine Quality Prediction (UCI Machine Learning Repository)\n"
        "**Date:** October 6, 2026\n"
        "**Tech Stack:** Python, pandas, numpy, seaborn, matplotlib, scikit-learn\n\n"
        "## Objectives for Day 2\n"
        "1. **Univariate Profiling:** Perform comprehensive statistical profiling (central tendency, dispersion, skewness, and kurtosis) across all 11 physicochemical features.\n"
        "2. **Distribution Modeling:** Visualize univariate distributions via histograms and Kernel Density Estimates (KDE) to identify normality vs. heavy skew.\n"
        "3. **Outlier Detection:** Quantify anomalous values across each attribute using Tukey's Interquartile Range (IQR) method and inspect spread via boxplots.\n"
        "4. **Bivariate Target Analysis:** Investigate how individual chemical properties shift across sensory quality ratings (scale 3–8).\n"
        "5. **Correlation & Multicollinearity:** Compute Pearson and Spearman rank correlation matrices, plot annotated heatmaps, and assess inter-feature collinearity.\n"
        "6. **Domain Chemical Interpretation:** Connect data insights to enological principles (e.g., volatile acidity as acetic acid spoilage, alcohol and sulphates as preservatives and body enhancers).\n"
        "7. **Actionable Roadmap for Day 3:** Establish clear requirements for feature engineering, skewness transforms, feature scaling, and class reformulation."
    ))

    # Code: Imports & Setup
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Ensure project root is in sys.path for local imports\n"
        "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "if str(project_root) not in sys.path:\n"
        "    sys.path.insert(0, str(project_root))\n\n"
        "from src.data_loader import load_wine_data\n"
        "from src.eda_profiling import (\n"
        "    get_feature_columns,\n"
        "    compute_univariate_stats,\n"
        "    detect_outliers_iqr,\n"
        "    compute_target_correlations,\n"
        "    plot_univariate_distributions,\n"
        "    plot_feature_boxplots,\n"
        "    plot_features_vs_quality,\n"
        "    plot_correlation_heatmap,\n"
        "    plot_top_correlations_bar\n"
        ")\n\n"
        "# Visual style configuration\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n"
        "plt.rcParams['figure.dpi'] = 120\n\n"
        "print('Setup complete. All modules imported successfully!')"
    ))

    # Markdown: Data Ingestion
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load Physicochemical Red Wine Dataset\n"
        "We load the red wine dataset from `Data/winequality-red.csv`, containing 1,599 observations of 11 physicochemical measurements and 1 sensory quality rating."
    ))

    # Code: Data Ingestion
    cells.append(nbf.v4.new_code_cell(
        "df = load_wine_data('red', data_dir=project_root / 'Data')\n"
        "features = get_feature_columns(df)\n"
        "print(f'Dataset Shape: {df.shape[0]} samples, {df.shape[1]} columns')\n"
        "print(f'Features ({len(features)}): {features}')\n"
        "df.head()"
    ))

    # Markdown: Univariate Profiling
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Univariate Statistical Profiling & Skewness Analysis\n"
        "We compute a complete descriptive profile for all 11 physicochemical attributes, evaluating:\n"
        "- **Central Tendency:** Mean vs. Median (50% percentile)\n"
        "- **Dispersion:** Standard Deviation and Interquartile Range (IQR)\n"
        "- **Shape:** Skewness (asymmetry) and Kurtosis (tail heaviness / peakedness)"
    ))

    # Code: Univariate Stats Table
    cells.append(nbf.v4.new_code_cell(
        "stats_df = compute_univariate_stats(df, features)\n"
        "stats_df"
    ))

    # Markdown: Univariate Observations
    cells.append(nbf.v4.new_markdown_cell(
        "### Key Univariate Findings:\n"
        "1. **Heavily Skewed Features:**\n"
        "   - `chlorides` (skewness = **+5.680**, kurtosis = **41.714**): High positive skew with extreme outliers up to 0.611 g/dm$^3$ compared to a median of 0.079.\n"
        "   - `residual sugar` (skewness = **+4.541**, kurtosis = **28.618**): Extreme right skew where most dry red wines have 1.9–2.6 g/dm$^3$, but occasional sweet variants reach up to 15.5 g/dm$^3$.\n"
        "   - `sulphates` (skewness = **+2.429**, kurtosis = **11.720**): Noticeable right tail with potassium sulphate concentrations extending to 2.0 g/dm$^3$.\n"
        "   - `free sulfur dioxide` (skewness = **+1.251**) and `total sulfur dioxide` (skewness = **+1.516**): Positively skewed due to preservative concentrations in specific wine batches.\n\n"
        "2. **Approximately Normal Features:**\n"
        "   - `density` (skewness = **+0.071**): Virtually symmetric and centered tightly around 0.9967 g/cm$^3$.\n"
        "   - `pH` (skewness = **+0.194**): Near-perfect Gaussian distribution centered around 3.31.\n\n"
        "3. **Moderately Skewed Features:**\n"
        "   - `fixed acidity` (skewness = **+0.983**), `alcohol` (skewness = **+0.861**), and `volatile acidity` (skewness = **+0.672**)."
    ))

    # Markdown: Distribution Plots
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Visualizing Physicochemical Feature Distributions\n"
        "Histograms with overlaid Kernel Density Estimates (KDE) and markers for mean (red dashed line) and median (green dotted line) clarify distribution shapes and asymmetry."
    ))

    # Code: Distribution Plots
    cells.append(nbf.v4.new_code_cell(
        "fig_dir = project_root / 'reports' / 'figures'\n"
        "fig_dir.mkdir(parents=True, exist_ok=True)\n"
        "dist_path = fig_dir / 'day2_univariate_distributions.png'\n\n"
        "plot_univariate_distributions(df, features, save_path=str(dist_path))\n\n"
        "# Display inline\n"
        "from IPython.display import Image\n"
        "Image(filename=str(dist_path))"
    ))

    # Markdown: Outlier Detection
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Quantitative Outlier Detection & Boxplot Analysis\n"
        "Using **Tukey's Fences** with threshold factor $k=1.5$:\n"
        "$$\\text{Lower Fence} = Q_1 - 1.5 \\times \\text{IQR}, \\quad \\text{Upper Fence} = Q_3 + 1.5 \\times \\text{IQR}$$\n"
        "Observations falling outside these boundaries are classified as statistical outliers."
    ))

    # Code: Outlier Detection Table
    cells.append(nbf.v4.new_code_cell(
        "outlier_df = detect_outliers_iqr(df, features)\n"
        "outlier_df"
    ))

    # Markdown: Boxplot Visualizations
    cells.append(nbf.v4.new_markdown_cell(
        "### 4.1 Feature Boxplots (Spread & Outlier Concentration)\n"
        "Let's visualize the interquartile ranges and outlier points highlighted in red."
    ))

    # Code: Boxplots
    cells.append(nbf.v4.new_code_cell(
        "boxplot_path = fig_dir / 'day2_feature_boxplots.png'\n"
        "plot_feature_boxplots(df, features, save_path=str(boxplot_path))\n"
        "Image(filename=str(boxplot_path))"
    ))

    # Markdown: Outlier Insights
    cells.append(nbf.v4.new_markdown_cell(
        "### Outlier Evaluation & Treatment Strategy:\n"
        "1. **Prevalence:**\n"
        "   - `residual sugar`: 155 samples (**9.69%**) exceed the upper fence (3.65 g/dm$^3$). These reflect semi-sweet or sweet red wine styles, not erroneous records.\n"
        "   - `chlorides`: 112 samples (**7.00%**) exceed the upper fence (0.119 g/dm$^3$), reflecting higher mineral salt content.\n"
        "   - `total sulfur dioxide`: 55 samples (**3.44%**) exceed 122 mg/dm$^3$, with two extreme wines exceeding 275 mg/dm$^3$.\n"
        "2. **Domain Validity:** Chemical features are measured through precise laboratory assays; these values represent genuine enological variability rather than data entry errors.\n"
        "3. **Modeling Strategy for Day 3:**\n"
        "   - Tree-based ensemble models (Random Forest, Gradient Boosting) are naturally invariant to monotonic transformations and robust to univariate outliers.\n"
        "   - For distance-sensitive algorithms (Support Vector Classifier, Logistic Regression, SGD), `RobustScaler` (scaling by median and IQR) or log/power transforms will be applied to prevent extreme values from dominating decision hyperplanes."
    ))

    # Markdown: Bivariate Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Bivariate Analysis: Physicochemical Features Across Quality Scores\n"
        "How do chemical attributes change across sensory quality ratings (scale 3 to 8)?\n"
        "We inspect boxplots of each feature partitioned by target quality score."
    ))

    # Code: Bivariate Plots
    cells.append(nbf.v4.new_code_cell(
        "biv_path = fig_dir / 'day2_features_vs_quality.png'\n"
        "plot_features_vs_quality(df, features, target_col='quality', save_path=str(biv_path))\n"
        "Image(filename=str(biv_path))"
    ))

    # Markdown: Bivariate Observations
    cells.append(nbf.v4.new_markdown_cell(
        "### Key Bivariate Insights:\n"
        "1. **Alcohol (Strong Positive Trend):**\n"
        "   - Quality 3 & 4 wines have median alcohol levels around **9.9%–10.0%**.\n"
        "   - Quality 7 & 8 wines exhibit significantly higher median alcohol around **11.5%–12.1%**.\n"
        "   - Higher alcohol confers body, fullness, and perceived sweetness that wine tasters consistently reward.\n\n"
        "2. **Volatile Acidity (Strong Negative Trend):**\n"
        "   - Quality 3 wines exhibit high median volatile acidity (~**0.84 g/dm$^3$**).\n"
        "   - Quality 8 wines drop to median volatile acidity of ~**0.37 g/dm$^3$**.\n"
        "   - Volatile acidity corresponds to acetic acid (vinegar). Elevated concentrations directly degrade sensory aroma and taste.\n\n"
        "3. **Citric Acid (Positive Trend):**\n"
        "   - Quality 3 wines contain near-zero citric acid (median **0.035 g/dm$^3$**).\n"
        "   - Quality 7 & 8 wines average **0.40–0.42 g/dm$^3$**, providing freshness and crispness.\n\n"
        "4. **Sulphates (Positive Trend):**\n"
        "   - Potassium sulphate increases steadily from median ~**0.54** in score 3 to ~**0.74** in score 8. Sulphates serve as antimicrobial and antioxidant agents preserving wine integrity."
    ))

    # Markdown: Correlation Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Correlation Matrix & Multicollinearity Analysis\n"
        "We examine linear (Pearson) and monotonic (Spearman) relationships between predictors and target quality, as well as inter-feature correlations."
    ))

    # Code: Target Correlation Ranking
    cells.append(nbf.v4.new_code_cell(
        "target_corr = compute_target_correlations(df, target_col='quality')\n"
        "print('Feature Correlations with Sensory Quality:')\n"
        "target_corr"
    ))

    # Code: Target Correlation Bar Plot
    cells.append(nbf.v4.new_code_cell(
        "bar_path = fig_dir / 'day2_top_correlations.png'\n"
        "plot_top_correlations_bar(df, target_col='quality', save_path=str(bar_path))\n"
        "Image(filename=str(bar_path))"
    ))

    # Markdown: Correlation Heatmap
    cells.append(nbf.v4.new_markdown_cell(
        "### 6.1 Inter-Feature Correlation Heatmap\n"
        "Evaluating multicollinearity among the 11 physicochemical predictors."
    ))

    # Code: Heatmap Plot
    cells.append(nbf.v4.new_code_cell(
        "heatmap_path = fig_dir / 'day2_correlation_heatmap.png'\n"
        "plot_correlation_heatmap(df, save_path=str(heatmap_path), method='pearson')\n"
        "Image(filename=str(heatmap_path))"
    ))

    # Markdown: Multicollinearity Insights
    cells.append(nbf.v4.new_markdown_cell(
        "### Multicollinearity Analysis & Chemical Rationale:\n\n"
        "| Feature Pair | Pearson $r$ | Enological / Physical Explanation |\n"
        "| :--- | :---: | :--- |\n"
        "| `fixed acidity` vs. `pH` | **-0.68** | Direct chemical relationship: higher acidic proton concentration lowers pH ($-\\log[H^+]$). |\n"
        "| `fixed acidity` vs. `citric acid` | **+0.67** | Citric acid is a primary organic acid constituent of fixed acidity. |\n"
        "| `fixed acidity` vs. `density` | **+0.67** | Concentrated tartaric acid increases the overall mass density of aqueous wine solutions. |\n"
        "| `free SO2` vs. `total SO2` | **+0.67** | Free sulfur dioxide ($SO_2$) is an equilibrium component of total sulfur dioxide. |\n"
        "| `citric acid` vs. `volatile acidity` | **-0.55** | Clean fermentations with high citric acid typically prevent acetic acid bacterial spoilage. |\n"
        "| `density` vs. `alcohol` | **-0.50** | Ethanol has lower specific gravity ($\\approx 0.789$ g/cm$^3$) than water, decreasing wine density. |\n\n"
        "> **Collinearity Impact:** High correlations between `fixed acidity`, `citric acid`, `density`, and `pH` can cause variance inflation in unregularized linear models. Regularization ($L_1$/$L_2$ penalties) or tree-based feature selection will be essential in modeling."
    ))

    # Markdown: Conclusions & Day 3 Plan
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Day 2 Synthesis & Actionable Plan for Day 3\n\n"
        "### Key Takeaways from Day 2:\n"
        "1. **Core Predictors of Wine Quality:**\n"
        "   - **Alcohol Content** ($r = +0.48$): The strongest single positive predictor of quality.\n"
        "   - **Volatile Acidity** ($r = -0.39$): The strongest negative predictor; acetic acid spoilage strongly detracts from quality.\n"
        "   - **Sulphates** ($r = +0.25$) & **Citric Acid** ($r = +0.23$): Statistically significant positive contributors.\n"
        "2. **Negligible Direct Predictors:**\n"
        "   - `residual sugar` ($r = +0.01$) and `free sulfur dioxide` ($r = -0.05$) show minimal linear correlation with quality, indicating they are baseline sanitary requirements rather than discriminators of excellence.\n"
        "3. **Data Preprocessing Needs:**\n"
        "   - Right-skewed features (`chlorides`, `residual sugar`, `sulphates`) require log1p or power transforms for linear models.\n"
        "   - Feature scaling via `RobustScaler` is recommended due to valid extreme values in chemical assays.\n\n"
        "### Actionable Roadmap for Day 3 (Feature Engineering & Baseline Modeling):\n"
        "- **Target Transformation:** Reframe the imbalanced multi-class problem into actionable binary classification (Good Wine: $\\text{quality} \\ge 7$ vs. Normal/Poor Wine: $\\text{quality} < 7$).\n"
        "- **Feature Engineering:** Create domain-driven interaction features (e.g., total acid ratio, free-to-total $SO_2$ ratio, sugar-alcohol balance).\n"
        "- **Pipeline Construction:** Build standardized scikit-learn pipelines with `RobustScaler` and `StandardScaler`.\n"
        "- **Baseline Classification Models:** Train and benchmark Random Forest, Support Vector Classifier (SVC), and Stochastic Gradient Descent (SGD) with stratified cross-validation."
    ))

    nb.cells = cells
    notebook_path = "notebooks/Day2_Exploratory_Data_Analysis.ipynb"

    # Save unexecuted notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {notebook_path}")

    # Execute notebook using ExecutePreprocessor
    print("Executing Day 2 notebook to generate outputs and persist figures...")
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "notebooks"}})

    # Save executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed notebook saved with cell outputs to {notebook_path}")


if __name__ == "__main__":
    create_day2_notebook()
