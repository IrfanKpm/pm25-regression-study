# PM2.5 Regression Study

A machine learning study exploring the boundaries of **polynomial regression, overfitting, and regularization** using global air quality data.

## 🚀 Overview
This project investigates how PM2.5 concentrations can be modeled using environmental and contextual features. The primary focus is not forecasting, but understanding how model complexity affects generalization in regression models.

It serves as a hands-on study of the bias–variance tradeoff, demonstrating how polynomial transformations can both improve expressiveness and introduce severe overfitting.

## 🎯 Key Learning Objectives
*   **Overfitting Visualization**: Mapping how high-degree polynomials (1–12) "memorize" noise.
*   **Error Analysis**: Plotting the **U-shaped validation curve** to identify the optimal model degree.
*   **Regularization**: Applying **Ridge (L2)** and **Lasso (L1)** to penalize complexity.
*   **Hyperparameter Tuning**: Constructing a **degree vs. λ (lambda)** heatmap to find the "sweet spot" for performance.

## 🧰 Tech Stack
*   **Data**: [OpenAQ API](https://openaq.org)
*   **Analysis**: `numpy`, `pandas`
*   **Modeling**: `scikit-learn`
*   **Visualization**: `seaborn`, `matplotlib`

## 📊 Planned Experiments
1.  **Complexity Sweep**: Iterative training from Degree 1 (linear) to Degree 12.
2.  **RMSE Profiling**: Real-time tracking of Training vs. Validation error.
3.  **Penalty Testing**: Comparing L1 vs. L2 regularization effects on coefficient shrinkage.
4.  **Grid Mapping**: Manual cross-validation loops to generate performance heatmaps.

## 📁 Project Structure
```text
├── data/       # Raw and processed air quality datasets
├── notebooks/  # Exploration and experiment walkthroughs
├── src/        # Core scripts for data fetching and modeling
└── results/    # Generated plots, heatmaps, and metrics
```

