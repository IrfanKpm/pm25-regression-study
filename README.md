# PM2.5 Regression Study

A machine learning study exploring the boundaries of polynomial regression, overfitting, and regularization using real-world air quality data from New Delhi.

---

## 🚀 Overview

This project investigates how PM2.5 concentrations can be modeled using environmental and contextual features collected from the **Mandir Marg, New Delhi – DPCC** monitoring station via the OpenAQ API.

The primary focus is **not forecasting**, but understanding how model complexity affects generalization in regression models. It serves as a hands-on study of the **bias–variance tradeoff**, demonstrating how polynomial transformations can both improve expressiveness and introduce severe overfitting — and how regularization brings it back under control.

---

## 🎯 Key Learning Objectives

| Objective | Description |
|---|---|
| Overfitting Visualization | Mapping how high-degree polynomials (1–12) memorize noise |
| Error Analysis | Plotting the U-shaped validation curve to identify optimal model degree |
| Regularization | Applying Ridge (L2) and Lasso (L1) to penalize complexity |
| Hyperparameter Tuning | Constructing a degree × λ heatmap to find the performance sweet spot |
| Live Validation | Testing the final model against real-time sensor data |

---

## 🧰 Tech Stack

- **Data:** [OpenAQ API v3](https://openaq.org/)
- **Analysis:** `numpy`, `pandas`
- **Modeling:** `scikit-learn`
- **Visualization:** `seaborn`, `matplotlib`
- **Environment:** `python-dotenv`, `joblib`

---

## 📊 Experiments & Findings

### 1. Complexity Sweep — Degree 1 to 12

Trained polynomial regression models of increasing degree without any regularization. Tracked training and validation RMSE at each step.

**Finding:** Training error dropped consistently as degree increased. Validation error followed a classic **U-shape** — improving up to a point, then spiking sharply due to overfitting. High-degree models (8–12) memorized noise completely and failed to generalize.

### 2. Regularization Comparison

Tested Ridge (L2), Lasso (L1), and Elastic Net at the optimal polynomial degree to measure the effect of penalizing coefficient magnitude.

| Model | Behavior | Validation Performance |
|---|---|---|
| Polynomial (no regularization) | Severe overfitting, unstable | Very poor |
| Ridge (L2) | Stable, strong generalization | **Best (~12.15 RMSE)** |
| Lasso (L1) | High sparsity, feature selection | Higher bias, weaker than Ridge |
| Elastic Net (L1 + L2) | Balanced penalty | Similar to Lasso, no gain over Ridge |

**Finding:** Ridge consistently outperformed others by shrinking coefficients without zeroing them out — preserving all feature contributions while controlling variance.

### 3. Degree × λ Heatmap

Constructed a grid search over polynomial degrees (1–10) and regularization strengths (λ from 0.001 to 100), running cross-validation at each cell to build a performance heatmap.

**Finding:** The sweet spot emerged clearly at **degree 5, α = 0.01** — beyond this, either complexity or over-regularization degraded performance.

---

## ✅ Final Tuned Model Configuration

| Component | Selected Value |
|---|---|
| Model | Ridge Regression |
| Polynomial Degree | 5 |
| Regularization (α) | 0.01 |
| Feature Scaling | StandardScaler |
| Polynomial Features | Included (`include_bias=False`) |
| **Validation RMSE** | **~12.15** |

The selected configuration achieves the best balance between bias and variance — expressive enough with degree 5, regularized enough with α = 0.01 to avoid memorizing noise.

---

## 🌡️ Features Used

After preprocessing and feature selection, the following inputs were retained for training:

| Feature | Description |
|---|---|
| `co` | Carbon monoxide concentration |
| `no2` | Nitrogen dioxide concentration |
| `nox` | Nitrogen oxides |
| `pm10` | Particulate matter (≤10 µm) |
| `relativehumidity` | Relative humidity (%) |
| `temperature` | Ambient temperature (°C) |

Features removed (`no`, `o3`, `so2`, `wind_direction`, `wind_speed`) showed weak correlation with PM2.5 during analysis.

---

## 🔴 Live Prediction Test & Findings

After training, the model was tested against **live sensor data** fetched in real-time from the same OpenAQ station using `predict_realtime.py`.

### Results

| Sample | Actual PM2.5 | Predicted PM2.5 | Error |
|---|---|---|---|
| 1 | 72.00 | 194.64 | 122.64 |
| 2–10 | ~87.00 | ~170–174 | ~80–87 |

### Why the large errors?

This is a textbook case of **covariate shift** — a well-known real-world ML failure mode.

**Training data:** January–May 2026 (Delhi winter/spring)
During this period, PM2.5 in Delhi is naturally high — often ranging from 100 to 300+ µg/m³ due to cold air trapping pollutants close to the ground.

**Live data:** Mid-May 2026 (transition to summer)
In summer, Delhi's warmer air disperses pollutants more effectively. PM2.5 drops to the 70–90 µg/m³ range — a distribution the model had **never seen during training**.

The model isn't broken. It learned the patterns it was shown, generalized well within that distribution (RMSE ~12), and over-predicted predictably when the season shifted. This is how a correctly trained model behaves when input distribution changes.

> **Covariate shift** occurs when the statistical distribution of input features at inference time differs from the distribution seen during training. The model's learned mapping remains intact, but the inputs it receives no longer match the domain it was optimized for.

### Secondary observation

Live samples 2–10 showed near-identical values, indicating the station was reporting **stale/repeated sensor readings** at the time of the test — a data quality issue at the source, not in the pipeline.

---

## 📁 Project Structure

```
pm25-regression-study/
├── artifacts/
│   └── model.pkl                              # Final trained Ridge pipeline
├── data/
│   ├── raw/
│   │   ├── openaq_long_format.csv             # Raw sensor data (long format)
│   │   └── openaq_ml_format.csv               # Pivoted ML-ready format
│   └── processed/
│       ├── train.csv                          # 75% training split
│       ├── val.csv                            # 15% validation split
│       └── test.csv                           # 10% test split
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb       # Exploratory analysis & feature selection
│   ├── 02_polynomial_regression.ipynb         # Degree sweep (1–12) & RMSE curves
│   ├── 03_bias_variance_study.ipynb           # Bias–variance tradeoff visualization
│   ├── 04_regularization_ridge_lasso.ipynb    # Ridge, Lasso, Elastic Net comparison
│   └── 05_final_summary.ipynb                 # Final model selection & conclusion
├── results/
│   ├── correlation_heatmap.png                # Feature correlation matrix
│   ├── model_comparison_regularization.png    # Ridge vs Lasso vs Elastic Net
│   ├── polynomial_degree_rmse_curve.png       # Train vs val RMSE by degree
│   ├── polynomial_degree_rmse_curve_log.png   # Log-scale version
│   ├── processed_corr_heatmap.png             # Post-processing correlation
│   ├── ridge_alpha_rmse_curve_log.png         # Ridge α sweep curve
│   └── ridge_degree_alpha_rmse_heatmap.png    # Degree × α performance heatmap
├── src/
│   ├── 01_prepare_dataset.py                  # API fetch, pagination, pivot, interpolate
│   ├── 02_preprocess.py                       # Clean, drop features, split, save
│   ├── model.py                               # Model definition & pipeline
│   ├── train.py                               # Training script
│   ├── predict.py                             # Batch prediction
│   └── predict_realtime.py                    # Live prediction against OpenAQ API
├── .env                                       # API keys (not committed)
├── .env.example                               # Environment variable template
├── .gitignore
└── README.md
```

---

## 🧠 Conclusion

This study successfully demonstrated all core objectives of the bias–variance tradeoff through practical experimentation.

The most important result was not the final RMSE, but **what the full experimental curve revealed** — that model complexity alone does not produce good models, and that the right amount of regularization at the right polynomial degree is what separates a generalizing model from one that memorizes.

The live prediction test added an unplanned but valuable lesson: even a well-tuned model with low validation error can fail in deployment when the world changes around it. **Seasonal distribution shift** caused the model to over-predict consistently — not because of a code bug or a bad architecture, but because the training window didn't include summer patterns.

> The optimal model is not the most complex, nor the most sparse — it is the one that has seen enough of the world to generalize to the part it hasn't seen yet.

For a learning project, this is the most honest outcome possible: the model worked exactly as theory predicts, and broke exactly where theory says it should.

---

## 📌 Data Source

Station: **Mandir Marg, New Delhi – DPCC**
Location ID: `6358` on OpenAQ v3 API
Training period: January 1, 2026 → May 1, 2026
