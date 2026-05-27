# Full Report of the Bulldozers Price Prediction Project

## Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [Data Preprocessing and Exploratory Data Analysis](#2-data-preprocessing-and-exploratory-data-analysis)
3. [Modelling and Hyperparameter Tuning](#3-modelling-and-hyperparameter-tuning)
4. [Final Model Evaluation and Comparison](#4-final-model-evaluation-and-comparison)
5. [Feature Importance Analysis and Model Interpretability](#5-feature-importance-analysis-and-model-interpretability)
6. [Conclusion](#6-conclusion)
7. [Acknowledgements](#7-acknowledgements)

---

## 1. Problem Definition

### 1.1 Problem Statement
Predicting the future sale price of heavy equipment (such as bulldozers) based on historical auction data, usage specifications, and equipment characteristics. This helps buyers and sellers make data-driven valuation decisions in a dynamic second-hand machinery market.

### 1.2 Machine Learning Formulation
* **Learning Type:** Supervised Regression
* **Target Variable:** Sale Price (`SalePrice` - continuous numerical value)
* **Validation Strategy:** Chronological data splitting (training on historical records up to a specific cutoff date, and validating on subsequent time periods) to strictly prevent data leakage and simulate real-world forecasting constraints.

### 1.3 Evaluation Metrics
* **RMSLE (Root Mean Squared Logarithmic Error):** The primary evaluation metric. By taking the log of both predicted and actual values, RMSLE penalizes relative errors equally, preventing high-priced machinery from disproportionately dominating the loss function.
* **MAE (Mean Absolute Error):** Measures the average magnitude of absolute errors in dollar terms for intuitive business interpretation.
* **$R^2$ Score (Coefficient of Determination):** Evaluates the overall goodness of fit and the proportion of variance explained by the model.

### 1.4 Data Source
* **Dataset:** Heavy Equipment / Bulldozers Auction Dataset (inspired by the classic Kaggle Blue Book for Bulldozers challenge). Comprises rich multi-attribute records including equipment models, usage metrics, and temporal sale timestamps.

---

## 2. Data Preprocessing and Exploratory Data Analysis

### 2.1 Data Overview & Dictionary
* **Data Dictionary:** Due to the comprehensive and multi-attribute nature of the heavy machinery dataset, a detailed data dictionary is provided as an Excel file (`.xlsx`) directly inside the `data.zip` archive for complete field-by-field reference.
* **Preprocessing Pipeline:** Handled missing values systematically, converted raw timestamp objects into granular temporal features (`saleYear`, `saleMonth`, `saleDay`, `saleDayofweek`, `saleDayofyear`), and encoded categorical variables while maintaining strict pipeline boundaries to prevent data leakage.

### 2.2 Exploratory Data Analysis (EDA)

#### Target Variable Distribution (`SalePrice`)
<p align="center">
  <img src="plots/EDA/01_saleprice_distribution.png" alt="Distribution of SalePrice" width="85%">
</p>

* **Right-Skewed Distribution:** The distribution of `SalePrice` exhibits a pronounced right-skew, where a large concentration of heavy equipment items trades at lower price points (peaking between \$10,000 and \$20,000), while a long tail extends towards high-end machinery valued past \$100,000. 
* **Justification for RMSLE:** This skewness strongly justifies the use of Root Mean Squared Logarithmic Error (RMSLE) as the primary evaluation metric, ensuring that percentage errors across both cheap and expensive machinery are weighted equitably.

#### Temporal Trends & Seasonal Decomposition

<p align="center">
  <img src="plots/EDA/02_saleprice_seasonal_decomposition.png" alt="Seasonal Decomposition of SalePrice" width="80%">
</p>

* **Trend Component:** The time-series trend analysis uncovers macroeconomic shifts in second-hand equipment valuation over the multi-year timeline, showing distinct market cycles, valuation troughs around 1999–2000, and recovery peaks leading up to 2007.
* **Seasonality & Regularity:** The seasonal decomposition plot reveals strong, recurring annual periodicity in equipment auction pricing, emphasizing the importance of extracting granular calendar attributes (`saleMonth`, `saleDayofyear`) to help gradient boosting models capture cyclic market behavior.

#### SalePrice Distribution by Product Size and Enclosure Type
<p align="center">
  <img src="plots/EDA/03_saleprice_by_productsize_enclosure.png" alt="SalePrice Distribution by ProductSize and Enclosure" width="85%">
</p>

* **Product Scale & Valuation Impact:** As machinery size scales up from Mini and Compact to Medium, Large/Medium, and Large, the median and maximum sale prices shift upward significantly, demonstrating that physical scale is a primary structural determinant of equipment value.
* **Enclosure Specifications:** Across nearly all size categories, premium cab features—specifically enclosed operator cabins with air conditioning (`EROPS w AC`)—command a substantial price premium compared to open-roll-over protection structures (`OROPS` or standard `EROPS`), reflecting strong market demand for operator comfort features.

#### SalePrice vs. YearMade (Post-1950)
<p align="center">
  <img src="plots/EDA/04_saleprice_vs_yearmade.png" alt="SalePrice vs YearMade" width="85%">
</p>

* **Asset Age & Depreciation Dynamics:** The scatter plot illustrates the direct relationship between manufacturing year (`YearMade`) and auction sale price using a 5,000-sample subset. Newer machines consistently reach higher ceiling prices, while older equipment (pre-1980s) clusters heavily at lower price brackets due to depreciation and technological obsolescence.
* **Non-Linear Price Dispersion:** While newer manufacturing years unlock higher maximum market valuations (exceeding \$120,000 for models built post-2000), a wide dispersion in prices exists within individual manufacturing years, highlighting the combined influence of usage hours, mechanical condition, and specific product tiers.

---

## 3. Modelling and Hyperparameter Tuning

### 3.1 Model Formulation & Baseline Benchmarking
* **Initial Estimator Selection:** Built baseline models to establish performance floors, focusing primarily on high-efficiency gradient boosting frameworks (LightGBM) capable of handling tabular categorical interactions and temporal splits efficiently.
* **Baseline Comparison:** Evaluated `Baseline LightGBM` against a multi-estimator `Stacking (LGBM + Huber)` approach. While stacking yielded a marginal improvement in RMSLE (0.2867 vs 0.2874) and MAE (\$7,250.57 vs \$7,369.87), it incurred an extreme computational penalty, requiring **96.62s** of training time compared to just **1.11s** for the standalone baseline.

#### Baseline Model Comparison
<p align="center">
  <img src="plots/Model_Selection/01_baseline_model_comparison.png" alt="Baseline Model Comparison" width="90%">
</p>

### 3.2 Hyperparameter Optimization
* **Systematic Tuning Strategy:** Deployed targeted hyperparameter optimization (including learning rates, tree depth, and number of estimators) to boost predictive accuracy without triggering overfitting on chronological validation boundaries.
* **Efficiency vs. Accuracy Trade-off:** Tuning unlocked substantial performance gains for LightGBM, transforming it from a baseline estimator into the definitive champion model.

---

## 4. Final Model Evaluation and Comparison

### 4.1 Comprehensive Performance Metrics
We benchmarked four distinct configurations across Validation RMSLE, Validation MAE, $R^2$ Score, and Execution/Tuning Time.

<p align="center">
  <img src="plots/Model_Selection/02_final_model_comparison.png" alt="Final Model Comparison" width="90%">
</p>

### 4.2 Key Evaluation Findings
* **Champion Model:** **Tuned LightGBM** achieved the best overall performance across all accuracy metrics, securing the lowest Validation RMSLE (**0.2071**), the lowest Validation MAE (**\$5,092.09**), and the highest $R^2$ Score (**0.9205**).
* **Tuned Stacking Performance:** The `Tuned Stacking` configuration performed strongly with an RMSLE of **0.2412**, an MAE of **\$6,017.42**, and an $R^2$ of **0.8922**, but required the longest runtime (**104.71s**).
* **Cost-Benefit Conclusion:** Standalone **Tuned LightGBM** proved to be the optimal solution, outperforming complex stacked architectures while maintaining a faster and more efficient training time (**77.29s**).

---

## 5. Feature Importance Analysis and Model Interpretability

### 5.1 Permutation Importance
To evaluate which features exert the greatest overall influence on model predictions, we computed **Permutation Feature Importances** on the champion Tuned LightGBM model. 

<p align="center">
  <img src="plots/Features/01_permutation_importance_tuned_lgb.png" alt="Top 15 Permutation Feature Importances" width="85%">
</p>

* **Core Pricing Determinants:** `num__YearMade` and `cat__ProductSize` dominate the importance ranking, causing the largest decrease in model score (RMSLE) when permuted, confirming that asset age and physical scale are the primary pillars of equipment valuation.
* **Structural Specifications:** Secondary features such as `cat__Coupler_System`, `cat__fiProductClassDesc`, and `cat__fiSecondaryDesc` provide vital contextual refinement for the model's price estimations.

### 5.2 SHAP Model Interpretability (Shapley Additive exPlanations)
We further utilized **SHAP Beeswarm Plots** to uncover both the magnitude and direction of feature impacts on individual predictions.

<p align="center">
  <img src="plots/Features/02_shap_beeswarm_tuned_tuned_lgb.png" alt="SHAP Beeswarm Plot for Top 15 Features" width="85%">
</p>

* **Feature Directionality:** High values of `num__YearMade` (newer machines, shown in pink) exert a strong positive impact on predicted sale price, whereas older machines (blue) heavily depress valuations.
* **Non-Linear Interactions:** The distribution of points across features like `cat__ProductSize` and `cat__Enclosure` highlights complex, non-linear interactions where specific equipment configurations scale pricing outcomes dynamically across the auction dataset.

---

## 6. Conclusion

### 6.1 Summary of Achievements
This project successfully developed an end-to-end machine learning pipeline to forecast heavy machinery auction sales prices with high predictive accuracy. By implementing strict chronological data splitting, robust preprocessing architectures using Scikit-Learn `Pipeline`, and advanced feature engineering, we ensured the model reflects real-world market constraints without data leakage.

### 6.2 Key Takeaways & Impact
* **Champion Model Superiority:** The **Tuned LightGBM** model emerged as the definitive solution, striking an ideal balance between accuracy and computational efficiency with a validation RMSLE of **0.2071**, a low MAE of **\$5,092.09**, and an $R^2$ score of **0.9205**.
* **Model Interpretability:** Through comprehensive Permutation Importance and SHAP analysis, we demonstrated that equipment valuation is fundamentally driven by asset manufacturing year (`num__YearMade`) and physical scale (`cat__ProductSize`), alongside structural refinements like cabin enclosures.
* **Interactive Web Application:** Encapsulating the trained pipeline into an interactive **Streamlit** dashboard bridges the gap between raw data science modeling and practical end-user deployment.

---

## 7. Acknowledgements

-   **Dataset:** Bulldozers Price Predictor Dataset (inspired by the classic Kaggle Blue Book for Bulldozers challenge).
-   **Inspiration & Base Concepts:** Inspired by the foundational workflow from the [Zero to Mastery Machine Learning Course](https://github.com/mrdbourke/zero-to-mastery-ml).

### Key Improvements & Technical Enhancements

This project goes significantly beyond the baseline course material through full end-to-end refactoring and expansion:

-   **End-to-End Pipeline Architecture:** Fully rewritten training, preprocessing, and feature transformation workflows using Scikit-Learn `Pipeline` to prevent data leakage.
-   **Advanced Model Benchmarking:** Integrated advanced regression models alongside baseline estimators.
-   **Advanced Visualizations & Interpretation:** Built custom exploratory data analysis (EDA) visual formats, residual plots, and comprehensive feature importance analysis (including `shap`) for enhanced model interpretability.
-   **Refactored Model Evaluation:** Overhauled performance tracking with systematic cross-validation metrics, RMSLE/MAE comparisons, and structured evaluation outputs.
-   **Interactive Web Deployment:** Developed and deployed a dynamic, production-ready prediction dashboard using **Streamlit**.