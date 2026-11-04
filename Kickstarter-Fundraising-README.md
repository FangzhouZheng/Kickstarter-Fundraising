# Kickstarter Fundraising Success Prediction

## Project Overview

This project investigates factors associated with the success of Kickstarter crowdfunding campaigns and compares supervised machine learning models for predicting campaign outcomes. Using campaign metadata, the analysis explores how funding goals, project categories, campaign duration, and other attributes relate to fundraising success.

**Objectives**
- Explore patterns in Kickstarter campaign outcomes.
- Prepare campaign data and identify informative predictors.
- Train, tune, and compare classification models.
- Evaluate performance with particular attention to class imbalance.

## Dataset

**Source:** [Web Robots — Kickstarter Datasets](https://webrobots.io/kickstarter-datasets/)

| Item | Value |
| --- | ---: |
| Initial records | 192,548 |
| Records in the modeling subset | 43,698 |
| Selected launch period | January 2018 – December 2019 |
| Training / test split | 70% / 30% |
| Training records | 30,588 |
| Test records | 13,110 |

The analysis uses Kickstarter campaign records and focuses on a selected launch-date window. Results describe this sample and should not automatically be generalized to all campaigns or later years.

## Tools and Libraries

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Model selection:** GridSearchCV, RandomizedSearchCV, Bayesian optimization, cross-validation
- **Visualization:** Matplotlib, Seaborn, Missingno

## Methodology

### 1. Data Preparation and Exploratory Analysis

- Inspected data types, campaign attributes, and missingness.
- Filtered campaigns to the 2018–2019 study period.
- Engineered and transformed features for classification.
- Applied preprocessing for numeric and categorical variables, including scaling and encoding.
- Examined distributions and relationships between campaign characteristics and outcomes.

### 2. Feature Selection

A Random Forest-based `SelectFromModel` workflow was used to identify informative features. The notebook reports **11 selected features**, including campaign duration, funding goal in USD, description length, and project-category indicators.

### 3. Classification Models

The notebook evaluates multiple approaches:

- Logistic Regression, including L1/L2 regularization and class weighting
- Random Forest
- K-Nearest Neighbors (KNN)
- Voting ensemble
- Stacking ensemble with XGBoost as a meta-model

Hyperparameters were explored using grid search, randomized search, and Bayesian optimization. Cross-validation was used alongside holdout test-set evaluation.

### 4. Evaluation

Because the outcome classes are imbalanced, **balanced accuracy** is emphasized rather than relying on ordinary accuracy alone. The notebook also examines confusion matrices and classification metrics.

## Results

Selected balanced-accuracy results reported in the notebook are summarized below. These correspond to specific model configurations, not necessarily every experiment performed.

| Model / configuration | Balanced accuracy |
| --- | ---: |
| Logistic Regression (L2, balanced class weights), test set | **70.66%** |
| Logistic Regression (L2), cross-validation | 71.18% |
| Random Forest, test set | 69.20% |
| Stacking ensemble, test set | 69.42% |

**Key takeaway:** The reported **70.66% test balanced accuracy** for the tuned Logistic Regression model exceeds the listed Random Forest and Stacking test results. The **71.18% cross-validation score is not a holdout test score** and should not be presented as one.

The notebook explores additional KNN and Voting configurations; see the notebook for their full evaluation outputs.

## Business Relevance

This analysis demonstrates how campaign-level information can be used to study factors associated with fundraising outcomes. Potential applications include:

- Identifying campaign attributes associated with different success rates.
- Helping creators understand how project characteristics relate to historical outcomes.
- Comparing predictive methods before building a more rigorous decision-support tool.

These findings are **associational, not causal**. They do not establish that changing a feature will cause a campaign to succeed.

## Limitations and Future Work

- **Historical sample:** The modeling subset covers 2018–2019, so performance may change on newer campaigns.
- **Class imbalance:** Ordinary accuracy can obscure poor performance on a minority class; balanced accuracy helps, but additional metrics are useful.
- **Prediction-time availability:** Before deploying a model, verify that every predictor is available at the intended prediction time and does not leak the final fundraising outcome.
- **Reproducibility:** Re-run model evaluations and check prediction-variable reuse before treating all notebook outputs as finalized results.
- **Generalization:** Validate the model on newer, out-of-time data and compare it against simple baselines.

## Repository Structure

```text
Kickstarter-Fundraising/
├── README.md
└── Kickstarter_Fundraising_Analysis.ipynb
```

> Update the notebook filename above if your uploaded file uses a different name. The dataset is not included in this structure.

## Project Information

**Project type:** Academic team project  
**Team members:** Bei Su, Fangzhou Zheng, Haojiang Wu  
**Focus areas:** Exploratory Data Analysis, Feature Engineering, Supervised Machine Learning, Model Evaluation
