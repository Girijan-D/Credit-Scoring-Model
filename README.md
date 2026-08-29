# Credit Scoring Model for Loan Default Prediction
### An End-to-End Machine Learning, Explainable AI (XAI), and Business Intelligence Solution

[![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost%20%7C%20Streamlit-orange.svg)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## Table of Contents
1. [Project Title & Executive Summary](#1-project-title--executive-summary)
2. [Problem Statement & Business Context](#2-problem-statement--business-context)
3. [Project Objective](#3-project-objective)
4. [Dataset Description & Schema](#4-dataset-description--schema)
5. [Discovered Features & Data Dictionary](#5-discovered-features--data-dictionary)
6. [Data Preprocessing Pipeline (Zero Data Leakage)](#6-data-preprocessing-pipeline-zero-data-leakage)
7. [Exploratory Data Analysis (EDA) Findings](#7-exploratory-data-analysis-eda-findings)
8. [Domain Feature Engineering](#8-domain-feature-engineering)
9. [Machine Learning Models Evaluated](#9-machine-learning-models-evaluated)
10. [Comprehensive Evaluation Metrics](#10-comprehensive-evaluation-metrics)
11. [Final Model Selection & Justification](#11-final-model-selection--justification)
12. [Inference & Credit Risk Prediction Methodology](#12-inference--credit-risk-prediction-methodology)
13. [Explainable AI (SHAP & Feature Attributions)](#13-explainable-ai-shap--feature-attributions)
14. [Interactive Streamlit Web Application](#14-interactive-streamlit-web-application)
15. [Power BI Dashboard Specification & DAX Measures](#15-power-bi-dashboard-specification--dax-measures)
16. [Limitations & Model Governance](#16-limitations--model-governance)
17. [Responsible Use & Ethical AI Considerations](#17-responsible-use--ethical-ai-considerations)
18. [Installation & How to Run](#18-installation--how-to-run)
19. [Concise Project Audit](#19-concise-project-audit)

---

## 1. Project Title & Executive Summary

**Project Title:** Credit Scoring Model for Loan Default Prediction  
**Domain:** Financial Services / Credit Underwriting / Machine Learning  

This project delivers a machine learning solution that automates credit risk assessment and predicts the probability of loan applicant default using authentic financial, demographic, behavioral, and credit history variables. The pipeline includes data preprocessing, domain-specific feature engineering, multi-model benchmarking with 5-fold cross-validation, explainability (SHAP), an interactive 5-page Streamlit web dashboard, and export files with DAX formulas for Power BI.

---

## 2. Problem Statement & Business Context

Loan default occurs when a borrower fails to meet the legal obligations of a loan agreement. For financial institutions and digital lenders:
- **Default Losses (Charge-offs):** Defaults lead directly to capital write-downs, interest losses, and expensive collection proceedings.
- **The False Negative (FN) Cost:** Approving a high-risk applicant who later defaults is the single most expensive error in banking (loss of up to 100% of loan principal).
- **The False Positive (FP) Cost:** Declining a creditworthy applicant results in lost interest revenue and customer dissatisfaction.

A machine learning credit scoring model optimizes this trade-off by estimating default probabilities ($0.0\% - 100.0\%$) and categorizing applicants into actionable risk bands.

---

## 3. Project Objective

1. Develop a high-accuracy classification system to predict loan default (`loan_status` = 1) vs. non-default (`loan_status` = 0).
2. Utilize an authentic, verified credit dataset with zero synthetic or fabricated values.
3. Compare 4 model families (Logistic Regression, Decision Trees, Random Forest, XGBoost) using Stratified 5-Fold Cross-Validation.
4. Implement explainable AI (SHAP) for transparent underwriting decisions.
5. Deploy an interactive Streamlit application and provide Power BI analytics assets.

---

## 4. Dataset Description & Schema

- **Source:** Kaggle Credit Risk Dataset (`credit_risk_dataset.csv`).
- **Raw Observations:** 32,581 applicant records.
- **Cleaned Observations:** 32,409 records (165 duplicates and 7 data entry anomalies removed).
- **Class Balance:**
  - Non-Default (`0`): 25,321 (78.13%)
  - Default (`1`): 7,088 (21.87%)
- **Data Partitions (80/20 Stratified Split):**
  - **Training Set:** 25,927 records (Non-default: 20,257 | Default: 5,670)
  - **Testing Set:** 6,482 records (Non-default: 5,064 | Default: 1,418)

---

## 5. Discovered Features & Data Dictionary

| Variable Name | Category | Data Type | Description | Range / Values |
| :--- | :--- | :--- | :--- | :--- |
| `person_age` | Demographic | Integer | Applicant age in years | 20 – 94 |
| `person_income` | Financial | Integer | Annual income ($) | $4,000 – $6,000,000 |
| `person_home_ownership` | Demographic | Categorical | Residential status | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `person_emp_length` | Demographic/Financial | Float | Employment tenure (years) | 0.0 – 41.0 (895 missing) |
| `loan_intent` | Loan Purpose | Categorical | Purpose of credit requested | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` |
| `loan_grade` | Credit Quality | Categorical | Assigned credit rating tier | `A`, `B`, `C`, `D`, `E`, `F`, `G` |
| `loan_amnt` | Financial | Integer | Requested principal loan amount ($) | $500 – $35,000 |
| `loan_int_rate` | Financial / Risk | Float | Loan interest rate (%) | 5.42% – 23.22% (3,116 missing) |
| `loan_percent_income` | Financial Burden | Float | Ratio of loan amount to annual income | 0.00 – 0.83 |
| `cb_person_default_on_file` | Credit History | Categorical | Historical credit default on record | `Y` (Yes), `N` (No) |
| `cb_person_cred_hist_length` | Credit History | Integer | Credit bureau history length (years) | 2 – 30 |
| **`loan_status`** | **Target** | **Integer** | **Loan default indicator** | **0 = Non-Default, 1 = Default** |

---

## 6. Data Preprocessing Pipeline (Zero Data Leakage)

To guarantee zero data leakage between training and testing splits:
1. **Deduplication:** Removed 165 exact duplicate records.
2. **Anomaly Filtering:** Excluded impossible entries (`person_age > 100` and `person_emp_length > 60` or `emp_length > age`).
3. **Partitioning:** Conducted an 80/20 Stratified Split (`random_state=42`, `stratify=y`).
4. **ColumnTransformer Pipeline (Fitted ONLY on Training Set):**
   - **Numerical Features (14 variables):** Median Imputation (`SimpleImputer(strategy='median')`) followed by standard feature normalization (`StandardScaler()`).
   - **Categorical Features (4 variables):** Mode Imputation (`SimpleImputer(strategy='most_frequent')`) followed by One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
5. **Artifact Persistence:** Fitted pipeline serialized as `models/preprocessor.joblib`.

---

## 7. Exploratory Data Analysis (EDA) Findings

1. **Credit Grade Sensitivity:** Loan Grade is strongly associated with default risk:
   - Grade A Default Rate: **9.7%**
   - Grade D Default Rate: **59.3%**
   - Grade G Default Rate: **98.1%**
2. **Housing Status Impact:** Renters have a default rate of **31.6%**, compared to **7.5%** for outright homeowners and **12.8%** for mortgage holders.
3. **Loan Intent Risk:** `DEBTCONSOLIDATION` (28.4%) and `MEDICAL` (27.3%) show higher default incidence than `VENTURE` (14.8%) and `EDUCATION` (17.1%).
4. **Historical Credit Record:** Applicants with a prior default record default at **37.6%**, compared to **17.9%** for those without prior defaults.
5. **Income Leverage:** Defaulted borrowers exhibit a median loan-to-income ratio of **0.26**, compared to **0.13** for non-defaulted borrowers.

All visual assets are stored in `outputs/eda_charts/`.

---

## 8. Domain Feature Engineering

The following domain features are implemented in `src/feature_engineering.py`:

1. **Loan-to-Income Ratio ($\text{LTI}$):**
   $$\text{LTI} = \frac{\text{loan\_amnt}}{\max(\text{person\_income}, 1)}$$
2. **Income-to-Loan Ratio ($\text{ITL}$):**
   $$\text{ITL} = \frac{\text{person\_income}}{\max(\text{loan\_amnt}, 1)}$$
3. **Annual Interest Expense Burden ($\$$):**
   $$\text{Interest Burden} = \text{loan\_amnt} \times \left(\frac{\text{loan\_int\_rate}}{100}\right)$$
4. **Interest Burden Ratio:**
   $$\text{Burden Ratio} = \frac{\text{Interest Burden}}{\max(\text{person\_income}, 1)}$$
5. **Credit Experience to Age Proportion:**
   $$\text{Credit Age Ratio} = \frac{\text{cb\_person\_cred\_hist\_length}}{\max(\text{person\_age}, 1)}$$
6. **Employment to Age Ratio:**
   $$\text{Emp Age Ratio} = \frac{\text{person\_emp\_length}}{\max(\text{person\_age}, 1)}$$
7. **High-Risk Credit Flag:**
   $$\text{High Risk} = \begin{cases} 1 & \text{if } \text{default\_on\_file} = \text{'Y'} \text{ and } \text{loan\_grade} \in \{\text{'D','E','F','G'}\} \\ 0 & \text{otherwise} \end{cases}$$

---

## 9. Machine Learning Models Evaluated

Four model architectures were trained with class weighting and Stratified 5-Fold Cross-Validation:

1. **Logistic Regression:** Regularized linear baseline with balanced class weights (`C=1.0`).
2. **Decision Tree Classifier:** Cost-complexity and depth-controlled tree (`max_depth=6`, `min_samples_split=20`).
3. **Random Forest Classifier:** Ensemble of 200 balanced decision trees (`n_estimators=200`, `max_depth=12`, `min_samples_split=10`).
4. **XGBoost Classifier:** Gradient boosted decision trees (`n_estimators=200`, `learning_rate=0.08`, `max_depth=5`, `scale_pos_weight=3.57`).

---

## 10. Comprehensive Evaluation Metrics

### 5-Fold Stratified Cross-Validation (Training Set)
| Model | Mean 5-Fold ROC-AUC | Std Deviation (CV) |
| :--- | :---: | :---: |
| Logistic Regression | 0.8795 | $\pm 0.0047$ |
| Decision Tree | 0.8908 | $\pm 0.0020$ |
| Random Forest | 0.9227 | $\pm 0.0034$ |
| **XGBoost** | **0.9445** | **$\pm 0.0029$** |

### Held-Out Test Set Performance (6,482 Unseen Applicants)
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC | PR-AUC | True Positives | False Negatives | False Positives | True Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 81.13% | 54.73% | 79.55% | 64.85% | 0.8835 | 0.6974 | 1,128 | 290 | 933 | 4,131 |
| **Decision Tree** | 90.67% | 83.46% | 71.51% | 77.02% | 0.8970 | 0.7718 | 1,014 | 404 | 201 | 4,863 |
| **Random Forest** | 91.81% | 85.23% | 75.67% | 80.16% | 0.9297 | 0.8496 | 1,073 | 345 | 186 | 4,878 |
| **XGBoost (Champion)** | **91.79%** | **81.24%** | **81.24%** | **81.24%** | **0.9495** | **0.8885** | **1,152** | **266** | **266** | **4,798** |

---

## 11. Final Model Selection & Justification

**Champion Model:** **XGBoost Classifier** (`models/best_model.joblib`)

**Rationale:**
1. **Highest Default Detection Recall (81.24%):** Captures 1,152 out of 1,418 actual defaults, minimizing costly False Negatives.
2. **Superior Discrimination (ROC-AUC 0.9495):** Demonstrates strong ranking ability across all decision thresholds.
3. **High PR-AUC (0.8885):** Maintains precision across the recall curve.
4. **Generalization:** Consistent performance between 5-fold CV (0.9445) and test set (0.9495) with no evidence of overfitting.

---

## 12. Inference & Credit Risk Prediction Methodology

The inference engine (`src/predict.py`) maps default probabilities $P(\text{Default})$ into three risk tiers:

| Risk Category | Probability Range | Policy Recommendation | Underwriting Action |
| :--- | :---: | :--- | :--- |
| **Low Risk** | $P < 20.0\%$ | Low Default Hazard | Instant Approval / Prime Interest Rate |
| **Medium Risk** | $20.0\% \le P < 50.0\%$ | Moderate Risk Profile | Manual Review / Collateral or Co-signer Required |
| **High Risk** | $P \ge 50.0\%$ | High Default Probability | Decline / High Risk Charge-off Hazard |

*Note: Thresholds are configurable in `CreditScorer` based on institution risk appetite.*

---

## 13. Explainable AI (SHAP & Feature Attributions)

Global SHAP analysis (`outputs/evaluation_charts/shap_summary_plot.png`) identified the top default drivers:

1. **`loan_grade_D`, `loan_grade_E`, `loan_grade_F`:** Lower credit grades contribute strongly to default risk.
2. **`interest_burden_ratio` & `loan_percent_income`:** High debt service relative to income increases default probability.
3. **`person_home_ownership_RENT`:** Renters display higher default likelihood than homeowners.
4. **`income_to_loan_ratio`:** Higher income-to-loan ratios reduce default risk.
5. **`loan_intent_DEBTCONSOLIDATION` / `MEDICAL`:** Elevates risk relative to educational/venture purposes.

*Disclaimer: Feature importance reflects statistical associations within historical training data and does not constitute direct causality.*

---

## 14. Interactive Streamlit Web Application

The application (`app/app.py`) provides 5 dedicated modules:
- **Page 1 (Home & Architecture):** Project overview, executive KPI metrics, and system pipeline.
- **Page 2 (Credit Risk Prediction):** Interactive applicant input form, real-time risk gauge, calculated financial ratios, and local SHAP risk drivers.
- **Page 3 (Model Performance):** Leaderboard, interactive Confusion Matrix, ROC/PR curves, and a Decision Threshold Simulator.
- **Page 4 (Data Insights & EDA):** Exploratory data analysis covering credit grades, demographics, loan purpose, and correlations.
- **Page 5 (Explainability & Governance):** Global SHAP feature importance, ethical lending disclosures, and model monitoring guidelines.

---

## 15. Power BI Dashboard Specification & DAX Measures

The `powerbi/` directory includes 4 ready-to-import CSV files and a design specification (`DASHBOARD_SPECIFICATION.md`):
- `powerbi_applicant_predictions.csv` (Fact table: 6,482 test records with features and model outputs)
- `powerbi_model_metrics.csv` (Model benchmark table)
- `powerbi_feature_importance.csv` (Explainability table)
- `powerbi_risk_segment_summary.csv` (Aggregated portfolio summary)

Includes ready-to-use DAX measures for Total Applicants, Default Rate, Predicted Default Rate, Avg Probability, and Loan Volume at Risk.

---

## 16. Limitations & Model Governance

- **Cross-Sectional Nature:** The dataset captures point-in-time loan originations without multi-year longitudinal macroeconomic indicators.
- **Extreme Outliers:** Handled through data cleaning (`person_age > 100` and `emp_length > 60` removed).
- **Model Drift:** Ongoing monitoring via Population Stability Index (PSI) is recommended to detect macroeconomic shifts.

---

## 17. Responsible Use & Ethical AI Considerations

- **Protected Attributes:** The model avoids protected demographics (race, religion, gender, marital status) in accordance with the Equal Credit Opportunity Act (ECOA) and Fair Housing Act.
- **Human-in-the-Loop:** High and Medium risk classifications should be subject to human underwriter review.

---

## 18. Installation & How to Run

### Prerequisites
- Python 3.10+ (tested on Python 3.13)

### Installation
```bash
git clone <repository_url>
cd credit_scoring_project
pip install -r requirements.txt
```

### 1. Run the Full ML Pipeline (Preprocessing -> Training -> Evaluation -> Explainability)
```bash
python -c "
from src.preprocessing import prepare_and_save_data
from src.eda import generate_eda_visualizations
from src.train import train_models
from src.evaluate import evaluate_models
from src.explainability import explain_model

prepare_and_save_data()
generate_eda_visualizations()
train_models()
evaluate_models()
explain_model()
"
```

### 2. Run the Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```

---

## 19. Concise Project Audit

- **Dataset Used:** Authentic Kaggle Credit Risk Dataset (`credit_risk_dataset.csv`)
- **Total Records:** 32,581 Raw | 32,409 Cleaned
- **Number of Raw Features:** 11 predictors + 1 target
- **Number of Engineered Features:** 7 domain features (33 post-encoding features)
- **Target Variable:** `loan_status` (0 = Non-Default: 78.13%, 1 = Default: 21.87%)
- **Preprocessing:** Anomaly filtering, median/modal imputation, standard scaling, one-hot encoding (zero data leakage)
- **Models Trained:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Champion Model:** XGBoost Classifier
- **Actual Evaluation Metrics (Held-out Test Set, n=6,482):**
  - **Accuracy:** 91.79%
  - **ROC-AUC:** 0.9495
  - **Default Recall:** 81.24%
  - **Default Precision:** 81.24%
  - **F1-Score:** 81.24%
  - **PR-AUC:** 0.8885
- **Important Features:** `loan_grade_D`, `interest_burden_ratio`, `person_home_ownership_RENT`, `income_to_loan_ratio`, `loan_grade_A`
- **Streamlit Status:** Fully functional 5-page web application (`app/app.py`)
- **Power BI Output Status:** 4 structured CSV datasets and complete DAX specification (`powerbi/`)
- **Test Suite Status:** 9/9 Automated Tests Passing (100% success)
- **Known Limitations:** Point-in-time snapshot, requiring ongoing drift monitoring (PSI)
- **Unresolved Issues:** None
