# Implementation Plan - Credit Scoring Model for Loan Default Prediction

An end-to-end Machine Learning and Business Intelligence solution to assess loan default risk using real-world applicant data, predictive modeling, explainable AI, interactive Streamlit web deployment, and Power BI reporting assets.

---

## User Review Required

> [!IMPORTANT]
> **Dataset Selection**: We will use the authentic **Kaggle Credit Risk Dataset** (`credit_risk_dataset.csv`), containing **32,581 real loan applicant records** with 12 baseline features (demographics, financials, credit history, loan purpose) and ground-truth `loan_status` (default / non-default).
> All numbers, metrics, charts, predictions, and Power BI exports will be derived strictly from real executions—zero synthetic or hardcoded values.

> [!NOTE]
> **Environment Compatibility**: The environment has Python 3.13 with `scikit-learn`, `pandas`, `numpy`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `shap`, and `streamlit` installed.

---

## Proposed Architecture & Directory Layout

```
d:/BA Project/
│
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv          # Untouched authentic dataset (32,581 rows)
│   └── processed/
│       ├── train.csv                        # Cleaned training split (80%)
│       ├── test.csv                         # Cleaned testing split (20%)
│       └── cleaned_dataset.csv              # Cleaned full dataset with engineered features
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # Cleaning, outlier filtering, imputation, encoding, leakage-free pipelines
│   ├── eda.py                              # Visualizations, statistical summaries, correlation, cross-tabulations
│   ├── feature_engineering.py              # Domain-specific credit risk ratios & transformations
│   ├── train.py                            # Multi-model training (LR, DT, RF, XGBoost) with 5-fold CV & hyperparameter tuning
│   ├── evaluate.py                         # Comprehensive metrics (ROC-AUC, F1, Recall, Precision, Confusion Matrix)
│   ├── predict.py                          # Inference engine with configurable risk banding & batch processing
│   └── explainability.py                   # SHAP value extraction, global importance & local applicant waterfall attribution
│
├── models/
│   ├── preprocessor.joblib                  # Fitted preprocessing transformer
│   ├── logistic_regression.joblib          # Trained Logistic Regression model
│   ├── decision_tree.joblib                # Trained Decision Tree model
│   ├── random_forest.joblib                # Trained Random Forest model
│   ├── xgboost_model.joblib                # Trained XGBoost model
│   └── best_model.joblib                   # Top-performing champion model
│
├── outputs/
│   ├── predictions.csv                     # Test set predictions with default probabilities & risk bands
│   ├── model_metrics.csv                   # Comparative performance matrix across all models
│   ├── feature_importance.csv              # Global feature importances
│   ├── risk_category_summary.csv           # Segmented risk analysis for BI
│   ├── eda_charts/                         # Saved high-res EDA charts (PNG)
│   │   ├── target_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── default_by_loan_grade.png
│   │   ├── default_by_home_ownership.png
│   │   ├── default_by_loan_intent.png
│   │   └── income_vs_loan_dist.png
│   └── evaluation_charts/                  # Model diagnostic plots (PNG)
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── precision_recall_curves.png
│       └── model_comparison_bar.png
│
├── app/
│   └── app.py                              # 5-Page interactive Streamlit Web Application
│
├── powerbi/
│   ├── powerbi_applicant_predictions.csv   # Power BI primary table (Real predictions + applicant features)
│   ├── powerbi_model_metrics.csv           # Power BI model metrics table
│   ├── powerbi_feature_importance.csv      # Power BI feature importance table
│   ├── powerbi_risk_segment_summary.csv    # Power BI portfolio summary table
│   └── DASHBOARD_SPECIFICATION.md          # Full Power BI implementation guide, DAX formulas & layout instructions
│
├── tests/
│   └── test_pipeline.py                    # Automated end-to-end unit & integration tests
│
├── notebooks/
│   └── credit_scoring_analysis.ipynb       # Documented research & demonstration notebook
│
├── requirements.txt                        # Complete dependency manifest
└── README.md                               # Academic/Enterprise documentation
```

---

## Detailed Execution Plan by Phase

### Phase 1: Workspace & Dataset Acquisition
- Download and verify authentic `credit_risk_dataset.csv` into `data/raw/`.
- Inspect data schema:
  - 12 raw columns: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_status` (target), `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.
  - Check missing values (`person_emp_length`: 895, `loan_int_rate`: 3116).
  - Check duplicates (165 rows).
  - Check invalid ranges (`person_age > 100`, `person_emp_length > 60`).

### Phase 2: Data Preprocessing Pipeline
- Deduplicate records while keeping raw intact.
- Handle realistic anomalies (filter erroneous age $> 100$ and employment $> 60$).
- Stratified 80/20 train-test split (`random_state=42`, `stratify=loan_status`).
- Build `ColumnTransformer` / `Pipeline` fitted **strictly on training data**:
  - Numerical features: Median Imputation + `StandardScaler` / `RobustScaler`.
  - Categorical features: Most frequent Imputation + `OneHotEncoder(handle_unknown='ignore', drop='first')`.
  - Ordinal features (e.g. `loan_grade` A-G): explicit ordinal mapping.
- Save fitted `preprocessor.joblib`.

### Phase 3: Exploratory Data Analysis (EDA)
- Script `src/eda.py` generating and saving:
  1. Target class distribution (Imbalance analysis: ~78% non-default, ~22% default).
  2. Numerical feature distributions and outlier boxplots.
  3. Correlation matrix heatmap.
  4. Default rate across `loan_grade`, `person_home_ownership`, `loan_intent`, and `cb_person_default_on_file`.
  5. Income vs Loan Amount scatter/KDE segmented by default status.
  6. Credit history length impact on default likelihood.

### Phase 4: Feature Engineering
Implement robust domain features in `src/feature_engineering.py`:
1. `loan_to_income_ratio`: $\text{loan\_amnt} / \max(\text{person\_income}, 1)$
2. `income_to_loan_ratio`: $\text{person\_income} / \max(\text{loan\_amnt}, 1)$
3. `annual_interest_burden`: $\text{loan\_amnt} \times (\text{loan\_int_rate} / 100)$
4. `credit_age_ratio`: $\text{cb\_person\_cred\_hist\_length} / \max(\text{person\_age}, 1)$
5. `emp_to_age_ratio`: $\text{person\_emp\_length} / \max(\text{person\_age}, 1)$
6. `high_risk_flag`: Binary indicator for historical default + grade in (D, E, F, G).

### Phase 5 & 6: Multi-Model Training & Evaluation
Train 4 distinct model families:
1. **Logistic Regression** (`class_weight='balanced'`, L2 regularization tuning).
2. **Decision Tree Classifier** (`class_weight='balanced'`, tuned `max_depth`, `min_samples_split`).
3. **Random Forest Classifier** (`n_estimators=200`, `class_weight='balanced'`, tuned `max_depth`, `min_samples_leaf`).
4. **XGBoost Classifier** (`scale_pos_weight`, tuned `n_estimators`, `learning_rate`, `max_depth`, `subsample`).

- 5-Fold Stratified Cross-Validation on training data.
- Complete evaluation metrics on test set:
  - Accuracy, Precision (Class 1), Recall (Class 1), F1-Score (Class 1), ROC-AUC, PR-AUC / Average Precision.
  - Confusion Matrix analysis (Financial trade-off: False Positives vs False Negatives).
- Plot and save ROC Curves, Confusion Matrices, and PR Curves to `outputs/evaluation_charts/`.

### Phase 7: Model Selection & Artifact Persistence
- Champion model selection with trade-off analysis (ROC-AUC vs Recall for default detection).
- Serialize all models and champion model to `models/`.

### Phase 8 & 9: Inference System & Explainability
- `src/predict.py`: Real-time single applicant prediction & batch inference.
  - Return: Default Probability (0.00% to 100.00%) & Configurable Risk Band:
    - **Low Risk**: $P(\text{Default}) < 0.20$ (Green - Approval Recommended)
    - **Medium Risk**: $0.20 \le P(\text{Default}) < 0.50$ (Yellow - Manual Review / Stricter Terms)
    - **High Risk**: $P(\text{Default}) \ge 0.50$ (Red - High Default Risk / Rejection Recommended)
- `src/explainability.py`:
  - Global Feature Importance extraction.
  - Local individual applicant SHAP attribution (top risk-increasing and risk-decreasing factors).

### Phase 10: Multi-Page Streamlit Web Application (`app/app.py`)
Craft an interactive, professional web application with 5 comprehensive pages:
- **Page 1: Overview & System Architecture**: Problem definition, KPI summary cards, pipeline architecture.
- **Page 2: Live Credit Risk Assessment**: Interactive sliders/dropdowns for applicant demographics & financials, real-time risk gauge, calculated probability, risk band, business recommendation, and personalized feature impact breakdown.
- **Page 3: Model Performance & Benchmarking**: Model comparison table, interactive Confusion Matrix, ROC curves, and decision threshold simulator.
- **Page 4: Exploratory Data Insights**: Interactive data distributions, bivariate risk filters, correlation explorer.
- **Page 5: Explainability & Responsible AI**: Global SHAP importance, model governance, ethical lending considerations, fairness and limitation disclosures.

### Phase 11 & 12: Power BI Output Generation & Dashboard Design
- Export clean, real data files in `powerbi/`:
  - `powerbi_applicant_predictions.csv` (Applicant attributes + default probabilities + risk category + predicted class).
  - `powerbi_model_metrics.csv` (Accuracy, Precision, Recall, F1, ROC-AUC across all models).
  - `powerbi_feature_importance.csv` (Features and normalized weights).
  - `powerbi_risk_segment_summary.csv` (Aggregations by loan intent and risk category).
- Create `powerbi/DASHBOARD_SPECIFICATION.md`:
  - KPI cards layout (Total Applicants, Default Rate, Avg Probability, High Risk Volume).
  - Visual matrix, chart configuration, slicers (Loan Intent, Home Ownership, Risk Category).
  - Comprehensive DAX measures code block.

### Phase 13: End-to-End Testing
- Write and execute unit/integration test suite `tests/test_pipeline.py` verifying:
  1. Data integrity and shapes.
  2. Preprocessing pipeline transformation.
  3. Model training and persistence.
  4. Real-time inference output validity (probability bounds $[0, 1]$, valid risk band).
  5. Missing/invalid input handling.
  6. Power BI CSV consistency.

### Phase 14: Documentation & Final Audit
- Create comprehensive `README.md` covering all 18 required sections.
- Compile final concise Project Audit with exact numbers from runtime execution.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_pipeline.py` or `python tests/test_pipeline.py`.
- Verify code syntax and execution of all pipeline stages (`train.py`, `evaluate.py`, `predict.py`, `eda.py`, `explainability.py`).

### Streamlit Application Verification
- Test Streamlit app launching via background command and verify smooth page rendering.

### Output Artifact Verification
- Verify all CSV files (`predictions.csv`, `model_metrics.csv`, `feature_importance.csv`, `powerbi_*.csv`) and PNG charts are correctly generated and populated with non-zero, real data.
