# Power BI Dashboard Specification: Credit Scoring & Loan Default Analytics

## 1. Dashboard Overview & Architecture

This document provides the complete architecture, data model relationships, visual layout specifications, and copy-paste ready DAX measures for building an executive-grade Power BI report based on the real outputs of the Credit Scoring Machine Learning model.

### Data Sources (Included in `powerbi/` folder):
1. `powerbi_applicant_predictions.csv` (Primary Fact Table: 6,482 test applicant records with real features, actual loan status, predicted probabilities, and risk tiers)
2. `powerbi_model_metrics.csv` (Model Evaluation Dimension Table: Comparative metrics across Logistic Regression, Decision Tree, Random Forest, and XGBoost)
3. `powerbi_feature_importance.csv` (Explainability Table: 33 features with relative importance percentages)
4. `powerbi_risk_segment_summary.csv` (Summary Table: Aggregated metrics across Low, Medium, and High Risk categories)

---

## 2. Power BI Data Model & Star Schema

### Tables & Relationships:
- **`ApplicantPredictions` (Fact Table)**: Key columns: `loan_status` (actual), `predicted_default`, `default_probability`, `risk_category`, `loan_grade`, `loan_intent`, `person_home_ownership`, `person_income`, `loan_amnt`, `loan_int_rate`.
- **`ModelMetrics` (Dimension Table)**: Key columns: `Model`, `Accuracy`, `Precision (Default)`, `Recall (Default)`, `F1-Score (Default)`, `ROC-AUC`.
- **`FeatureImportance` (Dimension Table)**: Key columns: `Feature`, `Importance`, `Relative_Importance_Pct`.
- **`RiskSummary` (Aggregated Table)**: Key columns: `risk_category`, `Applicant_Count`, `Actual_Defaults`, `Actual_Default_Rate`, `Average_Default_Probability`.

---

## 3. Comprehensive DAX Measures Library

Create a dedicated measure table `_Measures` and insert the following formulas:

```dax
// 1. Total Applicant Volume
Total Applicants = COUNTROWS('ApplicantPredictions')

// 2. Total Actual Defaults
Actual Defaults = SUM('ApplicantPredictions'[actual_default])

// 3. Baseline Actual Default Rate
Actual Default Rate = 
DIVIDE([Actual Defaults], [Total Applicants], 0)

// 4. Total Model Predicted Defaults
Predicted Defaults = SUM('ApplicantPredictions'[predicted_default])

// 5. Model Predicted Default Rate
Predicted Default Rate = 
DIVIDE([Predicted Defaults], [Total Applicants], 0)

// 6. Portfolio Average Default Probability
Avg Default Probability = 
AVERAGE('ApplicantPredictions'[default_probability])

// 7. Total Loan Exposure / Portfolio Amount ($)
Total Loan Volume = 
SUM('ApplicantPredictions'[loan_amnt])

// 8. Total Loan Volume at Risk (High Risk Band)
High Risk Loan Volume = 
CALCULATE(
    SUM('ApplicantPredictions'[loan_amnt]),
    'ApplicantPredictions'[risk_category] = "High Risk"
)

// 9. Percentage of Portfolio in High Risk Band
Pct Volume at High Risk = 
DIVIDE([High Risk Loan Volume], [Total Loan Volume], 0)

// 10. Low Risk Approval Rate
Low Risk Share = 
DIVIDE(
    CALCULATE(COUNTROWS('ApplicantPredictions'), 'ApplicantPredictions'[risk_category] = "Low Risk"),
    [Total Applicants],
    0
)

// 11. Average Applicant Annual Income
Avg Applicant Income = 
AVERAGE('ApplicantPredictions'[person_income])

// 12. Average Loan-to-Income Ratio
Avg Loan to Income = 
AVERAGE('ApplicantPredictions'[loan_percent_income])
```

---

## 4. Multi-Page Dashboard Layout Blueprint

### PAGE 1: EXECUTIVE CREDIT RISK SUMMARY
**Header / Title**: "Executive Credit Risk & Portfolio Default Analytics"

**Top KPI Cards (Row 1)**:
- **Card 1**: `Total Applicants` (6,482)
- **Card 2**: `Actual Default Rate` (21.88%)
- **Card 3**: `Avg Default Probability` (22.34%)
- **Card 4**: `Total Loan Volume` ($68.2M)
- **Card 5**: `High Risk Loan Volume` ($21.4M / 31.4%)

**Left Visuals (Column 1)**:
- **Donut Chart**: Risk Tier Breakdown (`Low Risk`, `Medium Risk`, `High Risk` by Total Applicants).
  - *Data*: Legend = `risk_category`, Values = `[Total Applicants]`.
  - *Colors*: Low Risk = Green (`#2ecc71`), Medium Risk = Yellow (`#f1c40f`), High Risk = Red (`#e74c3c`).
- **Clustered Bar Chart**: Default Rate by Loan Grade (Grades A through G).
  - *Data*: Y-Axis = `loan_grade`, X-Axis = `[Actual Default Rate]`.

**Right Visuals (Column 2)**:
- **Clustered Column Chart**: Default Rate by Loan Purpose / Intent.
  - *Data*: X-Axis = `loan_intent`, Y-Axis = `[Actual Default Rate]`.
- **Clustered Bar Chart**: Default Rate by Home Ownership Status.
  - *Data*: Y-Axis = `person_home_ownership`, X-Axis = `[Actual Default Rate]`.

**Top Slicers Panel**:
- `loan_grade` (Dropdown multi-select)
- `loan_intent` (Tile / button multi-select)
- `person_home_ownership` (Dropdown multi-select)
- `cb_person_default_on_file` (Radio button: Y / N)

---

### PAGE 2: MACHINE LEARNING MODEL DIAGNOSTICS & EXPLAINABILITY
**Header / Title**: "Model Performance Benchmarking & Explainable AI (XAI)"

**Top KPI Cards**:
- **Champion Model**: "XGBoost Classifier"
- **Champion ROC-AUC**: "0.9495"
- **Default Recall**: "81.24%"
- **Default Precision**: "81.24%"
- **Model Accuracy**: "91.79%"

**Main Visuals**:
- **Table Visual**: Model Comparison Leaderboard.
  - *Columns*: `Model`, `Accuracy`, `Recall (Default)`, `Precision (Default)`, `F1-Score (Default)`, `ROC-AUC`.
- **Horizontal Bar Chart**: Top 15 Most Influential Features (`FeatureImportance` table).
  - *Data*: Y-Axis = `Feature`, X-Axis = `Relative_Importance_Pct`.
  - *Sorting*: Descending by Relative Importance.
- **Matrix Visual**: Confusion Matrix representation showing True Negatives (4,868), False Positives (196), False Negatives (266), True Positives (1,152).

---

## 5. Step-by-Step Instructions to Build in Power BI Desktop

1. Open Power BI Desktop and select **Get Data -> Text/CSV**.
2. Load all 4 CSV files from the `powerbi/` directory:
   - `powerbi_applicant_predictions.csv`
   - `powerbi_model_metrics.csv`
   - `powerbi_feature_importance.csv`
   - `powerbi_risk_segment_summary.csv`
3. In the Model View, verify that the tables are loaded cleanly.
4. Create a **New Table** called `_Measures` and paste the DAX code from Section 3.
5. Create Page 1 ("Executive Credit Risk Summary") and Page 2 ("Model Diagnostics & XAI") following the layout blueprints in Section 4.
6. Apply theme colors: Dark Blue (`#1e293b`), Emerald Green (`#10b981`), Amber (`#f59e0b`), Crimson (`#ef4444`).
7. Save the report as `Credit_Scoring_Dashboard.pbix`.
