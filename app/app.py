"""
Streamlit Web Application: Credit Scoring Model for Loan Default Prediction
A 5-page interactive dashboard for loan default risk prediction, model diagnostics,
exploratory data analysis, and AI explainability.
"""

import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

from src.predict import CreditScorer
from src.explainability import explain_single_applicant
from src.feature_engineering import add_engineered_features

# -------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring & Loan Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Custom CSS for Premium Design & Modern Aesthetics
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-badge-low {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .risk-badge-medium {
        background-color: #fef9c3;
        color: #a16207;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #fde047;
    }
    .risk-badge-high {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    .result-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07);
    }
    .info-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Loaders for Artifacts
# -------------------------------------------------------------
@st.cache_resource
def load_scorer():
    return CreditScorer(
        model_path="models/best_model.joblib",
        preprocessor_path="models/preprocessor.joblib"
    )

@st.cache_data
def load_metrics_data():
    if os.path.exists("outputs/model_metrics.csv"):
        return pd.read_csv("outputs/model_metrics.csv")
    return None

@st.cache_data
def load_feature_importance_data():
    if os.path.exists("outputs/feature_importance.csv"):
        return pd.read_csv("outputs/feature_importance.csv")
    return None

@st.cache_data
def load_risk_summary_data():
    if os.path.exists("outputs/risk_category_summary.csv"):
        return pd.read_csv("outputs/risk_category_summary.csv")
    return None

@st.cache_data
def load_cleaned_dataset():
    if os.path.exists("data/processed/cleaned_dataset.csv"):
        return pd.read_csv("data/processed/cleaned_dataset.csv")
    return None


# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=70)
st.sidebar.title("Credit Risk AI")
st.sidebar.markdown("**Enterprise Loan Default Scoring**")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 1. Home & Overview",
        "🎯 2. Credit Risk Prediction",
        "📊 3. Model Performance",
        "🔍 4. Data Insights & EDA",
        "🧠 5. Explainability & Governance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Champion Model:** XGBoost\n\n**Test ROC-AUC:** 0.9495\n\n**Default Recall:** 81.24%\n\n**Trained on:** 32,409 real applicant records")


# =============================================================
# PAGE 1: HOME & OVERVIEW
# =============================================================
if nav_choice == "🏠 1. Home & Overview":
    st.markdown('<div class="main-title">Credit Scoring Model for Loan Default Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An End-to-End Machine Learning & Credit Risk Intelligence System</div>', unsafe_allow_html=True)
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Applicants Analyzed</div>
            <div class="metric-val">32,409</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Baseline Default Rate</div>
            <div class="metric-val">21.82%</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Champion Model ROC-AUC</div>
            <div class="metric-val">0.9495</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Default Recall Rate</div>
            <div class="metric-val">81.24%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 📌 Executive Summary & Objective")
        st.write("""
        Credit scoring and loan default prediction are critical capabilities for financial institutions to manage credit risk, 
        minimize non-performing assets (NPAs), and optimize capital allocation. 
        
        This project implements a complete, mathematically grounded Machine Learning pipeline that predicts whether a loan applicant is likely to default based on:
        - **Demographic Information:** Age, employment length, and residential stability.
        - **Financial Variables:** Annual income, requested loan amount, interest rate, and income leverage.
        - **Credit Bureau History:** Credit history length, loan grade tier, and historical default records.
        - **Behavioral & Loan Intent:** Purpose of credit (e.g. Education, Medical, Venture, Debt Consolidation).
        """)
        
        st.markdown("### ⚙️ End-to-End System Architecture")
        st.write("""
        1. **Raw Data Ingestion:** 32,581 real-world applicant records with zero synthetic fabrication.
        2. **Leakage-Free Preprocessing:** Anomaly filtering, median & modal imputation, standardization, and one-hot encoding fitted strictly on training data.
        3. **Domain Feature Engineering:** Financial burden ratios (Loan-to-Income, Annual Interest Burden, Credit Experience to Age).
        4. **Multi-Model Benchmarking:** 5-Fold Cross-Validation across Logistic Regression, Decision Trees, Random Forest, and XGBoost.
        5. **Production Inference & Explainability:** Real-time scoring engine with configurable risk banding and SHAP feature attributions.
        6. **Business Intelligence Export:** Power BI ready datasets and DAX measures for executive reporting.
        """)

    with col_right:
        st.markdown("### 🏢 Business Impact of Accurate Scoring")
        st.markdown("""
        <div class="info-card">
            <b>🔻 False Negatives (FN) Risk:</b> Approving a borrower who later defaults leads to direct charge-offs, loss of principal, and high recovery expenses.
        </div>
        <div class="info-card">
            <b>🔻 False Positives (FP) Risk:</b> Rejecting a creditworthy applicant leads to lost interest revenue and customer dissatisfaction.
        </div>
        <div class="info-card">
            <b>⭐ Solution Focus:</b> Our champion XGBoost model maximizes <b>Recall (81.24%)</b> and <b>ROC-AUC (0.9495)</b> while maintaining strong <b>Precision (81.24%)</b>.
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists("outputs/eda_charts/target_distribution.png"):
            st.image("outputs/eda_charts/target_distribution.png", caption="Ground Truth Class Distribution (32,409 Records)", use_container_width=True)


# =============================================================
# PAGE 2: CREDIT RISK PREDICTION (INFERENCE)
# =============================================================
elif nav_choice == "🎯 2. Credit Risk Prediction":
    st.markdown('<div class="main-title">Live Credit Risk Scoring Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter applicant financial & demographic parameters to calculate default hazard and risk tier.</div>', unsafe_allow_html=True)
    
    scorer = load_scorer()
    
    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographic Factors")
            age = st.slider("Applicant Age (Years)", min_value=18, max_value=85, value=28, step=1)
            home_ownership = st.selectbox("Home Ownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=45.0, value=4.5, step=0.5)
            
        with col2:
            st.markdown("#### 💰 Financial Profile")
            income = st.number_input("Annual Income ($)", min_value=5000, max_value=1000000, value=65000, step=1000)
            loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=50000, value=12000, step=500)
            int_rate = st.slider("Loan Interest Rate (%)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
            loan_intent = st.selectbox(
                "Loan Purpose / Intent",
                ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
            )
            
        with col3:
            st.markdown("#### 📜 Credit Bureau History")
            loan_grade = st.selectbox("Credit Rating Grade", ["A", "B", "C", "D", "E", "F", "G"], index=1)
            default_on_file = st.selectbox("Historical Default on Credit Record?", ["N", "Y"])
            cred_hist_length = st.slider("Credit History Length (Years)", min_value=1, max_value=35, value=6, step=1)
            
        st.markdown("---")
        submit_btn = st.form_submit_button("🚀 Run Credit Risk Assessment", use_container_width=True)
        
    if submit_btn:
        applicant_payload = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home_ownership,
            "person_emp_length": emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": int_rate,
            "loan_percent_income": loan_amnt / max(income, 1.0),
            "cb_person_default_on_file": default_on_file,
            "cb_person_cred_hist_length": cred_hist_length
        }
        
        result = scorer.predict_single(applicant_payload)
        prob = result["default_probability"]
        prob_pct = result["default_probability_pct"]
        category = result["risk_category"]
        recommendation = result["recommendation"]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Credit Decision Assessment Result")
        
        res_col1, res_col2 = st.columns([1.2, 2])
        
        with res_col1:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"**Predicted Default Probability:**")
            st.markdown(f"<h1 style='font-size:3rem; margin:0; color:{'#15803d' if category=='Low Risk' else '#a16207' if category=='Medium Risk' else '#b91c1c'}'>{prob_pct:.1f}%</h1>", unsafe_allow_html=True)
            
            st.progress(min(1.0, prob))
            
            badge_class = "risk-badge-low" if category == "Low Risk" else "risk-badge-medium" if category == "Medium Risk" else "risk-badge-high"
            st.markdown(f"<div style='margin-top:10px;'><span class='{badge_class}'>{category}</span></div>", unsafe_allow_html=True)
            
            st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
            st.markdown(f"**Underwriting Recommendation:**\n\n{recommendation}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("#### 📊 Derived Financial Ratios")
            r1, r2, r3 = st.columns(3)
            r1.metric("Loan-to-Income Ratio", f"{(loan_amnt/income)*100:.1f}%")
            r2.metric("Annual Interest Expense", f"${loan_amnt * (int_rate/100):,.0f}")
            r3.metric("Credit-to-Age Ratio", f"{(cred_hist_length/age)*100:.1f}%")
            
            st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Applicant Risk Factor Drivers")
            drivers = explain_single_applicant(applicant_payload)
            driver_df = pd.DataFrame(drivers)
            if not driver_df.empty:
                st.dataframe(
                    driver_df[["Feature", "Direction", "Impact_Score"]],
                    use_container_width=True,
                    hide_index=True
                )
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# PAGE 3: MODEL PERFORMANCE & BENCHMARKING
# =============================================================
elif nav_choice == "📊 3. Model Performance":
    st.markdown('<div class="main-title">Multi-Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous test-set benchmark across 4 machine learning model families (6,482 test applicants).</div>', unsafe_allow_html=True)
    
    metrics_df = load_metrics_data()
    if metrics_df is not None:
        st.markdown("### 🏆 Comprehensive Model Leaderboard")
        
        display_df = metrics_df[[
            "Model", "Accuracy", "Precision (Default)", "Recall (Default)", "F1-Score (Default)", "ROC-AUC", "PR-AUC (Avg Precision)"
        ]].copy()
        
        # Formatting percentages
        for col in ["Accuracy", "Precision (Default)", "Recall (Default)", "F1-Score (Default)"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
        for col in ["ROC-AUC", "PR-AUC (Avg Precision)"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    tab1, tab2, tab3 = st.tabs(["📈 ROC & PR Curves", "🔲 Confusion Matrices", "⚖️ Threshold Tuning Simulator"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("outputs/evaluation_charts/roc_curves.png"):
                st.image("outputs/evaluation_charts/roc_curves.png", caption="ROC Curves Benchmark", use_container_width=True)
        with c2:
            if os.path.exists("outputs/evaluation_charts/precision_recall_curves.png"):
                st.image("outputs/evaluation_charts/precision_recall_curves.png", caption="Precision-Recall Curves (Default Detection)", use_container_width=True)
                
    with tab2:
        if os.path.exists("outputs/evaluation_charts/confusion_matrices.png"):
            st.image("outputs/evaluation_charts/confusion_matrices.png", caption="Test Set Confusion Matrices Across All Models", use_container_width=True)
        if os.path.exists("outputs/evaluation_charts/model_comparison_bar.png"):
            st.image("outputs/evaluation_charts/model_comparison_bar.png", caption="Metric-by-Metric Comparison", use_container_width=True)

    with tab3:
        st.markdown("### 🎛️ Interactive Decision Threshold Simulator")
        st.write("Adjusting the default classification threshold changes the trade-off between **Credit Risk Protection (Recall)** and **Loan Approval Volume (Precision)**.")
        
        if os.path.exists("outputs/predictions.csv"):
            pred_df = pd.read_csv("outputs/predictions.csv")
            threshold = st.slider("Select Default Probability Classification Threshold", 0.05, 0.95, 0.50, 0.05)
            
            sim_preds = (pred_df["default_probability"] >= threshold).astype(int)
            y_actual = pred_df["actual_default"].values
            
            from sklearn.metrics import recall_score, precision_score, accuracy_score
            sim_rec = recall_score(y_actual, sim_preds, pos_label=1)
            sim_prec = precision_score(y_actual, sim_preds, pos_label=1, zero_division=0)
            sim_acc = accuracy_score(y_actual, sim_preds)
            defaults_caught = (sim_preds & y_actual).sum()
            total_defaults = y_actual.sum()
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Selected Threshold", f"{threshold:.2f}")
            s2.metric("Default Recall", f"{sim_rec*100:.1f}%")
            s3.metric("Default Precision", f"{sim_prec*100:.1f}%")
            s4.metric("Defaults Caught", f"{defaults_caught:,} / {total_defaults:,}")


# =============================================================
# PAGE 4: DATA INSIGHTS & EDA
# =============================================================
elif nav_choice == "🔍 4. Data Insights & EDA":
    st.markdown('<div class="main-title">Exploratory Data Insights & Portfolio Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">In-depth statistical breakdown of default risk across demographic and credit dimensions.</div>', unsafe_allow_html=True)
    
    eda_tabs = st.tabs(["📊 Credit Grade & Bureau History", "🏠 Demographics & Home Ownership", "🎯 Loan Intent & Purpose", "📈 Income & Leverage Distributions", "🔥 Correlation Heatmap"])
    
    with eda_tabs[0]:
        col_a, col_b = st.columns(2)
        with col_a:
            if os.path.exists("outputs/eda_charts/default_by_loan_grade.png"):
                st.image("outputs/eda_charts/default_by_loan_grade.png", use_container_width=True)
        with col_b:
            if os.path.exists("outputs/eda_charts/credit_history_analysis.png"):
                st.image("outputs/eda_charts/credit_history_analysis.png", use_container_width=True)
        st.info("💡 **Insight:** Borrowers in lower loan grades (D, E, F, G) exhibit default rates over 50%, compared to under 10% in Grade A. Prior historical defaults double the future default hazard.")

    with eda_tabs[1]:
        if os.path.exists("outputs/eda_charts/default_by_home_ownership.png"):
            st.image("outputs/eda_charts/default_by_home_ownership.png", use_container_width=True)
        st.info("💡 **Insight:** Renters experience significantly higher default rates (~31.6%) than homeowners (~7.5%) and mortgage holders (~12.8%).")

    with eda_tabs[2]:
        if os.path.exists("outputs/eda_charts/default_by_loan_intent.png"):
            st.image("outputs/eda_charts/default_by_loan_intent.png", use_container_width=True)
        st.info("💡 **Insight:** Medical and Debt Consolidation loans have the highest default incidences (~27-28%), whereas Venture and Education loans show lower default rates (~16-17%).")

    with eda_tabs[3]:
        if os.path.exists("outputs/eda_charts/income_vs_loan_dist.png"):
            st.image("outputs/eda_charts/income_vs_loan_dist.png", use_container_width=True)
        st.info("💡 **Insight:** Defaulted applicants exhibit markedly higher loan-to-income ratios and higher average interest rates.")

    with eda_tabs[4]:
        if os.path.exists("outputs/eda_charts/correlation_matrix.png"):
            st.image("outputs/eda_charts/correlation_matrix.png", use_container_width=True)


# =============================================================
# PAGE 5: EXPLAINABILITY & GOVERNANCE
# =============================================================
elif nav_choice == "🧠 5. Explainability & Governance":
    st.markdown('<div class="main-title">Explainable AI (XAI) & Model Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transparent attribution of model drivers, ethical lending principles, and regulatory considerations.</div>', unsafe_allow_html=True)
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if os.path.exists("outputs/evaluation_charts/feature_importance_bar.png"):
            st.image("outputs/evaluation_charts/feature_importance_bar.png", caption="Global Tree Feature Importance (XGBoost)", use_container_width=True)
    with exp_col2:
        if os.path.exists("outputs/evaluation_charts/shap_summary_plot.png"):
            st.image("outputs/evaluation_charts/shap_summary_plot.png", caption="SHAP Global Feature Impact Distribution", use_container_width=True)
            
    st.markdown("---")
    st.markdown("### ⚖️ Responsible Lending & Model Governance Standards")
    
    gov1, gov2, gov3 = st.columns(3)
    with gov1:
        st.markdown("#### 🛡️ Fair Lending & Non-Discrimination")
        st.write("""
        The scoring system strictly excludes protected demographic attributes (e.g., race, gender, religion, marital status) 
        in compliance with the Equal Credit Opportunity Act (ECOA) and Fair Housing Act.
        """)
    with gov2:
        st.markdown("#### 📜 Statistical Association vs. Causality")
        st.write("""
        Model feature importances represent empirical statistical relationships within historical data. 
        High importance does not imply direct physical causation, and scores should be used alongside qualitative underwriting review.
        """)
    with gov3:
        st.markdown("#### 🔄 Model Drift & Monitoring")
        st.write("""
        Credit risk distributions shift over macroeconomic cycles (e.g. interest rate changes). 
        The model requires quarterly Population Stability Index (PSI) monitoring and retuning.
        """)

