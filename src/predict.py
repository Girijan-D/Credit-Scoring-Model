"""
Credit Risk Prediction & Inference Engine
Handles single applicant inference, batch predictions, and configurable risk categorization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from src.feature_engineering import add_engineered_features


class CreditScorer:
    """
    Production-ready Credit Scoring inference engine.
    """
    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        low_risk_threshold: float = 0.20,
        high_risk_threshold: float = 0.50
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.low_threshold = low_risk_threshold
        self.high_threshold = high_risk_threshold

    def assign_risk_category(self, prob: float) -> tuple[str, str]:
        """Maps default probability to documented business risk tiers and recommendations."""
        if prob < self.low_threshold:
            category = "Low Risk"
            recommendation = "Approve - Standard / Prime Terms (Low Default Hazard)"
        elif prob < self.high_threshold:
            category = "Medium Risk"
            recommendation = "Conditional Approval / Manual Underwriting (Additional Collateral / Higher Margin Recommended)"
        else:
            category = "High Risk"
            recommendation = "Decline / Elevated Risk (High Probability of Default)"
        return category, recommendation

    def predict_single(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts credit default probability for a single applicant.
        
        Expected fields in applicant_data:
        - person_age: int/float
        - person_income: int/float
        - person_home_ownership: 'RENT', 'OWN', 'MORTGAGE', 'OTHER'
        - person_emp_length: float
        - loan_intent: 'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'
        - loan_grade: 'A', 'B', 'C', 'D', 'E', 'F', 'G'
        - loan_amnt: int/float
        - loan_int_rate: float
        - loan_percent_income: float (optional, will be computed if missing)
        - cb_person_default_on_file: 'Y', 'N'
        - cb_person_cred_hist_length: int/float
        """
        df = pd.DataFrame([applicant_data])
        
        # Calculate loan_percent_income if not present
        if "loan_percent_income" not in df.columns or pd.isna(df["loan_percent_income"].iloc[0]):
            df["loan_percent_income"] = df["loan_amnt"] / np.maximum(df["person_income"], 1.0)
            
        # Add engineered features
        df_feat = add_engineered_features(df)
        
        # Transform features
        X_trans = self.preprocessor.transform(df_feat)
        
        # Probability of default (class 1)
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        pred_class = int(prob >= 0.50)
        
        category, recommendation = self.assign_risk_category(prob)
        
        return {
            "default_probability": round(prob, 4),
            "default_probability_pct": round(prob * 100, 2),
            "predicted_class": pred_class,
            "risk_category": category,
            "recommendation": recommendation,
            "thresholds": {
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold
            },
            "applicant_summary": {
                "income": applicant_data.get("person_income"),
                "loan_amount": applicant_data.get("loan_amnt"),
                "loan_grade": applicant_data.get("loan_grade"),
                "loan_intent": applicant_data.get("loan_intent"),
                "interest_rate": applicant_data.get("loan_int_rate"),
                "emp_length": applicant_data.get("person_emp_length"),
                "prior_default": applicant_data.get("cb_person_default_on_file")
            }
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Runs batch inference on a DataFrame of applicants."""
        df = df_batch.copy()
        if "loan_percent_income" not in df.columns:
            df["loan_percent_income"] = df["loan_amnt"] / np.maximum(df["person_income"], 1.0)
            
        df_feat = add_engineered_features(df)
        X_trans = self.preprocessor.transform(df_feat)
        
        probs = self.model.predict_proba(X_trans)[:, 1]
        df["default_probability"] = np.round(probs, 4)
        df["predicted_class"] = (probs >= 0.50).astype(int)
        
        risk_info = [self.assign_risk_category(p) for p in probs]
        df["risk_category"] = [cat for cat, _ in risk_info]
        df["recommendation"] = [rec for _, rec in risk_info]
        
        return df


def predict_sample():
    """Demonstration of single applicant prediction."""
    sample = {
        "person_age": 28,
        "person_income": 65000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 12000,
        "loan_int_rate": 11.2,
        "loan_percent_income": 0.18,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6
    }
    
    scorer = CreditScorer()
    result = scorer.predict_single(sample)
    print("Sample Applicant Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


if __name__ == "__main__":
    predict_sample()
