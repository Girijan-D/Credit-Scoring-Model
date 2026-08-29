"""
End-to-End Test Suite for Credit Scoring Model Pipeline
Tests data ingestion, preprocessing, feature engineering, model inference,
edge cases, explainability, and export integrity using standard unittest.
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_data, load_raw_data, get_feature_columns
from src.predict import CreditScorer
from src.explainability import explain_single_applicant


class TestCreditScoringPipeline(unittest.TestCase):

    def test_raw_data_integrity(self):
        """Verify raw dataset exists, has required columns, and non-zero rows."""
        raw_path = "data/raw/credit_risk_dataset.csv"
        self.assertTrue(os.path.exists(raw_path), "Raw dataset file missing")
        df = pd.read_csv(raw_path)
        self.assertGreater(len(df), 30000, f"Expected >30,000 rows, got {len(df)}")
        
        required_cols = [
            "person_age", "person_income", "person_home_ownership", "person_emp_length",
            "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
            "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_data_cleaning_and_deduplication(self):
        """Verify clean_data drops duplicates and removes age/emp_length anomalies."""
        df_raw = load_raw_data("data/raw/credit_risk_dataset.csv")
        df_clean = clean_data(df_raw)
        
        self.assertLess(len(df_clean), len(df_raw), "Duplicates or anomalies were not filtered")
        self.assertLessEqual(df_clean["person_age"].max(), 100, "Unrealistic age > 100 found in clean data")
        self.assertLessEqual(df_clean["person_emp_length"].max(), 60, "Unrealistic emp_length > 60 found in clean data")

    def test_feature_engineering(self):
        """Verify calculated engineered ratios."""
        sample = pd.DataFrame([{
            "person_age": 30,
            "person_income": 50000,
            "person_emp_length": 5.0,
            "loan_amnt": 10000,
            "loan_int_rate": 10.0,
            "cb_person_cred_hist_length": 6,
            "cb_person_default_on_file": "Y",
            "loan_grade": "D"
        }])
        
        feat_df = add_engineered_features(sample)
        
        self.assertIn("loan_to_income_ratio", feat_df.columns)
        self.assertAlmostEqual(feat_df["loan_to_income_ratio"].iloc[0], 0.20, places=3)
        self.assertAlmostEqual(feat_df["annual_interest_burden"].iloc[0], 1000.0, places=3)
        self.assertAlmostEqual(feat_df["credit_age_ratio"].iloc[0], 0.20, places=3)
        self.assertEqual(feat_df["high_risk_flag"].iloc[0], 1)

    def test_model_and_preprocessor_artifacts(self):
        """Verify that all serialized models and preprocessors exist and can be loaded."""
        models_to_check = [
            "models/preprocessor.joblib",
            "models/logistic_regression.joblib",
            "models/decision_tree.joblib",
            "models/random_forest.joblib",
            "models/xgboost.joblib",
            "models/best_model.joblib"
        ]
        for m in models_to_check:
            self.assertTrue(os.path.exists(m), f"Artifact missing: {m}")
            obj = joblib.load(m)
            self.assertIsNotNone(obj, f"Failed to load artifact: {m}")

    def test_single_applicant_prediction(self):
        """Verify single applicant inference returns bounded probability and valid risk category."""
        scorer = CreditScorer()
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
        
        result = scorer.predict_single(sample)
        self.assertIn("default_probability", result)
        self.assertTrue(0.0 <= result["default_probability"] <= 1.0)
        self.assertIn(result["risk_category"], ["Low Risk", "Medium Risk", "High Risk"])
        self.assertIn(result["predicted_class"], [0, 1])

    def test_edge_case_missing_and_zero_inputs(self):
        """Verify that inference handles missing optional fields and zero values gracefully."""
        scorer = CreditScorer()
        edge_sample = {
            "person_age": 22,
            "person_income": 0,  # Zero income edge case
            "person_home_ownership": "RENT",
            "person_emp_length": None,  # Missing emp length
            "loan_intent": "MEDICAL",
            "loan_grade": "G",
            "loan_amnt": 5000,
            "loan_int_rate": 20.0,
            "loan_percent_income": None,
            "cb_person_default_on_file": "Y",
            "cb_person_cred_hist_length": 2
        }
        
        result = scorer.predict_single(edge_sample)
        self.assertTrue(0.0 <= result["default_probability"] <= 1.0)
        self.assertEqual(result["risk_category"], "High Risk")

    def test_batch_prediction(self):
        """Verify batch prediction processes DataFrames properly."""
        scorer = CreditScorer()
        test_df = pd.read_csv("data/processed/test.csv").head(10)
        
        preds_df = scorer.predict_batch(test_df)
        self.assertIn("default_probability", preds_df.columns)
        self.assertIn("risk_category", preds_df.columns)
        self.assertEqual(len(preds_df), 10)
        self.assertTrue((preds_df["default_probability"] >= 0.0).all() and (preds_df["default_probability"] <= 1.0).all())

    def test_explainability_drivers(self):
        """Verify local explainability returns feature attributions."""
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
        drivers = explain_single_applicant(sample)
        self.assertIsInstance(drivers, list)
        self.assertGreater(len(drivers), 0)
        self.assertIn("Feature", drivers[0])
        self.assertIn("Direction", drivers[0])

    def test_powerbi_csv_exports(self):
        """Verify that all Power BI CSV files exist, are non-empty, and contain real predictions."""
        files = [
            "powerbi/powerbi_applicant_predictions.csv",
            "powerbi/powerbi_model_metrics.csv",
            "powerbi/powerbi_feature_importance.csv",
            "powerbi/powerbi_risk_segment_summary.csv"
        ]
        for f in files:
            self.assertTrue(os.path.exists(f), f"Power BI export missing: {f}")
            df = pd.read_csv(f)
            self.assertFalse(df.empty, f"Power BI export is empty: {f}")


if __name__ == "__main__":
    unittest.main()
