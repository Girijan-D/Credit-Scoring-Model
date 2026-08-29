"""
Data Preprocessing Pipeline for Credit Scoring Model
Handles data loading, cleaning, anomaly filtering, leakage-free transformation, and persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.feature_engineering import add_engineered_features


def load_raw_data(filepath: str = "data/raw/credit_risk_dataset.csv") -> pd.DataFrame:
    """Loads the original raw dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by removing exact duplicates and filtering out 
    unrealistic data entry anomalies (e.g. age > 100 or employment length > 60).
    """
    df_clean = df.copy()
    
    # 1. Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    dedup_count = initial_rows - len(df_clean)
    
    # 2. Filter realistic anomalies
    # Filter age > 100 (data entry errors like 144)
    df_clean = df_clean[df_clean["person_age"] <= 100].reset_index(drop=True)
    
    # Filter employment length > 60 or employment length > age
    df_clean = df_clean[
        (df_clean["person_emp_length"].isna()) | 
        ((df_clean["person_emp_length"] <= 60) & (df_clean["person_emp_length"] < df_clean["person_age"]))
    ].reset_index(drop=True)
    
    # Ensure proper data types
    df_clean["loan_status"] = df_clean["loan_status"].astype(int)
    
    return df_clean


def get_feature_columns():
    """Returns numerical and categorical feature column definitions."""
    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "loan_to_income_ratio",
        "income_to_loan_ratio",
        "annual_interest_burden",
        "interest_burden_ratio",
        "credit_age_ratio",
        "emp_to_age_ratio",
        "high_risk_flag"
    ]
    
    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]
    
    return numeric_cols, categorical_cols


def build_preprocessor_pipeline():
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    Uses median imputation + StandardScaler for numeric features,
    and most frequent imputation + OneHotEncoder for categoricals.
    """
    numeric_cols, categorical_cols = get_feature_columns()
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor


def get_feature_names_out(preprocessor, numeric_cols, categorical_cols):
    """Extracts explicit feature names after ColumnTransformer preprocessing."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    return numeric_cols + cat_feature_names


def prepare_and_save_data(
    raw_path: str = "data/raw/credit_risk_dataset.csv",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Orchestrates data loading, cleaning, feature engineering, stratified splitting,
    fitting the preprocessor strictly on train data, and saving artifacts.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load & Clean
    raw_df = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)
    
    # 2. Add Engineered Features
    full_df = add_engineered_features(clean_df)
    full_df.to_csv(os.path.join(processed_dir, "cleaned_dataset.csv"), index=False)
    
    # 3. Train-Test Split (Stratified on loan_status)
    X = full_df.drop(columns=["loan_status"])
    y = full_df["loan_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    # 4. Fit Preprocessor ONLY on training set
    preprocessor = build_preprocessor_pipeline()
    numeric_cols, categorical_cols = get_feature_columns()
    
    preprocessor.fit(X_train)
    
    # 5. Save Preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    
    feature_names = get_feature_names_out(preprocessor, numeric_cols, categorical_cols)
    
    print(f"Preprocessing completed successfully:")
    print(f"- Raw rows: {len(raw_df)}")
    print(f"- Cleaned rows: {len(clean_df)}")
    print(f"- Train rows: {len(train_df)} (Defaults: {y_train.sum()}, Non-defaults: {len(y_train) - y_train.sum()})")
    print(f"- Test rows: {len(test_df)} (Defaults: {y_test.sum()}, Non-defaults: {len(y_test) - y_test.sum()})")
    print(f"- Total processed features: {len(feature_names)}")
    print(f"- Saved preprocessor to: {preprocessor_path}")
    
    return train_df, test_df, preprocessor, feature_names


if __name__ == "__main__":
    prepare_and_save_data()
