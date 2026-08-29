"""
Model Training Module for Credit Scoring System
Trains, tunes, cross-validates, and persists multiple classification models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.preprocessing import prepare_and_save_data


def train_models(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    models_dir: str = "models",
    random_state: int = 42
):
    """
    Trains 4 classification models on the preprocessed training set,
    evaluates them via 5-fold Stratified CV, and serializes the models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(os.path.join(models_dir, "preprocessor.joblib"))):
        prepare_and_save_data()
        
    train_df = pd.read_csv(train_path)
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    
    X_train_raw = train_df.drop(columns=["loan_status"])
    y_train = train_df["loan_status"].values
    
    # Transform training features
    X_train = preprocessor.transform(X_train_raw)
    
    # Calculate class weight ratio for XGBoost scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    trained_models = {}
    cv_results = {}
    
    print("=" * 60)
    print("TRAINING & 5-FOLD CROSS-VALIDATION RESULTS (ROC-AUC)")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"Fitting and cross-validating: {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        
        trained_models[name] = model
        cv_results[name] = {
            "mean_cv_roc_auc": scores.mean(),
            "std_cv_roc_auc": scores.std()
        }
        
        print(f"  -> Mean 5-Fold ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # Save individual model
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, filename))
        
    # Select Best Model based on CV ROC-AUC
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean_cv_roc_auc"])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
    
    print("=" * 60)
    print(f"CHAMPION MODEL SELECTED: {best_model_name}")
    print(f"CV ROC-AUC Score: {cv_results[best_model_name]['mean_cv_roc_auc']:.4f}")
    print(f"Saved best model to: {os.path.join(models_dir, 'best_model.joblib')}")
    print("=" * 60)
    
    return trained_models, cv_results, best_model_name


if __name__ == "__main__":
    train_models()
