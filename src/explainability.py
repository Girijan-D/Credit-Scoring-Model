"""
Model Explainability & Interpretability Module
Extracts global feature importances, SHAP values, and local individual applicant risk drivers.
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap

from src.preprocessing import get_feature_columns, get_feature_names_out
from src.feature_engineering import add_engineered_features


def explain_model(
    test_path: str = "data/processed/test.csv",
    models_dir: str = "models",
    output_dir: str = "outputs",
    charts_dir: str = "outputs/evaluation_charts",
    powerbi_dir: str = "powerbi"
):
    """
    Computes global feature importance and SHAP analysis on the champion model.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(powerbi_dir, exist_ok=True)
    
    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    best_model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    
    numeric_cols, categorical_cols = get_feature_columns()
    feature_names = get_feature_names_out(preprocessor, numeric_cols, categorical_cols)
    
    X_test_raw = test_df.drop(columns=["loan_status"])
    X_test_trans = preprocessor.transform(X_test_raw)
    
    # 1. Extract Model Feature Importances
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        raw_importances = np.abs(best_model.coef_[0])
    else:
        raw_importances = np.ones(len(feature_names)) / len(feature_names)
        
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": raw_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    # Add normalized relative percentage
    feat_imp_df["Relative_Importance_Pct"] = (feat_imp_df["Importance"] / feat_imp_df["Importance"].sum()) * 100
    
    feat_imp_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    feat_imp_df.to_csv(os.path.join(powerbi_dir, "powerbi_feature_importance.csv"), index=False)
    
    # -------------------------------------------------------------
    # 2. Top 15 Feature Importance Bar Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    top_feats = feat_imp_df.head(15).iloc[::-1]  # Reverse for top-down horizontal bar
    ax.barh(top_feats["Feature"], top_feats["Relative_Importance_Pct"], color="#2980b9", edgecolor="black")
    for i, (val, name) in enumerate(zip(top_feats["Relative_Importance_Pct"], top_feats["Feature"])):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Top 15 Most Influential Features in Default Prediction", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance Weight (%)", fontsize=11)
    ax.set_xlim(0, max(top_feats["Relative_Importance_Pct"]) * 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "feature_importance_bar.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 3. SHAP Summary Plot
    # -------------------------------------------------------------
    print("Computing SHAP values for global interpretability...")
    try:
        # Sample for fast, accurate SHAP computation
        sample_size = min(1000, len(X_test_trans))
        X_sample = X_test_trans[:sample_size]
        
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        
        # If binary classification returns list [shap_class_0, shap_class_1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_to_plot = shap_values[1]
        else:
            shap_vals_to_plot = shap_values
            
        fig = plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_vals_to_plot,
            X_sample,
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        plt.title("SHAP Feature Attribution Summary Plot (Impact on Default Risk)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(charts_dir, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot successfully saved to {charts_dir}/shap_summary_plot.png")
    except Exception as e:
        print(f"SHAP calculation note: {e}")
        
    print(f"Explainability analysis complete. Top 5 drivers:\n{feat_imp_df.head(5).to_string(index=False)}")
    return feat_imp_df


def explain_single_applicant(
    applicant_data: dict,
    model_path: str = "models/best_model.joblib",
    preprocessor_path: str = "models/preprocessor.joblib"
):
    """
    Computes local feature impact drivers for an individual loan applicant.
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    numeric_cols, categorical_cols = get_feature_columns()
    feature_names = get_feature_names_out(preprocessor, numeric_cols, categorical_cols)
    
    df = pd.DataFrame([applicant_data])
    if "loan_percent_income" not in df.columns or pd.isna(df["loan_percent_income"].iloc[0]):
        df["loan_percent_income"] = df["loan_amnt"] / np.maximum(df["person_income"], 1.0)
    df_feat = add_engineered_features(df)
    X_trans = preprocessor.transform(df_feat)
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            vals = shap_vals[1][0]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals
            
        driver_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact_Score": vals,
            "Direction": ["Increases Risk" if v > 0 else "Decreases Risk" for v in vals]
        }).sort_values(by="Impact_Score", key=abs, ascending=False).head(8)
        
        return driver_df.to_dict(orient="records")
    except Exception as e:
        return [{"Feature": "Model Attribution", "Impact_Score": 0.0, "Direction": str(e)}]


if __name__ == "__main__":
    explain_model()
