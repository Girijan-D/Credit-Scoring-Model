"""
Model Evaluation Module for Credit Scoring System
Computes comprehensive classification metrics, confusion matrices, ROC/PR curves,
and exports Power BI ready data files.
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


def evaluate_models(
    test_path: str = "data/processed/test.csv",
    models_dir: str = "models",
    output_dir: str = "outputs",
    charts_dir: str = "outputs/evaluation_charts",
    powerbi_dir: str = "powerbi"
):
    """
    Evaluates all trained models on the held-out test set,
    generates publication-ready diagnostic plots, and exports CSV metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(powerbi_dir, exist_ok=True)
    
    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    
    X_test_raw = test_df.drop(columns=["loan_status"])
    y_test = test_df["loan_status"].values
    
    X_test = preprocessor.transform(X_test_raw)
    
    model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    models = {}
    for name in model_names:
        filename = name.lower().replace(" ", "_") + ".joblib"
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            
    metrics_list = []
    roc_data = {}
    pr_data = {}
    cm_dict = {}
    test_predictions_dict = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        cm = confusion_matrix(y_test, y_pred)
        cm_dict[name] = cm
        
        # Calculate FPR, TPR for ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        
        # Calculate Precision, Recall for PR curve
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (p_curve, r_curve, pr_auc)
        
        test_predictions_dict[f"{name}_pred"] = y_pred
        test_predictions_dict[f"{name}_prob"] = y_prob
        
        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Default)": prec,
            "Recall (Default)": rec,
            "F1-Score (Default)": f1,
            "ROC-AUC": auc,
            "PR-AUC (Avg Precision)": pr_auc,
            "True Negatives": cm[0, 0],
            "False Positives": cm[0, 1],
            "False Negatives": cm[1, 0],
            "True Positives": cm[1, 1]
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
    metrics_df.to_csv(os.path.join(powerbi_dir, "powerbi_model_metrics.csv"), index=False)
    
    print("=" * 70)
    print("TEST SET EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(metrics_df[["Model", "Accuracy", "Precision (Default)", "Recall (Default)", "F1-Score (Default)", "ROC-AUC"]].to_string(index=False))
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. ROC Curves Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#9b59b6", "#2ecc71", "#e67e22"]
    for (name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", linewidth=2.2, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "roc_curves.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Precision-Recall Curves Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, (p_curve, r_curve, pr_auc_score)), color in zip(pr_data.items(), colors):
        ax.plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_auc_score:.4f})", linewidth=2.2, color=color)
    baseline_pr = y_test.sum() / len(y_test)
    ax.axhline(baseline_pr, color="k", linestyle="--", alpha=0.6, label=f"Baseline Default Rate ({baseline_pr:.2%})")
    ax.set_title("Precision-Recall Curves (Default Class Detection)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Defaults Captured)", fontsize=11)
    ax.set_ylabel("Precision (Defaults Correctness)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 3. Side-by-Side Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Non-Default (0)", "Default (1)"],
                    yticklabels=["Non-Default (0)", "Default (1)"],
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"{name}\nAcc: {metrics_df.loc[metrics_df['Model']==name, 'Accuracy'].values[0]:.2%} | Recall: {metrics_df.loc[metrics_df['Model']==name, 'Recall (Default)'].values[0]:.2%}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("Actual Class", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Model Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    comp_df = metrics_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Recall (Default)", "Precision (Default)", "F1-Score (Default)", "ROC-AUC"],
                              var_name="Metric", value_name="Score")
    sns.barplot(data=comp_df, x="Metric", y="Score", hue="Model", palette="tab10", ax=ax, edgecolor="black")
    ax.set_title("Multi-Model Comparative Performance Benchmark", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.legend(title="Model Family", loc="lower right", fontsize=9)
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=7.5, rotation=0, xytext=(0, 2),
                        textcoords="offset points")
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "model_comparison_bar.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 5. Generate Test Applicant Predictions & Power BI Exports
    # -------------------------------------------------------------
    # Using Champion model (e.g., XGBoost / best model)
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    best_model = joblib.load(best_model_path)
    
    best_probs = best_model.predict_proba(X_test)[:, 1]
    best_preds = best_model.predict(X_test)
    
    pred_export = test_df.copy()
    pred_export["actual_default"] = y_test
    pred_export["predicted_default"] = best_preds
    pred_export["default_probability"] = np.round(best_probs, 4)
    
    # Assign Configurable Risk Categories:
    # Low Risk: < 0.20 | Medium Risk: 0.20 - 0.50 | High Risk: >= 0.50
    def assign_risk_category(prob):
        if prob < 0.20:
            return "Low Risk"
        elif prob < 0.50:
            return "Medium Risk"
        else:
            return "High Risk"
            
    pred_export["risk_category"] = pred_export["default_probability"].apply(assign_risk_category)
    
    # Save test set predictions
    pred_export.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    pred_export.to_csv(os.path.join(powerbi_dir, "powerbi_applicant_predictions.csv"), index=False)
    
    # Generate Risk Category Summary Table
    risk_summary = pred_export.groupby("risk_category").agg(
        Applicant_Count=("actual_default", "count"),
        Actual_Defaults=("actual_default", "sum"),
        Actual_Default_Rate=("actual_default", "mean"),
        Average_Default_Probability=("default_probability", "mean"),
        Avg_Loan_Amount=("loan_amnt", "mean"),
        Avg_Income=("person_income", "mean"),
        Avg_Interest_Rate=("loan_int_rate", "mean")
    ).reset_index()
    
    risk_summary.to_csv(os.path.join(output_dir, "risk_category_summary.csv"), index=False)
    risk_summary.to_csv(os.path.join(powerbi_dir, "powerbi_risk_segment_summary.csv"), index=False)
    
    print(f"Evaluation complete. Reports and charts saved to {output_dir} and {powerbi_dir}")
    return metrics_df, pred_export, risk_summary


if __name__ == "__main__":
    evaluate_models()
