"""
Exploratory Data Analysis (EDA) Module
Generates production-grade analytical charts and statistical breakdowns from the real credit dataset.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def generate_eda_visualizations(
    data_path: str = "data/processed/cleaned_dataset.csv",
    output_dir: str = "outputs/eda_charts"
):
    """
    Generates and saves all EDA visualizations using actual dataset statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cleaned data
    if not os.path.exists(data_path):
        from src.preprocessing import prepare_and_save_data
        prepare_and_save_data()
        
    df = pd.read_csv(data_path)
    
    # Styling configurations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    palette = ["#2ecc71", "#e74c3c"]  # Green for Non-Default, Red for Default
    
    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    target_counts = df["loan_status"].value_counts()
    target_pct = df["loan_status"].value_counts(normalize=True) * 100
    bars = ax.bar(["Non-Default (0)", "Default (1)"], target_counts.values, color=palette, width=0.55, edgecolor="black", linewidth=1.2)
    for bar, pct in zip(bars, target_pct.values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 400, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Loan Status Class Distribution (Target Variable)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Applicants", fontsize=11)
    ax.set_ylim(0, max(target_counts) * 1.18)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Correlation Matrix Heatmap
    fig, ax = plt.subplots(figsize=(11, 8))
    numeric_cols = [
        "person_age", "person_income", "person_emp_length", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "loan_to_income_ratio", "income_to_loan_ratio", "annual_interest_burden",
        "loan_status"
    ]
    corr_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
        linewidths=0.7, ax=ax, annot_kws={"size": 9}
    )
    ax.set_title("Correlation Heatmap of Financial & Demographic Features", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    # 3. Default Rate by Loan Grade
    fig, ax = plt.subplots(figsize=(8, 5))
    grade_order = sorted(df["loan_grade"].dropna().unique())
    grade_stat = df.groupby("loan_grade")["loan_status"].agg(Total="count", Defaults="sum", Default_Rate="mean").loc[grade_order]
    
    bars = ax.bar(grade_stat.index, grade_stat["Default_Rate"] * 100, color="#3498db", edgecolor="black", width=0.6)
    for bar, (_, row) in zip(bars, grade_stat.iterrows()):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%\n(n={int(row['Total']):,})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Default Rate by Credit Loan Grade (A to G)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Loan Grade (Assigned Credit Tier)", fontsize=11)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.set_ylim(0, max(grade_stat["Default_Rate"] * 100) * 1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "default_by_loan_grade.png"), dpi=300)
    plt.close()
    
    # 4. Default Rate by Home Ownership
    fig, ax = plt.subplots(figsize=(8, 5))
    home_stat = df.groupby("person_home_ownership")["loan_status"].agg(Total="count", Defaults="sum", Default_Rate="mean").sort_values("Default_Rate", ascending=False)
    
    bars = ax.bar(home_stat.index, home_stat["Default_Rate"] * 100, color="#9b59b6", edgecolor="black", width=0.55)
    for bar, (_, row) in zip(bars, home_stat.iterrows()):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%\n(n={int(row['Total']):,})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Default Rate by Home Ownership Status", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Home Ownership", fontsize=11)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.set_ylim(0, max(home_stat["Default_Rate"] * 100) * 1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "default_by_home_ownership.png"), dpi=300)
    plt.close()
    
    # 5. Default Rate by Loan Intent / Purpose
    fig, ax = plt.subplots(figsize=(10, 5))
    intent_stat = df.groupby("loan_intent")["loan_status"].agg(Total="count", Defaults="sum", Default_Rate="mean").sort_values("Default_Rate", ascending=False)
    
    bars = ax.bar(intent_stat.index, intent_stat["Default_Rate"] * 100, color="#e67e22", edgecolor="black", width=0.6)
    for bar, (_, row) in zip(bars, intent_stat.iterrows()):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%\n(n={int(row['Total']):,})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Default Rate by Loan Intent / Purpose", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Loan Purpose", fontsize=11)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.set_ylim(0, max(intent_stat["Default_Rate"] * 100) * 1.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "default_by_loan_intent.png"), dpi=300)
    plt.close()
    
    # 6. Income & Loan Amount Distributions by Default Status
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Boxplot of Loan Percent Income
    sns.boxplot(x="loan_status", y="loan_percent_income", data=df, palette=palette, ax=axes[0], width=0.4)
    axes[0].set_xticklabels(["Non-Default (0)", "Default (1)"])
    axes[0].set_title("Loan Percent of Income by Default Status", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Loan as % of Income", fontsize=10)
    
    # Boxplot of Loan Interest Rate
    sns.boxplot(x="loan_status", y="loan_int_rate", data=df, palette=palette, ax=axes[1], width=0.4)
    axes[1].set_xticklabels(["Non-Default (0)", "Default (1)"])
    axes[1].set_title("Interest Rate by Default Status", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Interest Rate (%)", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "income_vs_loan_dist.png"), dpi=300)
    plt.close()
    
    # 7. Historical Default on File vs Current Default
    fig, ax = plt.subplots(figsize=(7, 5))
    cb_stat = df.groupby("cb_person_default_on_file")["loan_status"].agg(Total="count", Defaults="sum", Default_Rate="mean")
    bars = ax.bar(["No Prior Default (N)", "Prior Default on File (Y)"], cb_stat["Default_Rate"] * 100, color=["#27ae60", "#c0392b"], edgecolor="black", width=0.5)
    for bar, (_, row) in zip(bars, cb_stat.iterrows()):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%\n(n={int(row['Total']):,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Default Rate by Historical Credit Bureau Default Record", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Current Default Rate (%)", fontsize=11)
    ax.set_ylim(0, max(cb_stat["Default_Rate"] * 100) * 1.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "credit_history_analysis.png"), dpi=300)
    plt.close()
    
    print(f"EDA charts successfully generated and saved to {output_dir}")


if __name__ == "__main__":
    generate_eda_visualizations()
