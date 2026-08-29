"""
Feature Engineering Module for Credit Scoring Model
Creates domain-specific financial, credit risk, and behavioral features.
"""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and adds engineered credit risk features to the dataframe.
    
    Formulas:
    1. loan_to_income_ratio = loan_amnt / person_income
    2. income_to_loan_ratio = person_income / loan_amnt
    3. annual_interest_burden = loan_amnt * (loan_int_rate / 100)
    4. interest_burden_ratio = annual_interest_burden / person_income
    5. credit_age_ratio = cb_person_cred_hist_length / person_age
    6. emp_to_age_ratio = person_emp_length / person_age
    7. high_risk_flag = 1 if (cb_person_default_on_file == 'Y' and loan_grade in ['D', 'E', 'F', 'G']) else 0
    """
    df = df.copy()
    
    # 1. Loan to Income Ratio
    df['loan_to_income_ratio'] = df['loan_amnt'] / np.maximum(df['person_income'], 1.0)
    
    # 2. Income to Loan Ratio
    df['income_to_loan_ratio'] = df['person_income'] / np.maximum(df['loan_amnt'], 1.0)
    
    # 3. Annual Interest Burden ($)
    int_rate = df['loan_int_rate'].fillna(df['loan_int_rate'].median() if 'loan_int_rate' in df else 11.0)
    df['annual_interest_burden'] = df['loan_amnt'] * (int_rate / 100.0)
    
    # 4. Interest Burden Ratio
    df['interest_burden_ratio'] = df['annual_interest_burden'] / np.maximum(df['person_income'], 1.0)
    
    # 5. Credit History to Age Ratio
    df['credit_age_ratio'] = df['cb_person_cred_hist_length'] / np.maximum(df['person_age'], 1.0)
    
    # 6. Employment Length to Age Ratio
    emp_len = df['person_emp_length'].fillna(df['person_emp_length'].median() if 'person_emp_length' in df else 4.0)
    df['emp_to_age_ratio'] = emp_len / np.maximum(df['person_age'], 1.0)
    
    # 7. High Risk Historical Flag
    is_default_on_file = df['cb_person_default_on_file'].astype(str).str.upper() == 'Y'
    is_low_grade = df['loan_grade'].astype(str).isin(['D', 'E', 'F', 'G'])
    df['high_risk_flag'] = (is_default_on_file & is_low_grade).astype(int)
    
    return df
