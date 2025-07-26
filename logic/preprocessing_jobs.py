# logic/preprocessing_jobs.py

import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def run_preprocessing_job(df, missing_value_strategy, scaling_strategy):
    """
    Applies selected preprocessing steps to a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        missing_value_strategy (str): 'mean', 'median', or 'drop'.
        scaling_strategy (str): 'standard' or 'none'.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Make a copy to avoid modifying the original DataFrame in memory
    processed_df = df.copy()

    # 1. Handle Missing Values
    if missing_value_strategy == 'drop':
        processed_df.dropna(inplace=True)
    elif missing_value_strategy in ['mean', 'median']:
        # Impute only numeric columns
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        imputer = SimpleImputer(strategy=missing_value_strategy)
        processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])

    # 2. Scale Numerical Features
    if scaling_strategy == 'standard':
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])

    return processed_df