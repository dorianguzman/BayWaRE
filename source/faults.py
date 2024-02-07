# -*- coding: utf-8 -*-
"""
Created on Feb 07

@author: DorianGuzman
"""
import pandas as pd
import numpy as np


def create_faults_report(outliers: pd.DataFrame) -> dict:
    """
    Create a report of faulty days by String based on the provided DataFrame of outliers.

    Parameters
    ----------
    outliers : pd.DataFrame
        DataFrame containing outliers detected based on the confidence interval.

    Returns
    -------
    dict
        Dictionary where keys are Strings and values are lists of faulty days for each String.
    """

    # Initialize an empty dictionary to store faults by string
    faults_report = {}

    # Iterate over each column (String) in outliers
    for col in outliers.columns:
        # Filter rows where outliers are present in the column
        faulty_days = outliers[col].dropna().index.date
        faulty_days = pd.Index(faulty_days).unique()

        # Only add to the report if there are faulty days for the current column
        if len(faulty_days) > 0:
            faults_report[col] = faulty_days

    return faults_report


def replace_outliers_by_threshold(outliers: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    """
    Replace outliers in the input DataFrame with NaN based on the specified threshold.

    Parameters
    ----------
    outliers : pd.DataFrame
        DataFrame containing outliers detected based on the confidence interval.
    threshold : int, optional
        Threshold for the number of outliers to consider a day faulty, by default 20.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers replaced by NaN for days exceeding the threshold.
    """

    # Count outliers for each day and column
    daily_outlier_count = outliers.apply(lambda col: col.groupby(outliers.index.date).transform('count'))
    
    # Identify days and columns where the outlier count exceeds the threshold
    days_to_replace = daily_outlier_count[daily_outlier_count < threshold]
    
    # Create a mask for outliers to be replaced
    mask = (days_to_replace <= threshold) & (days_to_replace.notna())
    
    # Replace outliers with NaN using the mask
    outliers[mask] = np.nan
    
    return outliers


def detect_outliers(norm_df: pd.DataFrame, n_std: int = 2) -> pd.DataFrame:
    """
    Detect outliers in the input DataFrame based on mean and standard deviation.

    Parameters
    ----------
    norm_df : pd.DataFrame
        The input DataFrame with time series data.
    n_std : int, optional
        Number of standard deviations for the confidence interval, by default 2.

    Returns
    -------
    pd.DataFrame
        DataFrame containing outliers detected based on the confidence interval.
    """

    # Calculate mean and standard deviation along each row
    mean_df = norm_df.mean(axis=1)
    std_df = norm_df.std(axis=1)

    # Calculate lower and upper bounds
    lower_df = mean_df - (n_std * std_df)
    lower_df = lower_df.apply(lambda x: max(0, x))  # Replace negative values with 0
    upper_df = mean_df + (n_std * std_df)

    # Detect values outside the confidence interval
    outliers_df = norm_df.apply(lambda col: col[col < lower_df])

    return outliers_df

