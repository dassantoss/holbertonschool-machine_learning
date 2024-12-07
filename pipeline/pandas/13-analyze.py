#!/usr/bin/env python3
"""
Module that contains function to compute descriptive statistics
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp
    Args:
        df: pandas DataFrame
    Returns:
        DataFrame containing descriptive statistics
    """
    # Drop Timestamp column if it exists and compute statistics
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp', axis=1)

    return df.describe()
