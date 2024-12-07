#!/usr/bin/env python3
"""Module to slice specific columns from a DataFrame."""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.

    Parameters:
    - df: pd.DataFrame to slice

    Returns:
    - pd.DataFrame containing selected columns and rows
    """
    # Select the specified columns
    columns_to_extract = ['High', 'Low', 'Close', 'Volume_(BTC)']

    # Extract columns and select every 60th row
    return df[columns_to_extract][::60]
