#!/usr/bin/env python3
"""Module to rename a DataFrame column and convert its values."""
import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime' and converts its values to
    datetime.

    Parameters:
    - df: pd.DataFrame containing a column named 'Timestamp'

    Returns:
    - pd.DataFrame with 'Datetime' and 'Close' columns
    """
    # Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the 'Datetime' column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Return only the 'Datetime' and 'Close' columns
    return df[['Datetime', 'Close']]
