#!/usr/bin/env python3
"""
Module that contains function to remove NaN values from Close column
"""


def prune(df):
    """
    Removes entries where Close has NaN values
    Args:
        df: pandas DataFrame
    Returns:
        Modified pandas DataFrame with NaN values removed from Close column
    """
    return df.dropna(subset=['Close'])
