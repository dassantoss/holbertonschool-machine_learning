#!/usr/bin/env python3
"""
Module that contains function to set Timestamp as DataFrame index
"""


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame
    Args:
        df: pandas DataFrame
    Returns:
        Modified pandas DataFrame with Timestamp as index
    """
    return df.set_index('Timestamp')
