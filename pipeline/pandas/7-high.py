#!/usr/bin/env python3
"""
Module that contains function to sort DataFrame by High price
"""


def high(df):
    """
    Sorts DataFrame by High price in descending order
    Args:
        df: pandas DataFrame
    Returns:
        Sorted pandas DataFrame
    """
    return df.sort_values(by='High', ascending=False)
