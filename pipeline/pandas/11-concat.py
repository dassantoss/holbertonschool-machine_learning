#!/usr/bin/env python3
"""
Module that contains function to concatenate two DataFrames
"""
import pandas as pd


def concat(df1, df2):
    """
    Concatenates two DataFrames with specific conditions
    Args:
        df1: First DataFrame (coinbase)
        df2: Second DataFrame (bitstamp)
    Returns:
        Concatenated DataFrame with keys
    """
    # Import index function
    index = __import__('10-index').index

    # Index both dataframes on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select rows from df2 up to and including timestamp 1417411920
    df2_filtered = df2[df2.index <= 1417411920]

    # Concatenate DataFrames with keys
    df_concat = pd.concat([df2_filtered, df1],
                          keys=['bitstamp', 'coinbase'],
                          axis=0)

    return df_concat
