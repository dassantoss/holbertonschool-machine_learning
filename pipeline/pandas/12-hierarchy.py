#!/usr/bin/env python3
"""
Module that contains function to create hierarchical index DataFrame
"""
import pandas as pd


def hierarchy(df1, df2):
    """
    Creates hierarchical index DataFrame from two DataFrames
    Args:
        df1: First DataFrame (coinbase)
        df2: Second DataFrame (bitstamp)
    Returns:
        Concatenated DataFrame with hierarchical index
    """
    # Import index function
    index = __import__('10-index').index

    # Index both dataframes on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Filter both DataFrames for the specified time range
    start_time = 1417411980
    end_time = 1417417980

    df1_filtered = df1[(df1.index >= start_time) & (df1.index <= end_time)]
    df2_filtered = df2[(df2.index >= start_time) & (df2.index <= end_time)]

    # Concatenate DataFrames with keys
    df_concat = pd.concat([df2_filtered, df1_filtered],
                          keys=['bitstamp', 'coinbase'])

    # Rearrange MultiIndex to put Timestamp first
    df_concat = df_concat.swaplevel()

    # Sort by timestamp and then by exchange
    df_concat = df_concat.sort_index()

    return df_concat
