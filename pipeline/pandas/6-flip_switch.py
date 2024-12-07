#!/usr/bin/env python3
"""Module to sort and transpose a DataFrame."""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order and transposes it.

    Parameters:
    - df: pd.DataFrame to transform

    Returns:
    - pd.DataFrame sorted and transposed
    """
    # Sort in reverse chronological order (descending)
    df_sorted = df.sort_index(ascending=False)

    # Transpose the sorted DataFrame
    return df_sorted.T
