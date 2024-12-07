#!/usr/bin/env python3
"""Module to create a DataFrame from a numpy array."""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.

    Parameters:
    - array: np.ndarray from which to create the DataFrame

    Returns:
    - pd.DataFrame with columns labeled A, B, C, ..., Z
    """
    # Generate column labels (A, B, C, etc.) based on array width
    columns = [chr(65 + i) for i in range(array.shape[1])]

    # Create DataFrame with the array and column labels
    df = pd.DataFrame(array, columns=columns)

    return df
