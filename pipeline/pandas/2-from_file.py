#!/usr/bin/env python3
"""Module to load data from a file into a DataFrame."""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.

    Parameters:
    - filename: the file to load from
    - delimiter: the column separator

    Returns:
    - pd.DataFrame containing the loaded data
    """
    # Use pandas to read the file with the specified delimiter
    df = pd.read_csv(filename, delimiter=delimiter)

    return df
