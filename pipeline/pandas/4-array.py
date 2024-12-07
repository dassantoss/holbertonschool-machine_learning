#!/usr/bin/env python3
"""Module to convert DataFrame columns to a numpy array."""


def array(df):
    """
    Selects the last 10 rows of 'High' and 'Close' columns and converts
    them to a numpy.ndarray.

    Parameters:
    - df: pd.DataFrame containing columns named 'High' and 'Close'

    Returns:
    - numpy.ndarray of the selected values
    """
    # Select the last 10 rows of 'High' and 'Close' columns
    selected_data = df[['High', 'Close']].tail(10)

    # Convert the selected data to a numpy array
    return selected_data.to_numpy()
