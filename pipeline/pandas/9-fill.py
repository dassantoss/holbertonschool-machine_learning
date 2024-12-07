#!/usr/bin/env python3
"""
Module that contains function to clean and fill missing values in DataFrame
"""


def fill(df):
    """
    Cleans DataFrame by removing and filling missing values
    Args:
        df: pandas DataFrame
    Returns:
        Modified pandas DataFrame with filled values and removed column
    """
    # Remove Weighted_Price column
    df = df.drop('Weighted_Price', axis=1)

    # Fill missing Close values with previous row's value
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill missing High, Low, Open with corresponding Close value
    for column in ['High', 'Low', 'Open']:
        df[column] = df[column].fillna(df['Close'])

    # Fill missing volume values with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
