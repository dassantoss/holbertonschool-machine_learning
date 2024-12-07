#!/usr/bin/env python3
"""Module to create a DataFrame from a dictionary."""
import pandas as pd


# Define the dictionary with the specified data
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create the DataFrame with the specified row labels
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
