#!/usr/bin/env python3
"""Script to visualize a transformed pd.DataFrame."""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data
df = from_file('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the 'Weighted_Price' column
df = df.drop(columns=['Weighted_Price'])

# Rename 'Timestamp' to 'Date'
df = df.rename(columns={'Timestamp': 'Date'})

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the DataFrame on 'Date'
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data for 2017 and beyond
df = df[df.index.year >= 2017]

# Resample data to daily intervals and aggregate
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
df_daily[['High', 'Low', 'Close']].plot(figsize=(12, 6),
                                        title='Bitcoin Prices (Daily)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# Print the transformed DataFrame
print(df_daily)
