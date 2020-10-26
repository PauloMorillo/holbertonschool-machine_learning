#!/usr/bin/env python3
"""
This module fills in the missing data points in the pd.DataFrame
"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# print(df.columns)
df = df.drop(columns=['Weighted_Price'])
df["Close"].fillna(method='ffill', inplace=True)
df["High"].fillna(df["Close"].shift(1, fill_value=0), inplace=True)
df["Low"].fillna(df["Close"].shift(1, fill_value=0), inplace=True)
df["Open"].fillna(df["Close"].shift(1, fill_value=0), inplace=True)
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)
# YOUR CODE HERE

print(df.head())
print(df.tail())
