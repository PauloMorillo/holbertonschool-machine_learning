#!/usr/bin/env python3
"""
This module Plot the data from 2017 and beyond at daily intervals
The column Weighted_Price should be removed
Rename the column Timestamp to Date
Convert the timestamp values to date values
Index the data frame on Date
Missing values in High, Low, Open, and Close should
be set to the previous row’s Close value
Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df["Close"].fillna(method='bfill', inplace=True)
df["High"].fillna(df["Close"].shift(1, fill_value=0), inplace=True)
df["Low"].fillna(df["Close"].shift(1, fill_value=0), inplace=True)
df["Open"].fillna(df["Close"].shift(1, fill_value=0), inplace=True)
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)
df = df[df["Date"] >= "2017-01-01"]
df = df.groupby(df["Date"].dt.date).sum()
df = df.loc[:, df.columns != "Date"]
df.plot()
plt.show()
