#!/usr/bin/env python3
"""
This module indexes the pd.DataFrame on the Timestamp column
"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.set_index("Timestamp")  # YOUR CODE HERE

print(df.tail())  # print(df.columns)
