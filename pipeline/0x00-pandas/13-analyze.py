#!/usr/bin/env python3
"""
This module calculate descriptive statistics for all columns
in pd.DataFrame except Timestamp
"""

import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

stats = df.loc[:, df.columns != "Timestamp"].describe()  # YOUR CODE HERE

print(stats)
