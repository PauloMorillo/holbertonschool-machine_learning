#!/usr/bin/env python3
"""
This module has the from_numpy(array): method
"""

import pandas as pd


def from_numpy(array):
    """
    This method creates a pd.DataFrame from a np.ndarray
    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(array)
    df_l = len(df.columns)
    cap_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    if df_l > 26:
        df = df.iloc[:, 0:26]
        cap_letters = cap_letters[0:26]
    else:
        cap_letters = cap_letters[0:df_l]

    df.columns = cap_letters
    return df
