#!/usr/bin/env python3
"""
This module has the from_file(filename, delimiter): method
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    This method loads data from a file as a pd.DataFrame
    filename is the file to load from
    delimiter is the column separator
    Returns: the loaded pd.DataFrame
    """
    pd.set_option("display.max_columns", None)
    return pd.read_csv(filename, delimiter=delimiter)
