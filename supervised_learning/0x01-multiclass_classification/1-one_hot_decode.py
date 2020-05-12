#!/usr/bin/env python3
""" This module has the method one_hot_decode"""

import numpy as np


def one_hot_decode(one_hot):
    """
    This method decode a one-hot matrix into a numerical label
    """
    try:
        decoding = np.zeros(one_hot.shape[0])
        rowspos = 0
        for row in one_hot:
            colspos = 0
            for col in row:
                if col == 1:
                    decoding[colspos] = rowspos
                colspos = colspos + 1
            rowspos = rowspos + 1
    except E:
        return None
    return decoding.astype(int)
