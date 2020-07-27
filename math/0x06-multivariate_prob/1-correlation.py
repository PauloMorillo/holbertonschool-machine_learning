#!/usr/bin/env python3
"""This module has the method correlation(C)"""

import numpy as np


def correlation(C):
    """This function calculates correlation"""

    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) < 2 or C.shape[0] is not C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    return C / np.sqrt(np.outer(np.diagonal(C), np.diagonal(C)))
