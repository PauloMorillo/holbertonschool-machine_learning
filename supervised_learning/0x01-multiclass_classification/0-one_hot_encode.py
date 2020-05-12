#!/usr/bin/env python3
""" This module has the method one_hot_encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """This method encode a numerical label vector into
    a one-hot matrix
    """
    if Y is None:
        return None
    if type(classes) is not int or classes <= np.amax(Y):
        return None
    encoding_y = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        encoding_y[Y[i]][i] = 1
    return encoding_y
