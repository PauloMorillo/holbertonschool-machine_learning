#!/usr/bin/env python3
""" This module has the method one_hot_decode"""

import numpy as np


def one_hot_decode(one_hot):
    """
    This method decode a one-hot matrix into a numerical label
    """
    try:
        return np.argmax(one_hot, axis=0)
    except E:
        return None
