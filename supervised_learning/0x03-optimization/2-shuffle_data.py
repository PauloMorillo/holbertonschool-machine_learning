#!/usr/bin/env python3
""" This module has the shuffle_data method"""
import numpy as np


def shuffle_data(X, Y):
    """
    This method shuffle_data
    """
    X = np.random.permutation(X)
    np.random.seed(0)
    Y = np.random.permutation(Y)
    return X, Y
