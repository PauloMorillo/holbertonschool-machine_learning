#!/usr/bin/env python3
""" This module has the shuffle_data method"""
import numpy as np


def shuffle_data(X, Y):
    """
    This method shuffle_data
    """
    vec = np.arange(X.shape[0])
    i = np.random.permutation(vec)
    return X[i], Y[i]
