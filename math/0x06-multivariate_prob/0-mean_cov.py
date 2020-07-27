#!/usr/bin/env python3
"""This module has the method mean_cov(X)"""

import numpy as np


def mean_cov(X):
    """This function calculates mean and covariance"""

    if type(X) is not np.ndarray or len(X.shape) is not 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    mean_m = np.mean(X.T, axis=1)
    resta = X - mean_m
    covar = np.dot(resta.T, resta) / (X.shape[0] - 1)
    return mean_m, covar
