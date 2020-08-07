#!/usr/bin/env python3
"""
This module has the method variance(X, C):
"""
import numpy as np


def variance(X, C):
    """
    This method calculates the total intra-cluster
    variance for a data set
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    Returns: var, or None on failure
    var is the total variance
    """
    D = np.min(np.linalg.norm(C - X[:, None], axis=-1), axis=1)
    V = np.sum(D ** 2)
    return V
