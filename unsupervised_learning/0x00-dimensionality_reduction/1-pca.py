#!/usr/bin/env python3
""" This module has the method pca(X, ndim)"""

import numpy as np


def pca(X, ndim):
    """
    This method perfoms PCA on a dataset
    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
    containing the transformed version of X
    """
    X = X - np.mean(X, axis=0)
    u, s, v = np.linalg.svd(X)
    return np.matmul(X, v[:ndim].T)
