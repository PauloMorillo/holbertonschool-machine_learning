#!/usr/bin/env python3
"""
This module has the method optimum_k(X, kmin=1, kmax=None, iterations=1000):
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters
    to check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters
    to check for (inclusive)
    iterations is a positive integer containing the maximum number of
    iterations for K-means
    Returns: results, d_vars, or None, None on failure
    results is a list containing the outputs of K-means for each cluster size
    d_vars is a list containing the difference in variance from the smallest
    cluster size for each cluster size
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) != int or kmin <= 0 or X.shape[0] <= kmin:
        return None, None
    if type(kmax) != int or kmax <= 0 or X.shape[0] < kmax:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if kmin is k:
            smallest_var = variance(X, C)
        var = variance(X, C)
        results = results + [(C, clss)]
        d_vars = d_vars + [smallest_var - var]
    return results, d_vars
