#!/usr/bin/env python3
"""
This module has the expectation(X, pi, m, S)
"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for each cluster
    Returns: g, l, or None, None on failure
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    l is the total log likelihood
    """
    try:
        K = m.shape[0]
        n = X.shape[0]
        d = X.shape[1]
        if m.shape[1] != d or S.shape[1] != d or S.shape[2] != d:
            return None, None
        if S.shape[0] != K:
            return None, None

            # sum of pi equal to 1
        if not np.isclose([np.sum(pi)], [1])[0]:
            return None, None
        numerator = np.zeros((K, n))
        for k in range(K):
            numerator[k] = pi[k] * pdf(X, m[k], S[k])
        denominator = np.sum(numerator, axis=0)
        g = numerator / denominator
        likelihood = np.sum(np.log(np.sum(numerator, axis=0)))
        return g, likelihood
    except Exception as e:
        return None, None
