#!/usr/bin/env python3
"""
This module has the maximization(X, g)
"""
import numpy as np


def maximization(X, g):
    """
    This method calculates the maximization step in the EM
    algorithm for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the updated priors
    for each cluster m is a numpy.ndarray of shape (k, d) containing
    the updated centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated
    covariance matrices for each cluster
    """
    try:
        K = g.shape[0]
        if int(np.sum(g)) != g.shape[1]:
            return None, None, None
        d = X.shape[1]
        m = np.zeros((K, d))
        S = np.zeros((K, d, d))
        pi = np.sum(g / len(g[0]), axis=1)
        for k in range(K):
            m[k] = np.matmul(g[k], X) / np.sum(g[k])
            x_m = (X - m[k])
            # print(x_m.shape)
            S[k] = np.matmul(g[k] * (X - m[k]).T, x_m) / np.sum(g[k])
        return pi, m, S
    except Exception as e:
        return None, None, None