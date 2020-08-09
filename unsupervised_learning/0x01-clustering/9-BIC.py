#!/usr/bin/env python3
"""
This module has the BIC(X, kmin=1, kmax=None,
iterations=1000, tol=1e-5, verbose=False):
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters
    to check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters
    to check for (inclusive)
    iterations is a positive integer containing the maximum number
    of iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the EM algorithm
    verbose is a boolean that determines if the EM algorithm should print
    information to the standard output
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
    best_k is the best value for k based on its BIC
    best_result is tuple containing pi, m, S
    pi is a numpy.ndarray of shape (k,) containing the cluster priors for
    the best number of clusters
    m is a numpy.ndarray of shape (k, d) containing the centroid means
    for the best number of clusters
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for the best number of clusters
    l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
    likelihood for each cluster size tested
    b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
    for each cluster size tested
    """
    # kmin, kmax, iterations, tol, verbose, X
    # expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    try:
        if kmin >= kmax:
            return None, None, None, None
        b = []
        d = X.shape[1]
        all_li = []
        for k in range(kmin, kmax + 1):
            pi, m, S, g, li = expectation_maximization(X,
                                                       k,
                                                       iterations,
                                                       tol, verbose)
            p = k - 1 + (k * d) + ((k * d) * (d + 1)) / 2
            BIC = p * np.log(X.shape[0]) - 2 * li
            if kmin is k:
                BIC_prev = BIC
            if BIC_prev >= BIC:
                best_k = k
                best_result = (pi, m, S)
                BIC_prev = BIC
            b.append(BIC)
            all_li.append(li)
            # print(BIC)

        return best_k, best_result, np.array(all_li), np.array(b)
    except Exception as e:
        return None, None, None, None
