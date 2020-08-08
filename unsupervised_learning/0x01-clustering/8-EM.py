#!/usr/bin/env python3
"""
This module has the
expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    This method calculates performs the expectation maximization for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations for the algorithm
    tol is a non-negative float containing tolerance of the log likelihood,
    used to determine early stopping i.e.
    verbose is a boolean that determines if the algo prints information
    """
    try:
        pi, m, S = initialize(X, k)
        l_d = 0
        for i in range(iterations + 1):
            g, li = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)
            # print(li)
            if verbose is True:
                if i % 10 == 0 or i == iterations or i == 0:
                    print("Log Likelihood after {} iterations: {}"
                          .format(i, round(li, 5)))
            if tol >= abs(li - l_d) and i is not 0:
                if verbose:
                    print("Log Likelihood after {} iterations: {}"
                          .format(i, round(li, 5)))
                break
            l_d = li
        return pi, m, S, g, li
    except Exception as e:
        return None, None, None, None, None
