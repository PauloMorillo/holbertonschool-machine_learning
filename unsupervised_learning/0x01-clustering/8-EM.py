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
        if type(tol) is not float or tol < 0:
            return None, None, None, None, None
        if type(verbose) is not bool:
            return None, None, None, None, None
        if iterations <= 0:
            return None, None, None, None, None
        pi, m, S = initialize(X, k)
        l_d = 0
        for i in range(1, iterations + 1):
            g, li = expectation(X, pi, m, S)
            # print(li)
            if verbose is True:
                if (i - 1) % 10 == 0 or i - 1 == 0:
                    print("Log Likelihood after {} iterations: {}"
                          .format(i - 1, round(li, 5)))
            if tol >= abs(li - l_d) and i is not 0:
                if verbose:
                    print("Log Likelihood after {} iterations: {}"
                          .format(i - 1, round(li, 5)))
                break
            pi, m, S = maximization(X, g)
            l_d = li
        g, li = expectation(X, pi, m, S)
        if verbose and iterations == i:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, round(li, 5)))
        return pi, m, S, g, li
    except Exception as e:
        return None, None, None, None, None
