#!/usr/bin/env python3
""" This module has the method P_affinities(X, tol=1e-5, perplexity=30.0)"""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset to
    be transformed by t-SNE
    n is the number of data points
    d is the number of dimensions in each point
    perplexity is the perplexity that all Gaussian distributions
    should have
    tol is the maximum tolerance allowed (inclusive) for the difference
    in Shannon entropy from perplexity for all Gaussian distributions
    You should use P_init = __import__('2-P_init').P_init and
    HP = __import__('3-entropy').HP
    Returns: P, a numpy.ndarray of shape (n, n) containing the symmetric
    P affinities
    """
    sum_Y = np.sum(np.square(Y), 1)
    Di = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    np.fill_diagonal(Di, 0)
    num = 1 / (1 + Di)
    np.fill_diagonal(num, 0)
    Qi = num / np.sum(num)
    return Qi
