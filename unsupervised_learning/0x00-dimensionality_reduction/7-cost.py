#!/usr/bin/env python3
""" This module has the method Q_affinities(Y)"""

import numpy as np


def Q_affinities(Y):
    """
    Y is a numpy.ndarray of shape (n, ndim) containing the low
    dimensional transformation of X n is the number of points
    ndim is the new dimensional representation of X
    Returns: Q, num Q is a numpy.ndarray of shape (n, n)
    containing the Q affinities num is a numpy.ndarray of
    shape (n, n) containing the numerator of the Q affinities
    """
    sum_Y = np.sum(np.square(Y), 1)
    Di = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    np.fill_diagonal(Di, 0)
    num = 1 / (1 + Di)
    np.fill_diagonal(num, 0)
    Qi = num / np.sum(num)
    return Qi, num
