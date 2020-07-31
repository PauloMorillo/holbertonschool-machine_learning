#!/usr/bin/env python3
"""
This module has the method likelihood(x, n, P)
"""

import numpy as np


def marginal(x, n, P, Pr):
    """
    This calculates the marginal probability of obtaining the data
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) is not 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factor2 = (P ** x) * (1 - P) ** (n - x)
    factor1 = np.math.factorial(n) / (np.math.factorial(x) *
                                      np.math.factorial(n - x))
    prob = factor1 * factor2 * Pr

    return np.sum(prob)
