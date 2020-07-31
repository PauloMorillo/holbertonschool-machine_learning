# !/usr/bin/env python3
"""
Marginal Probability marginal(x, n, P, Pr)
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    calculates the marginal probability of obtaining the data
    :param x: number of patients that develop severe side effects
    :param n: is the total number of patients observed
    :param P: s a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    :param Pr: is a 1D numpy.ndarray containing the prior beliefs of P
    :return: the marginal probability of obtaining x and n
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

    likhd = (np.math.factorial(n) / (np.math.factorial(x) *
                                     np.math.factorial(n - x))) * (P ** x) * \
            ((1 - P) ** (n - x))
    return np.sum(Pr * likhd)
