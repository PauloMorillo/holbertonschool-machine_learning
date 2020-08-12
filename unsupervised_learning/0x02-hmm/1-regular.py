#!/usr/bin/env python3
"""This module has the method regular(P)"""
import numpy as np


def regular(P):
    """
    this method determines the steady state probabilities of
    a regular markov chain
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the transition matrix
    P[i, j] is the probability of transitioning from state
    i to state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the
    steady state probabilities, or None on failure
    """
    try:
        cols = P.shape[0]
        ans = np.ones((1, cols))
        # eq = np.matmul(ans, P)
        # s = np.array(np.arange(1, cols + 1))
        eq = np.vstack([P.T - np.identity(cols), ans])
        # va, vec = np.linalg .eig(P)
        results = np.zeros((cols, 1))
        results = np.vstack([results, np.array([1])])
        statetionary = np.linalg.solve(eq.T.dot(eq), eq.T.dot(results)).T
        # print(statetionary)
        # print(np.argwhere(statetionary < 0))
        if len(np.argwhere(statetionary < 0)) > 0:
            return None
        return statetionary
    except Exception as e:
        return None
