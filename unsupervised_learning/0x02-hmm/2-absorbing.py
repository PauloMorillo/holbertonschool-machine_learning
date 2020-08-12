#!/usr/bin/env python3
"""This module has the method absorbing(P)"""
import numpy as np


def absorbing(P):
    """
    this method determines if a markov chain is absorbing
    P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    try:
        diagonal = np.diagonal(P)
        abs_state = np.argwhere(diagonal == 1)
        if len(abs_state) > 0:
            # print(diagonal)
            return True
        else:
            return False
    except Exception as e:
        return False
