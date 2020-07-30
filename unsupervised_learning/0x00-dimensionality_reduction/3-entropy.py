#!/usr/bin/env python3
""" This module has the method HP(Di, beta):"""

import numpy as np


def HP(Di, beta):
    """
    Di is a numpy.ndarray of shape (n - 1,) containing the pairwise
    distances between a data point and all other points except itself
    n is the number of data points
    beta is a numpy.ndarray of shape (1,) containing the beta value
    for the Gaussian distribution
    Returns: (Hi, Pi)
    Hi: the Shannon entropy of the points
    Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities
    of the points
    """
    Pi = np.exp(-Di * beta) / np.sum(np.exp(-Di * beta))
    Hi = np.sum(-Pi * np.log2(Pi))
    return Hi, Pi
