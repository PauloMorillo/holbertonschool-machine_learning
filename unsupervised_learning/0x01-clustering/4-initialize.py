#!/usr/bin/env python3
"""
This module has the method initialize(X, k):
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors
    for each cluster, initialized evenly
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster, initialized with K-means
    S is a numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster, initialized as identity matrices
    """
    d = X.shape[1]
    C, _ = kmeans(X, k)
    pi = np.ones((k,)) * (1 / k)
    S = np.identity(d)
    S = np.full((k, d, d), S)
    return pi, C, S
