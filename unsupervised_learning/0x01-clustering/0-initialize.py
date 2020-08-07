#!/usr/bin/env python3
"""
This module has the method initialize(X, k)::
"""
import numpy as np


def initialize(X, k):
    """
    This method iniziatializes cluster centroids for k-means
    X is a numpy.ndarray of shape (n, d) containing the dataset
    that will be used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    try:
        if type(k) is not int:
            return None
        lows = np.min(X, axis=0)
        highs = np.max(X, axis=0)
        centroids = np.random.uniform(lows, highs, size=(k, X.shape[1]))
        return centroids
    except Exception as e:
        return None
