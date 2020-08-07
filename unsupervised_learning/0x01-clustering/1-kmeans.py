#!/usr/bin/env python3
"""
This module has the method kmeans(X, k, iterations=1000):
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    This method performs K-means on a dataset
    """
    try:
        if iterations <= 0:
            return None, None
        lows = np.min(X, axis=0)
        highs = np.max(X, axis=0)
        C = np.random.uniform(lows, highs, size=(k, X.shape[1]))
        C1 = C
        for iteration in range(iterations):
            D = np.linalg.norm(X[:, None] - C, axis=-1)
            min_d = np.argmin(D, axis=-1)
            for cluster in range(k):
                pos = np.argwhere(min_d == cluster)
                if len(pos > 0):
                    c = np.sum(X[pos], axis=0) / len(X[pos])
                    C[cluster] = c
                else:
                    c = np.random.uniform(lows, highs, size=(1, X.shape[1]))
                    C[cluster] = c
            if (C1 == C).all:
                return C1, min_d
        D = np.linalg.norm(X[:, None] - C, axis=-1)
        min_d = np.argmin(D, axis=-1)
        return C, min_d
    except Exception as e:
        return None, None
