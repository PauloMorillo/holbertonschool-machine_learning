#!/usr/bin/env python3
"""This module has the class MultiNormal"""

import numpy as np


class MultiNormal():
    """
    This class represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """All begins here"""
        if type(data) is not np.ndarray or np.ndim(data) is not 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = np.array([np.mean(data, axis=1)]).T
        resta = data - self.mean
        self.cov = np.dot(data, resta.T) / (data.shape[1] - 1)
