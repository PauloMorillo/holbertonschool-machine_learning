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

    def pdf(self, x):
        """This method calculates the PDF at a data point"""
        if type(x) is not np.ndarray:
            raise TypeError('x must by a numpy.ndarray')
        d = self.cov.shape[0]
        if len(x.shape) is not 2 or x.shape[0] is not d or \
                x.shape[1] is not 1:
            raise ValueError('x mush have the shape ({}, 1)'.format(d))
        mean = self.mean
        cov = self.cov
        factor1 = 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov))
        factor2 = np.exp(-(np.linalg.solve(cov, x - mean).T.dot(x - mean)) / 2)
        return (factor1 * factor2)[0][0]
