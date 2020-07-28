#!/usr/bin/env python3
"""
Multivariate Normal distribution
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :return: mean, cov
    """
    d = X.shape[0]
    n = X.shape[1]
    mean = np.mean(X, axis=1).reshape(d, 1)
    X = X - mean
    cov = ((np.dot(X, X.T)) / (n - 1))
    return mean, cov


class MultiNormal():
    """
    Multinormal Class
    """

    def __init__(self, data):
        """
        class constructor
        :param data: numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) is not 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean, self.cov = mean_cov(data)

    def pdf(self, x):
        """
        calculates the PDF at a data point
        :param x: x is a numpy.ndarray of shape (d, 1) containing the data
        point whose PDF should be calculated
            d is the number of dimensions of the Multinomial instance
        :return: the value of the PDF
        """
        if x is None or type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        x_m = x - self.mean
        result = (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))) *
                  np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))
        return result[0][0]
