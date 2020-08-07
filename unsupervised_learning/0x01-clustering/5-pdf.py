#!/usr/bin/env python3
"""
This module has the pdf(X, m, S)
"""
import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution
    X is a numpy.ndarray of shape (n, d) containing the data points whose
    PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
    distribution
    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values for each
    data point
    """
    x_m = X - m
    d = X.shape[1]
    a = 1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(S)))
    S_inv = np.linalg.inv(S)
    fac = np.einsum('...k,kl,...l->...', x_m, S_inv, x_m)
    b = np.exp(-fac / 2)
    P = a * b
    return np.maximum(P, 1e-300)
