#!/usr/bin/env python3
"""This module has the class GaussianProcess"""
import numpy as np


class GaussianProcess:
    """
    this class represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, li=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        black-box function
        K, representing the current covariance kernel matrix for the
        Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.li = li
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        This method calculates the kernel covariance with RBF
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) \
            - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.li ** 2 * sqdist)

    def predict(self, X_s):
        """
        This method predicts the mean and standard deviation of points
        in a Gaussian process
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # Equation (4)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)

        # Equation (5)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu_s.T[0], np.diagonal(cov_s)

    def update(self, X_new, Y_new):
        """
        This method updates a Gaussian Process
        X_new is a numpy.ndarray of shape (1,) that represents
        the new sample point
        Y_new is a numpy.ndarray of shape (1,) that represents
        the new sample function value
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
