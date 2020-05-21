#!/usr/bin/env python3
""" This module has the batch_norm method"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of
    a neural network using batch normalization
    """
    znorm = (Z - np.mean(Z, axis=0)) / np.sqrt(np.var(Z, axis=0) + epsilon)
    z = np.multiply(gamma, znorm) + beta
    return z
