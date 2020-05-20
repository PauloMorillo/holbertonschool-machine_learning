#!/usr/bin/env python3
""" This module has the method normalization_constants"""
import numpy as np


def normalization_constants(X):
    """
    This method calculates the normalization (standardization)
    constants of a matrix
    """
    mean = np.mean(X, axis=0)
    variance = np.std(X, axis=0)
    return mean, variance
