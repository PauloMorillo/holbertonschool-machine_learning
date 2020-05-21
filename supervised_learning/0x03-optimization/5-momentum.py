#!/usr/bin/env python3
""" This module has the update_variables_momentum method"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    This method calculates the weighted
    moving average of a data set
    """
    v = np.multiply(beta1, v) + (np.multiply((1-beta1), grad))
    #print(v)
    var = var - np.multiply(alpha, v)
    return var, v
