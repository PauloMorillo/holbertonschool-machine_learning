#!/usr/bin/env python3
""" This module has the update_variables_RMSProp method"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    This method updates a variable using
    the RMSProp optimization algorithm
    """
    s = np.multiply(beta2, s) + (np.multiply((1 - beta2), grad ** 2))
    var = var - np.multiply(alpha, (grad / (np.sqrt(s) + epsilon)))
    return var, s
