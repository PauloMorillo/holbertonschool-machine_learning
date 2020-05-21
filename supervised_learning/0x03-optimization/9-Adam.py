#!/usr/bin/env python3
""" This module has the update_variables_RMSProp method"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    This method updates a variable in place
    using the Adam optimization algorithm
    """
    v = (beta1 * v) + ((1 - beta1) * grad)
    vcorrec = v / (1 - (beta1 ** t))
    s = (beta2 * s) + ((1 - beta2) * grad ** 2)
    scorrec = s / (1 - beta2 ** t)
    var = var - (alpha * (vcorrec / (np.sqrt(scorrec) + epsilon)))
    return var, v, s
