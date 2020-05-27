#!/usr/bin/env python3
"""This module has the method l2_reg_cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """This method calculates the cost of a NN
    with L2 regularization"""
    constant = lambtha / (2 * m)
    l2sum = 0
    # print(weights)
    for lay in range(1, L + 1):
        w = weights['W{}'.format(lay)]
        sumwsq = np.linalg.norm(w)
        l2sum = l2sum + sumwsq
    return cost + (constant * l2sum)
