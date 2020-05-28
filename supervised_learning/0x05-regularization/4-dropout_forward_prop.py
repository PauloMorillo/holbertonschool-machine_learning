#!/usr/bin/env python3
"""This module has the method l2_reg_create_layer(cost):"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """This method conducts forward propagation using Dropout"""
    cache = dict()
    cache['A0'] = X
    for lay in range(L):
        w = weights['W{}'.format(lay + 1)]
        b = weights['b{}'.format(lay + 1)]
        z = np.matmul(cache['A{}'.format(lay)].T, w.T).T + b
        a = np.sinh(z) / np.cosh(z)
        # print(a.shape)
        d = np.random.binomial(1, keep_prob, (a.shape[0], a.shape[1]))
        a = np.multiply(a, d) / keep_prob
        if lay == (L - 1):
            denominator = np.sum(np.exp(z), axis=0, keepdims=True)
            a = (np.exp(z)) / denominator
        cache['A{}'.format(lay + 1)] = a
        if lay < L - 1:
            cache['D{}'.format(lay + 1)] = d
    return cache
