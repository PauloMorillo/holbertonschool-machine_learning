#!/usr/bin/env python3
"""This module has the method l2_reg_gradient_descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """This method updates the weights and biases of
    a neural network using gradient descent with
    L2 regularization"""
    lay = L
    w = weights
    m = Y.shape[1]
    while lay > 0:
        A1 = cache['A{}'.format(lay)]
        if lay == L:
            dz = A1 - Y
        else:
            g1 = (1 - np.power(A1, 2))
            dz = np.matmul(w['W{}'.format(lay + 1)].T, dz) * g1
        dw = np.matmul(dz, cache['A{}'.format(lay - 1)].T) / (Y.shape[1])
        db = np.sum(dz, axis=1, keepdims=True) / (Y.shape[1])
        fix1 = (db * alpha)
        fix2 = alpha * (dw + ((lambtha / m) * w['W{}'.format(lay)]))
        if lay < L:
            weights['b{}'.format(lay + 1)] = blast
            weights['W{}'.format(lay + 1)] = wlast
            blast = w['b{}'.format(lay)] - fix1
            wlast = w['W{}'.format(lay)] - fix2

        else:
            blast = w['b{}'.format(lay)] - fix1
            wlast = w['W{}'.format(lay)] - fix2
        lay = lay - 1

    weights['b{}'.format(lay + 1)] = blast
    weights['W{}'.format(lay + 1)] = wlast
