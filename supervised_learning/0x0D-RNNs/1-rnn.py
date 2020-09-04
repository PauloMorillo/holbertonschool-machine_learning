#!/usr/bin/env python3
"""
This module has the method def rnn(rnn_cell, X, h_0)
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """This method performs forward propagation for a simple RNN"""
    h_next = h_0
    H = np.zeros((X.shape[0] + 1, h_0.shape[0], h_0.shape[1]))
    H[0, :, :] = h_0
    Y = []
    for t in range(X.shape[0]):
        h_next, y = rnn_cell.forward(h_next, X[t])
        H[t + 1, :, :] = (h_next)
        Y.append(y)
    return H, np.array(Y)
