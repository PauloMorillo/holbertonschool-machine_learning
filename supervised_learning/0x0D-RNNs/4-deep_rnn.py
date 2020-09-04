#!/usr/bin/env python3
"""
This module has the method def deep_rnn(rnn_cells, X, h_0)
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    This method performs forward propagation for a deep RNN
    rnn_cells is a list of RNNCell instances of length l that will
    be used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a
    numpy.ndarray of shape (l, m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """

    h_next = h_0
    L, m, i = h_0.shape
    H = np.zeros((X.shape[0] + 1, L, m, i))
    H[0, :, :, :] = h_0
    Y = []
    for t in range(X.shape[0]):
        a_prev = X[t]
        for l in range(L):
            # print(l, L)
            # print(rnn_cells[l])
            # print(h_next[l], h_next[l].shape)
            h_next, y = rnn_cells[l].forward(H[t, l], a_prev)
            a_prev = h_next
            H[t + 1, l, :, :] = h_next
        Y.append(y)
    return H, np.array(Y)
