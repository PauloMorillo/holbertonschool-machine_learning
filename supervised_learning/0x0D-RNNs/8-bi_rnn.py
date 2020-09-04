#!/usr/bin/env python3
"""
This module has the BidirectionalCell class
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    This method performs forward propagation for a bidirectional RNN
    bi_cell is an instance of BidirectinalCell that will be used for
    the forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction, given as
    a numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction, given as
    a numpy.ndarray of shape (m, h)
    Returns: H, Y
    """
    h_prev = h_0
    h_next = h_t
    H = np.zeros((X.shape[0], h_0.shape[0], h_0.shape[1] * 2))
    H_l = []
    H_r = []
    for t in range(X.shape[0]):
        h_l = bi_cell.forward(h_prev, X[t])
        h_prev = h_l
        H_l.append(h_prev)
    for t_r in range(X.shape[0] - 1, -1, -1):
        h_r = bi_cell.backward(h_next, X[t_r])
        h_next = h_r
        H_r.append(h_r)
    H_R = []
    H_l = np.array(H_l)
    H_r = np.flip(H_r, axis=0)
    # print("estoooooo es hr", H_r.shape)
    H = np.concatenate([H_l, H_r], axis=-1)
    y = bi_cell.output(H)
    return H, y
