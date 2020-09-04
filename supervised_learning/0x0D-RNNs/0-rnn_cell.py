#!/usr/bin/env python3
"""
This module has the RNNCell class
"""
import numpy as np


class RNNCell:
    """
    This class represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        All begins here
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        This method calculates de forward prop for one time-step
        x_t is a numpy.ndarray of shape (m, i) that contains the
        data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
        """
        h_next = np.tanh(np.dot(np.concatenate([h_prev, x_t],
                                               axis=1),
                                self.Wh) + self.bh)
        z = np.dot(h_next, self.Wy) + self.by
        # y = np.exp(z) / np.sum(np.exp(z), axis=0)
        e_x = np.exp(z - np.max(z))
        y = e_x / e_x.sum(axis=1, keepdims=True)
        # np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
        # softmax(np.dot(Wya, a_next) + by)
        return h_next, y
