#!/usr/bin/env python3
"""
This module has the BidirectionalCell class
"""
import numpy as np


class BidirectionalCell:
    """
    This class represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        All begins here
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
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
        h1 = np.tanh(np.dot(np.concatenate([h_prev, x_t],
                                           axis=1),
                            self.Whf) + self.bhf)
        h2 = np.tanh(np.dot(np.concatenate([h_prev, x_t],
                                           axis=1),
                            self.Whb) + self.bhb)
        # print(h1.shape, h2.shape)
        # z = np.dot(np.concatenate([h2, h1], axis=0), self.Wy) + self.by
        # y = np.exp(z) / np.sum(np.exp(z), axis=0)
        # e_x = np.exp(z - np.max(z))
        # y = e_x / e_x.sum(axis=1, keepdims=True)
        # np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
        # softmax(np.dot(Wya, a_next) + by)
        return h1
