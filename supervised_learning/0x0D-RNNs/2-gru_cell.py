#!/usr/bin/env python3
"""
This module has the GRUCell class
"""
import numpy as np


def sigmoid(z):
    """
    This method calculates the sigmoid
    """
    return 1 / (1 + (np.exp(-z)))


class GRUCell:
    """
    This class represents a GRUCell
    """

    def __init__(self, i, h, o):
        """
        All begins here
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
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
        concat = np.concatenate([h_prev, x_t], axis=1)
        r = sigmoid(np.dot(concat, self.Wr) + self.br)  # reset
        z = sigmoid(np.dot(concat, self.Wz) + self.bz)  # update
        copy_con = np.concatenate([h_prev * r, x_t], axis=1)
        cct = np.tanh(np.dot(copy_con, self.Wh) + self.bh)
        h_next = z * cct + (1 - z) * h_prev  # cell state
        z_y = np.dot(h_next, self.Wy) + self.by  # hidden state
        e_x = np.exp(z_y - np.max(z_y))
        y = e_x / e_x.sum(axis=1, keepdims=True)
        return h_next, y
