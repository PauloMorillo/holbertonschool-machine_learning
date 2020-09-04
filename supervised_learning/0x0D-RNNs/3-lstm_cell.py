#!/usr/bin/env python3
"""
This module has the LSTMCell class
"""
import numpy as np


def sigmoid(z):
    """
    This method calculates the sigmoid
    """
    return 1 / (1 + (np.exp(-z)))


class LSTMCell:
    """
    This class performs a LSTMCell
    """

    def __init__(self, i, h, o):
        """
        All begins here
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
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
        # Compute values for ft (forget gate), it (update gate),
        # cct (candidate value), c_next (cell state),
        # ot (output gate), a_next (hidden state) (≈6 lines)
        ft = sigmoid(np.dot(concat, self.Wf) + self.bf)  # forget gate
        it = sigmoid(np.dot(concat, self.Wu) + self.bu)  # update gate
        cct = np.tanh(np.dot(concat, self.Wc) + self.bc)  # candidate value
        c_next = (it * cct) + (ft * c_prev)  # cell state
        ot = sigmoid(np.dot(concat, self.Wo) + self.bo)  # output gate
        h_next = ot * np.tanh(c_next)  # hidden state
        z = np.dot(h_next, self.Wy) + self.by
        e_x = np.exp(z - np.max(z))
        y = e_x / e_x.sum(axis=1, keepdims=True)
        return h_next, c_next, y
