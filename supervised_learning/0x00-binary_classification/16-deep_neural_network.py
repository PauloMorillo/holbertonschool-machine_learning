#!/usr/bin/env python3
""" This module has a deep neural network class"""

import numpy as np


class DeepNeuralNetwork():
    """ This is the class to instance a DNN """

    def __init__(self, nx, layers):
        """ All begins here """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        weights = {}
        lay = 0
        while lay < len(layers):
            if layers[lay] < 1:
                raise TypeError("layers must be a list of positive integers")
            if lay is not 0:
                cols = layers[lay - 1]
            else:
                cols = nx
            w = np.random.normal(size=(layers[lay], cols)) * np.sqrt(2 / cols)
            weights['W{}'.format(lay + 1)] = w
            weights['b{}'.format(lay + 1)] = np.zeros((layers[lay], 1))
            lay = lay + 1
        self.weights = weights
        self.L = len(layers)
        self.cache = {}
