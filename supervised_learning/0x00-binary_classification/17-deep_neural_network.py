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
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        weights = {}
        lay = 0
        while lay < len(layers):
            if layers[lay] < 1:
                raise ValueError("layers must be a list of positive integers")
            if lay is not 0:
                cols = layers[lay - 1]
            else:
                cols = nx
            w = np.random.normal(size=(layers[lay], cols)) * np.sqrt(2 / cols)
            weights['W{}'.format(lay + 1)] = w
            weights['b{}'.format(lay + 1)] = np.zeros((layers[lay], 1))
            lay = lay + 1
        self.__weights = weights
        self.__L = len(layers)
        self.__cache = {}

    @property
    def weights(self):
        """This is a getter for weights attribute"""
        return self.__weights

    @property
    def L(self):
        """This is a getter for L attribute"""
        return self.__L

    @property
    def cache(self):
        """This is a getter for cache attribute"""
        return self.__cache
