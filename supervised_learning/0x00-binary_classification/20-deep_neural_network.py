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

    def forward_prop(self, X):
        """This method calculates forward prop for DNN"""
        self.__cache['A0'] = X
        for lay in range(self.__L):
            w = self.__weights['W{}'.format(lay + 1)]
            b = self.__weights['b{}'.format(lay + 1)]
            x = np.matmul(self.__cache['A{}'.format(lay)].T, w.T).T + b
            a = 1 / (1 + (np.exp(-x)))
            self.__cache['A{}'.format(lay + 1)] = a
        return a, self.__cache

    def cost(self, Y, A):
        """ This method calculates cost of the DDN model with logistic reg"""
        term1 = (-1 / (len(A.T)))
        costf = term1 * ((Y * (np.log(A))) + ((1 - Y) * np.log(1.0000001 - A)))
        return np.sum(costf)

    def evaluate(self, X, Y):
        """ This method evaluates the DNN's predictions """
        A, A2 = self.forward_prop(X)
        nA = np.where(A <= 0.5, 0, 1)
        return (nA, self.cost(Y, A))
