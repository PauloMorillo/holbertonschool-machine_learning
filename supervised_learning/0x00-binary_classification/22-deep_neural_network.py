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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """This method Calculates one pass of gradient descent on the DNN"""
        lay = self.__L
        w = self.__weights
        while lay > 0:
            A1 = cache['A{}'.format(lay)]
            if lay == self.__L:
                dz = A1 - Y
            else:
                g1 = (A1 * (1 - A1))
                dz = np.matmul(w['W{}'.format(lay + 1)].T, dz) * g1
            dw = np.matmul(dz, cache['A{}'.format(lay - 1)].T) / (len(Y.T))
            db = np.sum(dz, axis=1, keepdims=True) / (len(Y.T))
            fix1 = (db * alpha)
            fix2 = (dw * alpha)
            self.__weights['b{}'.format(lay)] = w['b{}'.format(lay)] - fix1
            self.__weights['W{}'.format(lay)] = w['W{}'.format(lay)] - fix2
            lay = lay - 1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ This method trains the DNN """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return (self.evaluate(X, Y))


"""
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / (len(Y.T))
        db2 = np.sum(dz2, axis=1, keepdims=True) / (len(Y.T))
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / (len(A1.T))
        db1 = np.sum(dz1, axis=1, keepdims=True) / (len(Y.T))
        self.__b1 = self.__b1 - ((db1) * alpha)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b2 = self.__b2 - ((db2) * alpha)
        self.__W2 = self.__W2 - (alpha * dw2)
"""
