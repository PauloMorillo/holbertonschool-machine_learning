#!/usr/bin/env python3
""" This module has a neural network class"""

import numpy as np


class NeuralNetwork():
    """ This is the class to instance a neural network """

    def __init__(self, nx, nodes):
        """ All begins here """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ This is a getter for W1"""
        return self.__W1

    @property
    def W2(self):
        """ This is a getter for W1"""
        return self.__W2

    @property
    def b1(self):
        """ This is a getter for W1"""
        return self.__b1

    @property
    def b2(self):
        """ This is a getter for W1"""
        return self.__b2

    @property
    def A1(self):
        """ This is a getter for W1"""
        return self.__A1

    @property
    def A2(self):
        """ This is a getter for W1"""
        return self.__A2

    def forward_prop(self, X):
        """ This method calculates de forward propagation"""
        x = np.matmul(X.T, self.__W1.T).T + self.__b1
        self.__A1 = 1 / (1 + (np.exp(-x)))
        x2 = np.matmul(self.__A1.T, self.__W2.T).T + self.__b2
        self.__A2 = 1 / (1 + (np.exp(-x2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ This method calculates cost of the model with logistic regrsion"""
        term1 = (-1 / (len(A.T)))
        costf = term1 * ((Y * (np.log(A))) + ((1 - Y) * np.log(1.0000001 - A)))
        return np.sum(costf)

    def evaluate(self, X, Y):
        """ This method evaluates the neuron's predictions """
        A1, A2 = self.forward_prop(X)
        nA = np.where(A2 <= 0.5, 0, 1)
        return (nA, self.cost(Y, A2))
