#!/usr/bin/env python3
""" This module has the Neuron """

import numpy as np


class Neuron():
    """ This is the class to instance a neuron """

    def __init__(self, nx):
        """ All begins here """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter to return value of W """
        return self.__W

    @property
    def b(self):
        """ Getter to return value of b """
        return self.__b

    @property
    def A(self):
        """ Getter to return value of A """
        return self.__A

    def forward_prop(self, X):
        """ This Method calculates the forward propagation of the neuron """
        x = np.matmul(X.T, self.__W.T).T + self.__b
        self.__A = 1 / (1 + (np.exp(-x)))
        return self.__A

    def cost(self, Y, A):
        """ This method calculates cost of the model with logistic regrsion"""
        term1 = (-1 / (len(A.T)))
        costf = term1 * ((Y * (np.log(A))) + ((1 - Y) * np.log(1.0000001 - A)))
        return costf.sum()
