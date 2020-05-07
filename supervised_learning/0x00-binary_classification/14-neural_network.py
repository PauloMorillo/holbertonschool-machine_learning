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
        """ This method evaluates the NN's predictions """
        A1, A2 = self.forward_prop(X)
        nA = np.where(A2 <= 0.5, 0, 1)
        return (nA, self.cost(Y, A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """This method Calculates one pass of gradient descent on the NN"""
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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ This method trains the NN """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return (self.evaluate(X, Y))
