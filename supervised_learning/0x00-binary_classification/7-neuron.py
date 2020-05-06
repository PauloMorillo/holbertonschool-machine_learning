#!/usr/bin/env python3
""" This module has the Neuron """

import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """ This method evaluates the neuron's predictions """
        A = self.forward_prop(X)
        nA = np.where(A <= 0.5, 0, 1)
        return (nA, self.cost(Y, A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """This method Calculates one pass of gradient descent on the neuron"""
        dz = A - Y
        dw = np.matmul(dz, X.T)/(len(A.T))
        db = np.sum((dz)) / (len(A.T))
        self.__b = self.__b - ((db) * alpha)
        self.__W = self.__W - (alpha * dw)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ This method trains the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            A, c = self.evaluate(X, Y)
            if verbose:
                if i == step or i == 0 or i == iterations:
                    print("Cost after {} iterations: {}".format(i, c))
            if graph:
                if i == step or i == 0 or i == iterations:
                    plt.xlabel('iteration')
                    plt.ylabel('cost')
                    plt.title('Training Cost')
                    plt.plot(c, 'b')
                    plt.show()
        return (self.evaluate(X, Y))
