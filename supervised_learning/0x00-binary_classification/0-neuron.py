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
            raise ValueError("nx must be positive integer")

        self.nx = nx
        self.W = np.random.normal(0, 1, size=(1, nx))
        self.b = 0
        self.A = 0
