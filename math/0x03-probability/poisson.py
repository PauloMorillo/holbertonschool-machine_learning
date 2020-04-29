#!/usr/bin/env python3
""" This module has the class poisson """


class Poisson():
    """ This class is to represent a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ All begins here """

        self.data = []
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)
        else:
            if lambtha >= 1:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')
