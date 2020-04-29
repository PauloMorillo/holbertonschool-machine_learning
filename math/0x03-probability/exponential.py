#!/usr/bin/env python3
""" This module has the class Exponencial """


def factorial(n):
    """ function to do a factorial """
    ans = 1
    while (n >= 1):
        ans = ans * n
        n = n - 1
    return ans


class Exponential():
    """ This class is to represent a exponencial distribution """

    def __init__(self, data=None, lambtha=1.):
        """ All begins here """

        self.data = []
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1/(sum(data) / len(data))
        else:
            if lambtha >= 1:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')

    def pdf(self, x):
        """ Method to calculate the pdf """
        if x < 0:
            return 0
        e = 2.7182818285
        return (self.lambtha * (e ** ((-self.lambtha)*x)))

    def cdf(self, x):
        """ This method calculates the CDF """
        if x < 0:
            return 0
        e = 2.7182818285
        return (1 - (e ** ((-self.lambtha)*x)))
