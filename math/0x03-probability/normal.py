#!/usr/bin/env python3
""" This module has the class normal """


class Normal():
    """ This class is to represent a normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ All begins here """

        self.data = []
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            sn = []
            num = 0
            for x in data:
                num1 = ((x - self.mean)**2)
                num = num + num1
            s2 = num/len(data)
            self.stddev = (s2)**(1/2)
        else:
            if stddev >= 1:
                self.stddev = float(stddev)
            else:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)

    def z_score(self, x):
        """ Method to return the z-score """
        print(x)
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Method returns correspond x-values for z"""
        print(z)
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ Method to calculate the pdf """
        if x < 0:
            return 0
        e = 2.7182818285
        pi = 3.1415926536
        term1 = 1 / (self.stddev * ((2 * pi) ** (1 / 2)))
        term2 = e ** ((-(1 / 2) * (((x - self.mean)/self.stddev) ** 2)))
        return term1 * term2

    def cdf(self, x):
        """ This method calculates the CDF """
        if type(x) is not int:
            x = int(x)
        if x < 0:
            return 0
        e = 2.7182818285
        return (1 - (e ** ((-self.lambtha)*x)))
