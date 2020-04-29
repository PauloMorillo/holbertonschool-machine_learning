#!/usr/bin/env python3
""" This module has the class poisson """


def factorial(n):
    """ function to do a factorial """
    ans = 1
    while (n >= 1):
        ans = ans * n
        n = n - 1
    return ans


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

    def pmf(self, k):
        """ Method to calculate the pmf """
        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0
        e = 2.7182818285
        return ((self.lambtha**k)*(e**(-self.lambtha)))/factorial(k)

    def cdf(self, k):
        """ This method calculates the CDF """
        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0
        e = 2.7182818285
        cdf = 0
        for k1 in range(k + 1):
            cdf = cdf + self.pmf(k1)
        return cdf
