#!/usr/bin/env python3
""" This module has the class Binomial """


def factorial(n):
    """ function to do a factorial """
    ans = 1
    while (n >= 1):
        ans = ans * n
        n = n - 1
    return ans


class Binomial():
    """ This class is to represent a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ All begins here """

        self.data = []
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            num = 0
            for daticos in data:
                num1 = (daticos - mean) ** 2
                num = num + num1
            s2 = num/len(data)
            p = ((s2 / mean) - 1) / -1
            n = mean/p
            self.n = n.__round__()
            self.p = mean / self.n
        else:
            if n >= 1:
                self.n = int(n)
            else:
                raise ValueError('n must be a positive value')
            if p > 0 and p < 1:
                self.p = p
            else:
                raise ValueError('p must be greater than 0 and less than 1')

    def pmf(self, k):
        """ Method to calculate the pmf """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        p = self.p
        q = 1 - p
        n = self.n
        term1 = (factorial(n) / (factorial(k) * factorial(n-k)))
        pmf = term1 * (p ** k) * (q ** (n - k))
        return pmf

    def cdf(self, k):
        """ This method calculates the CDF """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for k1 in range(k + 1):
            cdf = cdf + self.pmf(k1)
        return cdf
