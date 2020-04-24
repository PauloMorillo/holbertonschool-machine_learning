#!/usr/bin/env python3
""" Module with function of summation square function"""


def summation_i_squared(n):
    """ Function to return the summation of square function """
    if n and type(n) == int and n >= 1:
        ans = ((n * (n + 1) * ((2 * n) + 1)) / 6).__round__()
        return ans
    return None
