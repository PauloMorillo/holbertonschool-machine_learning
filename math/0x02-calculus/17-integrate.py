#!/usr/bin/env python3
"""This module has a function for calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """ Funtion solve polynomial integrals """
    if poly and type(poly) == list and len(poly) > 0 and type(C) == int:
        new_list = [0]
        for exp in range(len(poly)):
            if type(poly[exp]) == int:
                newesc = (int(poly[exp] * (exp + 1)) / int(exp + 1))
                # if (newesc % (exp + 1)) == 0:
                new_list = new_list + [(int(newesc) / int(exp + 1))]
                # else:
                # new_list = new_list + [(int(newesc) / int(exp + 1))]
            else:
                return None
        return new_list
    return None
