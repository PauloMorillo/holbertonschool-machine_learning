#!/usr/bin/env python3
"""This module has a function for calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """ Funtion solve polynomial integrals """
    if poly and type(poly) == list:
        if len(poly) > 0 and type(C) in [int, float]:
            ans = [C]
            for exp in range(len(poly)):
                if type(poly[exp]) == int:
                    newesc = (int(poly[exp] * (exp + 1)) / int(exp + 1))
                    if (newesc % (exp + 1)) == 0:
                        ans = ans + [(int(newesc) // int(exp + 1))]
                    else:
                        ans = ans + [(int(newesc) / int(exp + 1)).__round__(2)]
                else:
                    return None
            return ans
    return None
