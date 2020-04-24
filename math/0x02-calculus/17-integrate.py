#!/usr/bin/env python3
"""This module has a function for calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """ Funtion solve polynomial integrals """
    if poly and type(poly) == list:
        if len(poly) > 0 and type(C) == int:
            ans = [C]
            if len(poly) == 1 and poly[0] == 0:
                return [C]
            for exp in range(len(poly)):
                if type(poly[exp]) == int:
                    newesc = (poly[exp] * (exp + 1) / (exp + 1))
                    if (newesc % (exp + 1)) == 0:
                        ans = ans + [(newesc // (exp + 1)).__round__()]
                    else:
                        ans = ans + [(newesc / (exp + 1))]
                else:
                    return None
            return ans
    return None
