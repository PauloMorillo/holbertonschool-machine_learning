#!/usr/bin/env python3
""" This module has a function to solve derivate polynomial """


def poly_derivative(poly):
    """ Funtion to solve polynomial derivate """
    if poly and type(poly) == list and len(poly) > 0 and type(poly[0]) == int:
        if len(poly) == 1:
            return [0]
        poly.pop(0)
        new_list = []
        for exp in range(len(poly)):
            if type(poly[exp]) == int:
                new_list = new_list + [poly[exp] * (exp + 1)]
            else:
                return None
        return new_list
    return None
