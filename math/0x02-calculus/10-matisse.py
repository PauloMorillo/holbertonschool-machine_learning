#!/usr/bin/env python3
""" This module has a function to solve derivate polynomial """


def poly_derivative(poly):
    """ Funtion to solve polynomial derivate """
    if poly and type(poly) == list and len(poly) > 0:
        poly.pop(0)
        new_list = []
        for exp in range(len(poly)):
            print(new_list, poly)
            print("esto es exp", exp)
            new_list = new_list + [poly[exp] * (exp + 1)]
        return new_list
    return None
