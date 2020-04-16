#!/usr/bin/env python3
"""Function for use matrix operations with numpy"""


def np_elementwise(mat1, mat2):
    """ This function returns the add, sub, mul and div of a numpy array"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
