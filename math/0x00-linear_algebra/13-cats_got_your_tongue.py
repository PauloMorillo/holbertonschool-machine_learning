#!/usr/bin/env python3
"""Function for concat 2 matrix with numpy"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ This function returns the concatenation between two matrix by axis"""
    return np.concatenate((mat1, mat2), axis)
