#!/usr/bin/env python3
""" Module with function matrix shape"""


def matrix_shape(matrix):
    """ This function return the size of a matrix"""
    size = []
    if type(matrix) is list:
        return [len(matrix)] + matrix_shape(matrix[0])
    return size
