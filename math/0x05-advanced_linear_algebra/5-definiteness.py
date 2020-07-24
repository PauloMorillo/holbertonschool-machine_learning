#!/usr/bin/env python3
"""
This module has the method definiteness(matrix)
"""

import numpy as np


def definiteness(matrix):
    """
    This method calculates the definiteness of a matrix
    """

    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) < 2:
        return None
    if matrix.shape[0] is not matrix.shape[1]:
        return None
    eigen_val, eigen_vec = np.linalg.eig(matrix)
    pos_positive = np.where(eigen_val > 0)
    pos_negative = np.where(eigen_val < 0)
    pos_zeros = np.where(eigen_val == 0)
    # print(eigen_val)
    # print(pos_positive)
    if len(pos_positive[0]) == len(eigen_val):
        return 'Positive definite'
    if len(pos_negative[0]) == len(eigen_val):
        return 'Negative definite'
    if len(pos_negative[0]) > 0 and len(pos_zeros[0]) > 0:
        return 'Negative semi-definite'
    if len(pos_positive[0]) > 0 and len(pos_zeros[0]) > 0:
        return 'Positive semi-definite'
    return 'Indefinite'
