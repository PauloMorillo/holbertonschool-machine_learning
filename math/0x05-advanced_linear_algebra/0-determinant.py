#!/usr/bin/env python3
"""
This module has the method determinant(matrix)
"""


def determinant(matrix):
    """
    This method calculates the determinant of a matrix
    """
    if type(matrix) is not list \
            or len(matrix) < 1 or \
            type(matrix[0]) is not list:
        raise TypeError('matrix must be a list of lists')

    rows_length = len(matrix)

    for row in matrix:
        if len(row) is not rows_length and len(row) is not 0:
            raise ValueError('matrix must be a square matrix')
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')

    if rows_length is 1:
        if len(matrix[0]) is 0:
            return 1
        return matrix[0][0]
    elif rows_length is 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = (a * d) - (b * c)
        return det
    else:
        det = 0
        for j0 in range(len(matrix[0])):
            new_ma = []
            for i in range(1, len(matrix)):
                row = []
                for j in range(len(matrix[i])):
                    if j is not j0:
                        row.append(matrix[i][j])
                new_ma.append(row)
            if j0 % 2 is not 0:
                factor = -1
            else:
                factor = 1
            second_fact = determinant(new_ma)
            det = det + (factor * (matrix[0][j0] * (second_fact)))
        return det
