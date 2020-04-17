#!/usr/bin/env python3
""" Funtion for concatenate 2 matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ This funtion return a concatenated matrix"""
    new_matrix = []
    mat2n = []
    mat1n = []
    for rows in mat1:
        new_cols = []
        for cols in rows:
            new_cols = new_cols + [cols]
        mat1n = mat1n + [new_cols]
        new_matrix = new_matrix + [new_cols]

    for rows in mat2:
        new_cols = []
        for cols in rows:
            new_cols = new_cols + [cols]
        mat2n = mat2n + [new_cols]

    # print(new_matrix)

    if axis is 0:
        # print("por aqui pase")
        # for rows in mat2n:
        if len(mat2[0]) != len(mat1[0]):
            return None
        return mat1n + mat2n

    if axis is 1:
        for pos in range(len(mat2)):
            if len(new_matrix) == len(mat2n):
                new_matrix[pos].append(mat2n[pos][0])
            else:
                return None
        return new_matrix


"""


            new_cols = []
            for cols in rows:
                new_cols.append(cols)
            new_matrix.append(new_cols)






    for rows in mat2:
        new_cols = []
        pos = 0
        for cols in rows:
            new_cols.append(cols)
            if axis is 0:
                if len(new_cols) == len(new_matrix[len(new_matrix) - 1]):
                    new_matrix.append(new_cols)
                else:
                    return None
            if axis is 1:
                if len(new_cols) is 1:
                    if len(mat1) == len(mat2):
                        print(pos)
                        new_matrix[pos].append(new_cols[pos])
                        pos = pos + 1
                    else:
                        return None
                else:
                    return None
        return new_matrix
    return None
"""
