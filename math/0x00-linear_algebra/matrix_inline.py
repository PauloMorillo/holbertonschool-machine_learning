def matrix_transpose(matrix):
    """ This function returns the transpose of a 2D  matrix"""
    new_matrix = []
    for vector in matrix:
        for item in vector:
            new_matrix = new_matrix + [item]

    return new_matrix
