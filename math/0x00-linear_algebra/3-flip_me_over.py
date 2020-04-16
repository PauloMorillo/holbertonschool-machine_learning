def matrix_transpose(matrix):
    """ This function returns the transpose of a 2D  matrix"""
    new_matrix = [[matrix[j][i] for j in range(len(matrix))]
                  for i in range(len(matrix[0]))]
    return new_matrix
