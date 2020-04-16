def add_matrices2D(mat1, mat2):
    """ This function returns the add between two matrices"""
    if mat1 and mat2:
        if len(mat1) == len(mat2):
            a = 0
            new_mat = []
            while a < len(mat1):
                if len(mat1[a]) == len(mat2[a]):
                    b = 0
                    new_vec = []
                    while b < len(mat1[a]):
                        new_vec = new_vec + [mat1[a][b] + mat2[a][b]]
                        b = b + 1
                    new_mat = new_mat + [new_vec]
                else:
                    return None
                a = a + 1
            return new_mat
    return None
