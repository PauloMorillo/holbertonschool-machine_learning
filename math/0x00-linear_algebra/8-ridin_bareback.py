def mat_mul(mat1, mat2):
    """This funtion return the answer of a matrix multiplication"""
    if mat1 and mat2:
        new_matrix = []
        rowsmat1l = len(mat1)
        colsmat1l = len(mat1[0])
        rowsmat2l = len(mat2)
        colsmat2l = len(mat2[0])
        mat2trans = [[mat2[j][i] for j in range(len(mat2))]
                     for i in range(len(mat2[0]))]
        if colsmat1l == rowsmat2l:
            for count1 in range(rowsmat1l):
                new_cols = list()
                for count2 in range(colsmat2l):
                    multvector = []
                    for count3 in range(colsmat1l):
                        mult = mat1[count1][count3] * mat2trans[count2][count3]
                        multvector = multvector + [mult]
                    new_cols = new_cols + [sum(multvector)]
                new_matrix = new_matrix + [new_cols]
            return new_matrix
    return None
