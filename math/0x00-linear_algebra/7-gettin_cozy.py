def cat_matrices2D(mat1, mat2, axis=0):
    """ This funtion return a concatenated matrix"""
    if mat1 and mat2:
        new_matrix = []
        for rows in mat1:
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
                        new_matrix[pos].append(new_cols[0])
                        pos = pos + 1
                    else:
                        return None
                else:
                    return None
        return new_matrix
    return None
