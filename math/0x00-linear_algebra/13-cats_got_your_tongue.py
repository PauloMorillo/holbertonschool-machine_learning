def np_cat(mat1, mat2, axis=0):
    """ This function returns the concatenation between two matrix by axis"""
    import numpy as np
    return np.concatenate((mat1, mat2), axis)
