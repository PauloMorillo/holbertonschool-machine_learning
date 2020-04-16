def cat_arrays(arr1, arr2):
    """ This function returns the concatanation of two arrays"""
    if arr1 and arr2:
        new_arr = []
        for item in arr1:
            new_arr = new_arr + [item]
        for item in arr2:
            new_arr = new_arr + [item]
        return new_arr
