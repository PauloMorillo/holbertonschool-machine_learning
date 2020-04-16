def add_arrays(arr1, arr2):
    """ This function returns the add between two arrays"""
    if arr1 and arr2:
        if len(arr1) == len(arr2):
            a = 0
            new_vector = []
            while a < len(arr1):
                new_vector = new_vector + [arr1[a] + arr2[a]]
                a = a + 1
            return new_vector
    return None
