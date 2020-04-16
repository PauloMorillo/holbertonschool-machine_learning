#!/usr/bin/env python3
""" Function to concatanate two arrays"""


def cat_arrays(arr1, arr2):
    """ This function returns the concatanation of two arrays"""
    new_arr = []
    for item in arr1:
        new_arr = new_arr + [item]
    for item in arr2:
        new_arr = new_arr + [item]
    return new_arr
