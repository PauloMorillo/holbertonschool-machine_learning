#!/usr/bin/env python3
""" This module has the method
convolve_grayscale_padding(images, kernel, padding):
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """This method performs a convolution on grayscale
    images with custom padding
    """
    images = np.pad(images, ((0, 0), (padding[0], padding[0]),
                             (padding[1], padding[1])),
                    'constant', constant_values=0)
    rows_im = images.shape[1]
    cols_im = images.shape[2]
    rows_k = kernel.shape[0]
    cols_k = kernel.shape[1]
    new_rows = rows_im - rows_k + 1
    new_cols = cols_im - cols_k + 1
    # print(new_cols, new_rows)
    new = np.ones((images.shape[0], new_rows, new_cols))
    # print(new.shape)
    # print(new)
    for i in range(new.shape[1]):
        for j in range(new.shape[2]):
            ans = images[:, i:rows_k + i, j:cols_k + j] * kernel
            # print(ans.shape)
            # print(ans.T.shape)
            # print(np.sum(ans, axis=2).shape)
            mat = np.sum(np.sum(ans.T, axis=1), axis=0)
            new[:, i, j] = mat
    return new
