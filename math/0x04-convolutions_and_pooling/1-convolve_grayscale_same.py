#!/usr/bin/env python3
""" This module has the method
convolve_grayscale_same(images, kernel)
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """This method performs a same
    convolution on grayscale images
    """
    rows_im = images.shape[1]
    cols_im = images.shape[2]
    rows_k = kernel.shape[0]
    cols_k = kernel.shape[1]
    pw = max((cols_im - cols_k) * 1 + (cols_k - cols_im), 0)
    ph = max((rows_im - rows_k) * 1 + (rows_k - rows_im), 0)
    pw_r = pw // 2
    ph_t = ph // 2
    pw_l = pw - pw_r
    ph_b = ph - ph_t
    images = np.pad(images, ((0, 0), (ph_t, ph_b), (pw_l, pw_r)),
                    'constant', constant_values=0)
    rows_im = images.shape[1]
    cols_im = images.shape[2]
    new_rows = (rows_im - rows_k) + 1
    new_cols = (cols_im - cols_k) + 1
    # print(new_cols, new_rows)
    new = np.zeros((images.shape[0], new_rows, new_cols))
    # print(new.shape)
    # print(new)
    for i in range(new.shape[1]):
        for j in range(new.shape[2]):
            ans = images[:, i:rows_k + i, j:cols_k + j] * kernel
            # print(ans.shape)
            # print(ans.T.shape)
            # print(np.sum(ans, axis=2).shape)
            mat = np.sum(ans, axis=(1, 2))
            new[:, i, j] = mat
    return new
