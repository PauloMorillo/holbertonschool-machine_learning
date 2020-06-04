#!/usr/bin/env python3
""" This module has the method
convolve_channels(images, kernel, padding='same', stride=(1, 1))
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    This method performs a convolution on images with channels
    """
    rows_im = images.shape[1]
    cols_im = images.shape[2]
    # print(kernels)
    # print(kernels[:,:,:,0])
    rows_k = kernel_shape[0]
    cols_k = kernel_shape[1]
    new_rows = (rows_im - rows_k)
    new_cols = (cols_im - cols_k)
    dimension = (images.shape[3])
    # print(new_cols, new_rows)
    new = np.ones((images.shape[0], (new_rows // stride[0]) + 1,
                   (new_cols // stride[1]) + 1, dimension))
    # print(new.shape)
    # print(new)
    new_r = 0

    for i in range(0, new_rows + 1, stride[0]):
        new_c = 0
        for j in range(0, new_cols + 1, stride[1]):
            if mode == 'avg':
                ans = images[:, i:rows_k + i, j:cols_k + j]
                # print(ans.shape)
                # print(ans.T.shape)
                # print(np.sum(ans, axis=2).shape)
                # print(ans.shape, "esto es shape")
                # print(ans[0])
                mat = np.sum(ans, axis=(1, 2))
                # print(mat[0])
                # print(mat.shape)
                ans_f = mat / (kernel_shape[0] * kernel_shape[1])
                # print(ans_f[0])
                new[:, new_r, new_c] = ans_f
            if mode == 'max':
                ans = np.max(images[:, i:rows_k + i, j:cols_k + j, :])
                new[:, new_r, new_c] = ans
            new_c = new_c + 1
        new_r = new_r + 1
    return new
