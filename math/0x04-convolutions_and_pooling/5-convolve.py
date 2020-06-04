#!/usr/bin/env python3
""" This module has the method
convolve_channels(images, kernel, padding='same', stride=(1, 1))
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    This method performs a convolution on images with channels
    """
    if type(padding) is tuple:
        images = np.pad(images, ((0, 0), (padding[0], padding[0]),
                                 (padding[1], padding[1])),
                        'constant', constant_values=0)
    else:
        if padding == "same":
            images = np.pad(images, ((0, 0), (1, 1), (1, 1), (0, 0)),
                            'constant', constant_values=0)

    rows_im = images.shape[1]
    cols_im = images.shape[2]
    # print(kernels)
    # print(kernels[:,:,:,0])
    rows_k = kernels.shape[1]
    cols_k = kernels.shape[2]
    new_rows = (rows_im - rows_k + 1)
    new_cols = (cols_im - cols_k + 1)
    dimension = (images.shape[3])
    # print(new_cols, new_rows)
    new = np.ones((images.shape[0], new_rows // stride[0],
                   new_cols // stride[1], images.shape[3]))
    # print(new.shape)
    # print(new)
    new_r = 0

    for i in range(0, new_rows, stride[0]):
        new_c = 0
        for j in range(0, new_cols, stride[1]):
            for k in range(len(kernels)):
                ans = images[:, i:rows_k + i, j:cols_k + j, :] * \
                      kernels[:, :, :, k]
                # print(ans.shape)
                # print(ans.T.shape)
                # print(np.sum(ans, axis=2).shape)
                mat = np.sum(np.sum(ans, axis=1), axis=2)
                # print(mat.shape)
                new[:, new_r, new_c, k] = np.sum(mat, axis=1)
            new_c = new_c + 1
        new_r = new_r + 1
    return new
