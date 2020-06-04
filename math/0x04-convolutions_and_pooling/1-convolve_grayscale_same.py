#!/usr/bin/env python3
""" This module has the method
convolve_grayscale_same(images, kernel)
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """This method performs a same
    convolution on grayscale images
    """
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output_h = (w - kw) + 1
    # pw = max((output_w - cols_k) * 1 + (cols_k - cols_im), 0)
    # ph = max((rows_im - rows_k) * 1 + (rows_k - rows_im), 0)
    if h % 2 == 0:
        ph = h // 2
    else:
        ph = (h - 1) // 2
    if w % 2 == 0:
        pw = w // 2
    else:
        pw = (w - 1) // 2
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)

    new = np.zeros((images.shape[0], h, w))
    # print(new.shape)
    # print(new)
    for i in range(new.shape[1]):
        for j in range(new.shape[2]):
            ans = images[:, i:kh + i, j:kw + j] * kernel
            # print(ans.shape)
            # print(ans.T.shape)
            # print(np.sum(ans, axis=2).shape)
            mat = np.sum(ans, axis=(1, 2))
            new[:, i, j] = mat
    return new
