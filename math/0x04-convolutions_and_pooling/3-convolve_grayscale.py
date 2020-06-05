#!/usr/bin/env python3
""" This module has the method
convolve_grayscale(images, kernel, padding='same', stride=(1, 1))
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    This method performs a convolution
     on grayscale images
    """
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output_h = int(np.ceil((h - kh + 1) / stride[0]))
    output_w = int(np.ceil((w - kw + 1) / stride[1]))

    if type(padding) is tuple:
        output_w = ((w - kw + (2 * padding[1])) // stride[1]) + 1
        output_h = ((h - kh + (2 * padding[0])) // stride[0]) + 1
        images = np.pad(images, ((0, 0), (padding[0], padding[0]),
                                 (padding[1], padding[1])),
                        'constant', constant_values=0)
    else:
        if padding == "same":
            output_h = int(np.ceil(h / stride[0]))
            output_w = int(np.ceil(w / stride[1]))
            if kh % 2 == 0:
                pt = kh // 2
                pb = kh // 2
            else:
                ph = max((output_h - 1) * stride[0] + kh - h, 0)
                pt = ph // 2
                pb = ph - pt
            if kw % 2 == 0:
                pt = kh // 2
                pb = kh // 2
            else:
                pw = max((output_w - 1) * stride[1] + kw - w, 0)
                pl = pw // 2
                pr = pw - pl

            images = np.pad(images, ((0, 0), (pt, pb), (pl, pr)),
                            'constant', constant_values=0)

    new = np.zeros((images.shape[0], output_h, output_w))
    output_h = h - kh + 1
    output_w = w - kw + 1
    new_r = 0

    for i in range(0, output_h, stride[0]):
        new_c = 0
        for j in range(0, output_w, stride[1]):
            ans = images[:, i:kh + i, j:kw + j] * kernel
            # print(ans.shape)
            # print(ans.T.shape)
            # print(np.sum(ans, axis=2).shape)
            mat = np.sum(np.sum(ans.T, axis=1), axis=0)
            new[:, new_r, new_c] = mat
            new_c = new_c + 1
        new_r = new_r + 1
    return new
