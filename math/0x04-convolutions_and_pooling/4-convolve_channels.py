#!/usr/bin/env python3
""" This module has the method
convolve_channels(images, kernel, padding='same', stride=(1, 1))
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    This method performs a convolution on images with channels
    """
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    output_h = int(np.ceil((h - kh + 1) / sh))
    output_w = int(np.ceil((w - kw + 1) / sw))

    if type(padding) is tuple:
        print("hola")
        output_w = ((w - kw + (2 * padding[1])) // sw) + 1
        output_h = ((h - kh + (2 * padding[0])) // sh) + 1
        images = np.pad(images, ((0, 0), (padding[0], padding[0]),
                                 (padding[1], padding[1])),
                        'constant', constant_values=0)
    else:
        if padding == "same":
            print("hola")
            output_h = int(np.ceil(h / sh))
            output_w = int(np.ceil(w / sw))
            output_w = w
            output_h = h
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
            ph = int((((h - 1) * sh + kh - h) / 2) + 1)
            pw = int((((w - 1) * sw + kw - w) / 2) + 1)
            images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                            'constant', constant_values=0)

    new = np.ones((images.shape[0], output_h,
                   output_w))
    # print(new.shape)
    # print(new)
    new_r = 0

    for i in range(0, output_h * sh, sh):
        new_c = 0
        for j in range(0, output_w * sw, sw):
            ans = images[:, i:kh + i, j:kw + j, :] * kernel
            # print(ans.shape)
            # print(ans.T.shape)
            # print(np.sum(ans, axis=2).shape)
            mat = np.sum(ans, axis=(1, 3))
            # print(mat.shape)
            new[:, new_r, new_c] = np.sum(mat, axis=1)
            new_c = new_c + 1
        new_r = new_r + 1
    return new
