#!/usr/bin/env python3
"""
This script has the method pool_forward(A_prev,
kernel_shape, stride=(1, 1), mode='max'):
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw)
    containing the size of the kernel for the pooling
    kh is the kernel height
    kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width
    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively
    """
    m = A_prev[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    new_rows = h_prev - kh
    new_cols = w_prev - kw
    new = np.ones((A_prev.shape[0], (new_rows // sh) + 1,
                   (new_cols // sw) + 1, c_prev))
    new_r = 0

    for i in range(0, new_rows + 1, sh):
        new_c = 0
        for j in range(0, new_cols + 1, sw):
            if mode == 'avg':
                ans = A_prev[:, i:kh + i, j:kw + j]
                mat = np.sum(ans, axis=(1, 2))
                ans_f = mat / (kernel_shape[0] * kernel_shape[1])
                new[:, new_r, new_c] = ans_f
            if mode == 'max':
                ans = np.max(A_prev[:, i:kh + i, j:kw + j, :],
                             axis=(1, 2))
                new[:, new_r, new_c] = ans
            new_c = new_c + 1
        new_r = new_r + 1
    return new
