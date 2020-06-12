#!/usr/bin/env python3
"""
This script has the method pool_backward(dA, A_prev,
kernel_shape, stride=(1, 1), mode='max'):
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
    the output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
    kh is the kernel height
    kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    m, h_new, w_new, c_new = dA.shape
    sh, sw = stride
    da_prev = np.zeros(A_prev.shape)
    for im in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    f = np.ones((kh, kw))
                    if mode == "max":
                        f = f * dA[im, h, w, c]
                    if mode == "avg":
                        f = f * (dA[im, h, w, c] / (kh * kw))
                    da_prev[im, h * sh:h * sh + kh, w * sw:w * sw + kw, c] += f
    return da_prev
