#!/usr/bin/env python3
"""
This script has the method conv_forward(A_prev, W,
b, activation, padding="same", stride=(1, 1)):
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels
    for the convolution
    kh is the filter height
    kw is the filter width
    c_prev is the number of channels in the previous layer
    c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid,
    indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    """
    m = A_prev[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]
    sh = stride[0]
    sw = stride[1]
    output_h = int(np.ceil((h_prev - kh + 1) / sh))
    output_w = int(np.ceil((w_prev - kw + 1) / sw))
    if padding == "same":
        print("hola")
        output_w = h_prev
        output_h = w_prev
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
        # print(ph, pw)
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)

    new = np.ones((A_prev.shape[0], output_h, output_w, c_new))
    # print(new.shape)
    # print(new)
    new_r = 0
    for i in range(0, output_h * sh, sh):
        new_c = 0
        for j in range(0, output_w * sw, sw):
            for k in range(c_new):
                ans = A_prev[:, i:kh + i, j:kw + j, :] * \
                      W[:, :, :, k]
                # print(ans.shape)
                # print(ans.T.shape)
                # print(np.sum(ans, axis=2).shape)
                mat = np.sum(ans, axis=(1, 2, 3))
                new[:, new_r, new_c, k] = mat + b[:, :, :, k]
            new_c = new_c + 1
        new_r = new_r + 1
    return activation(new)
