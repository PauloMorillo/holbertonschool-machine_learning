#!/usr/bin/env python3
"""
This script has the method conv_backward(dZ,
A_prev, W, b, padding="same", stride=(1, 1)):
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the unactivated
    output of the convolutional layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
    kh is the filter height
    kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding is a string that is either same or valid, indicating the
    type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]
    sh = stride[0]
    sw = stride[1]
    # print(dZ.shape)
    # print(A_prev.shape)
    # print(W.shape)
    if padding == "same":
        ph = int(np.ceil((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(np.ceil((w_prev - 1) * sw + kw - w_prev) / 2)
        # print(ph, pw)
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)
    # print(m, output_h, output_w, c_prev)
    da = np.zeros(A_prev.shape)
    dw = np.zeros(W.shape)
    db = np.zeros(b.shape)
    # print(da.shape)
    # print(new)
    for im in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    da[im, h * sh:(h * sh) + kh, w * sw:(w * sw) + kw] += \
                        W[:, :, :, c] * dZ[im, h, w, c]
                    dw[:, :, :, c] += \
                        A_prev[im, h * sh:(h * sh) + kh, w * sw:(w * sw) + kw]\
                        * dZ[im, h, w, c]
    db[0, 0, 0] = np.sum(dZ, axis=(0, 1, 2))
    if padding == "same":
        da = da[:, ph:-ph, pw:-pw, :]
    return da, dw, db
