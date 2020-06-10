#!/usr/bin/env python3
"""
This script has the method conv_backward(dZ,
A_prev, W, b, padding="same", stride=(1, 1)):
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the unactivated output of the convolutional layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
    kh is the filter height
    kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    padding is a string that is either same or valid, indicating the type of padding used
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
    output_h = int(np.ceil((h_prev - kh + 1) / sh))
    output_w = int(np.ceil((w_prev - kw + 1) / sw))
    print(dZ.shape)
    print(A_prev.shape)
    print(W.shape)
    if padding == "same":
        output_w = h_prev
        output_h = w_prev
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
        # print(ph, pw)
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)
    print(m, output_h, output_w, c_prev)
    da = np.zeros((m, output_h, output_w, c_prev))
    dw = np.zeros(W.shape)
    db = np.zeros(b.shape)
    print(da.shape)
    #print(new)
    new_r = 0
    for i in range(0, kh * sh, sh):
        new_c = 0
        for j in range(0, kw * sw, sw):
            for k in range(c_new):
                a = np.multiply(A_prev[:, i:h_new + i, j:w_new + j],
                                dZ)
                # print(a.shape)
                # print(ans.T.shape)
                # print(np.sum(ans, axis=2).shape)
                # k en dw y dz para que se multiplique por todos los A
                mat = np.sum(a, axis=(1, 2))
                # print(mat.shape)
                dw[new_r, new_c] = np.sum(mat, axis=0)
                db[0, 0, 0] = np.sum(dZ, axis=(0, 1, 2))
            new_c = new_c + 1
        new_r = new_r + 1
    dZ = np.pad(dZ, ((0, 0), (1, 1), (1, 1), (0, 0)),
                    'constant', constant_values=0)
    new_r = 0
    for i in range(0, output_h * sh, sh):
        new_c = 0
        for j in range(0, output_w * sw, sw):
            for k in range(c_prev):
                print("esto es antes de multiplicar")
                a = np.multiply(dZ[:, i:kh + i, j:kw + j],
                                W[:, :, :, k])
                print(a.shape)
                # print(ans.T.shape)
                # print(np.sum(ans, axis=2).shape)
                # k en dw y dz para que se multiplique por todos los A
                mat = np.sum(a, axis=(1, 2, 3))
                print(mat.shape, "esto es despues de sumar", np.sum(mat))
                da[new_r, new_c, k] = np.sum(mat)
            new_c = new_c + 1
        new_r = new_r + 1
    print(da)
    print(dw)
    print(db)
    return
