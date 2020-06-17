#!/usr/bin/env python3
"""
This script has the method
projection_block(A_prev, filters, s=2)
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution as well
        as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main
     path and the shortcut connection
    """
    F11, F3, F12 = filters
    kernel = "he_normal"
    con1 = K.layers.Conv2D(F11, (1, 1), strides=s, padding="same",
                           kernel_initializer=kernel)(A_prev)
    norm1 = K.layers.BatchNormalization()(con1)
    act1 = K.layers.Activation(K.layers.ReLU())(norm1)
    con2 = K.layers.Conv2D(F3, (3, 3), padding="same",
                           kernel_initializer=kernel)(act1)
    norm2 = K.layers.BatchNormalization()(con2)
    act2 = K.layers.Activation(K.layers.ReLU())(norm2)
    con3 = K.layers.Conv2D(F12, (1, 1), padding="same",
                           kernel_initializer=kernel)(act2)
    con4 = K.layers.Conv2D(F12, (1, 1), s, padding="same",
                           kernel_initializer=kernel)(A_prev)
    norm3 = K.layers.BatchNormalization()(con3)
    norm4 = K.layers.BatchNormalization()(con4)
    add = K.layers.Add()([norm3, norm4])
    act3 = K.layers.Activation(K.layers.ReLU())
    return act3(add)
