#!/usr/bin/env python3
"""
This script has the method
identity_block(A_prev, filters)
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    """
    F11, F3, F12 = filters
    kernel = "he_normal"
    con1 = K.layers.Conv2D(F11, (1, 1), padding="same", activation='relu',
                           kernel_initializer=kernel)(A_prev)
    norm1 = K.layers.BatchNormalization(1)(con1)
    act1 = K.layers.Activation(K.layers.ReLU())(norm1)
    con2 = K.layers.Conv2D(F3, (3, 3), activation='relu', padding="same",
                           kernel_initializer=kernel)(act1)
    norm2 = K.layers.BatchNormalization(1)(con2)
    act2 = K.layers.Activation(K.layers.ReLU())(norm2)
    con3 = K.layers.Conv2D(F12, (1, 1), padding="same", activation='relu',
                           kernel_initializer=kernel)(act2)
    norm3 = K.layers.BatchNormalization(1)(con3)
    add = K.layers.Add()([norm3, A_prev])
    act3 = K.layers.Activation(K.layers.ReLU())
    return act3(add)
