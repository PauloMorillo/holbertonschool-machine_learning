#!/usr/bin/env python3
"""
This script has the method
projection_block(A_prev, filters, s=2)
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    """
    kernel = "he_normal"
    for i in range(layers):
        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation("relu")(norm1)
        con1 = K.layers.Conv2D((4 * growth_rate), (1, 1),
                               padding="same",
                               kernel_initializer=kernel)(act1)
        norm2 = K.layers.BatchNormalization()(con1)
        act2 = K.layers.Activation("relu")(norm2)
        con2 = K.layers.Conv2D(growth_rate, (3, 3),
                               padding="same",
                               kernel_initializer=kernel)(act2)
        X = K.layers.concatenate([X, con2])
    return X, X.shape[-1]
