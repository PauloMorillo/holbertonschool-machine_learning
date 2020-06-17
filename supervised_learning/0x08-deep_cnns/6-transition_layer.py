#!/usr/bin/env python3
"""
This script has the method
transition_layer(X, nb_filters, compression)
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    """
    kernel = "he_normal"
    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation("relu")(norm1)
    filt = int(nb_filters * compression)
    con1 = K.layers.Conv2D(filt, (1, 1),
                           padding="same",
                           kernel_initializer=kernel)(act1)
    Y = K.layers.AveragePooling2D((2, 2))(con1)
    return Y, Y.shape[-1]
