#!/usr/bin/env python3
"""
This script has the method
projection_block(A_prev, filters, s=2)
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    kernel = "he_normal"
    for i in range(layers):
        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation(K.layers.ReLU())(norm1)
        con1 = K.layers.Conv2D(2 * nb_filters, (1, 1), activation='relu',
                               padding="same",
                               kernel_initializer=kernel)(act1)
        norm2 = K.layers.BatchNormalization()(con1)
        act2 = K.layers.Activation(K.layers.ReLU())(norm2)
        con2 = K.layers.Conv2D(growth_rate, (3, 3), activation='relu',
                               padding="same",
                               kernel_initializer=kernel)(act2)
        X = K.layers.concatenate([X, con2])
    return X, nb_filters + (layers * growth_rate)
