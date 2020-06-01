#!/usr/bin/env python3
""" This module has the method build_model
    the prototype is
    def build_model(nx, layers, activations, lambtha, keep_prob):
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """This method build a model using keras"""
    count = 0
    for nodes, activation in zip(layers, activations):
        if count == 0:
            input = K.Input(shape=(nx,))
            y = K.layers.Dense(nodes, activation=activation,
                               kernel_regularizer=K.regularizers.l2(lambtha))(input)
        else:
            y = K.layers.Dense(nodes, activation=activation,
                               kernel_regularizer=K.regularizers.l2(lambtha))(x)

        if count < len(layers) - 1:
            x = K.layers.Dropout(keep_prob)(y)
        count = count + 1
    return K.Model(inputs=input, outputs=y)
