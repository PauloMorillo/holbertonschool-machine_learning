#!/usr/bin/env python3
""" This module has the method build_model
    the prototype is
    def build_model(nx, layers, activations, lambtha, keep_prob):
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """This method build a model using keras"""
    model = K.Sequential()
    count = 0
    for nodes, activation in zip(layers, activations):
        if count == 0:
            model.add(K.layers.Dense(nodes, input_shape=(nx,),
                                         activation=activation,
                                         kernel_regularizer=K.regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dense(nodes, activation=activation,
                                         kernel_regularizer=K.regularizers.l2(lambtha)))

        if count < len(layers) - 1:
            model.add(K.layers.Dropout(keep_prob))
        count = count + 1
    return model
