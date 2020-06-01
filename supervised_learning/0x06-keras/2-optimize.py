#!/usr/bin/env python3
""" This module has the method optimize_model
    the prototype is
    def optimize_model(network, alpha, beta1, beta2)
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ This method optimize the model"""
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
