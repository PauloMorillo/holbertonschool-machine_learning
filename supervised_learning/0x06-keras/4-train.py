#!/usr/bin/env python3
""" This module has the method train_model
    the prototype is
    def train_model(network, data, labels, batch_size,
    epochs, verbose=True, shuffle=False)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ This method train a model using mini-batch gradient descent"""
    return network.fit(data, labels, epochs=epochs,
                batch_size=batch_size,
                verbose=verbose, shuffle=shuffle)
