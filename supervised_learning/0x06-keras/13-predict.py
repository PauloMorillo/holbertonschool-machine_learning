#!/usr/bin/env python3
""" This module has the methods
    predict(network, data, verbose=False)
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ This method  makes a prediction
    using a neural network"""
    return network.predict(data, verbose=verbose)
