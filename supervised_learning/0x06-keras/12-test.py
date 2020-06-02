#!/usr/bin/env python3
""" This module has the methods
    test_model(network, data, labels, verbose=True)
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ This method test a NN"""
    return network.evaluate(data, labels, verbose=verbose)
