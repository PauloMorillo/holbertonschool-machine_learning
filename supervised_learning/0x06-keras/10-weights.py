#!/usr/bin/env python3
""" This module has the method train_model
    the prototype is
    save_model(network, filename)
    def load_model(filename)
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ This method save a model weights in a file"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ This method load a model weights in a file"""
    return network.load_weights(filename)
