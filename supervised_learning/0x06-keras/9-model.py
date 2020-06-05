#!/usr/bin/env python3
""" This module has the method train_model
    the prototype is
    save_model(network, filename)
    def load_model(filename)
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ This method save a model in a file"""
    network.save(filename)


def load_model(filename):
    """ This method load a model in a file"""
    return K.models.load_model(filename)
