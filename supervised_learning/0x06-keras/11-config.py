#!/usr/bin/env python3
""" This module has the methods
    save_config(network, filename)
    load_config(filename)
"""
import tensorflow.keras as K


def save_config(network, filename):
    """ This method save a model config in a JSON file"""
    model_str = network.to_json()
    with open(filename, 'w') as f:
        f.write(model_str)


def load_config(filename):
    """ This method load a model config in a JSON file"""
    with open(filename, 'r') as f:
        data = f.read()
    return K.models.model_from_json(data)
