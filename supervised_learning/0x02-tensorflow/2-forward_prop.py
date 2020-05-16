#!/usr/bin/env python3
""" This module has a method forwward_prop """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ This method create a forwdward prop for the NN """
    for node, activation in zip(layer_sizes, activations):
        y = create_layer(x, node, activation)
        x = y
    return x
