#!/usr/bin/env python3
""" This module has a method create layer """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ This method create a layer for the NN """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(units=n, activation=activation,
                          kernel_initializer=kernel, name="layer")
    return lay(prev)
