#!/usr/bin/env python3
"""This module has the method dropout_create_layer"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """This method creates a layer of a neural network using dropout"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.layers.Dropout(keep_prob)
    lay = tf.layers.Dense(units=n, activation=activation,
                          kernel_initializer=kernel,
                          kernel_regularizer=kernel_reg,
                          name="layer")
    return lay(prev)
