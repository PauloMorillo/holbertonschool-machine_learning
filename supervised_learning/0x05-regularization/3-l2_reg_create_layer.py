#!/usr/bin/env python3
"""This module has the method l2_reg_create_layer(cost):"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """This method calculates the cost of a neural
    network with L2 regularization"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.contrib.layers.l2_regularizer(lambtha)
    lay = tf.layers.Dense(units=n, activation=activation,
                          kernel_initializer=kernel,
                          kernel_regularizer=kernel_reg,
                          name="layer")
    return lay(prev)
