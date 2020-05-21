#!/usr/bin/env python3
""" This module has the learning_rate_decay method"""
import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer
    for a neural network a neural network
    """
    mean, var = tf.nn.momments(prev)
    gamma = np.ones((1, n))
    beta = np.zeros((1, n))
    znorm = tf.nn.batch_normalization(prev, mean, var, beta, gamma, 1e-8)
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    predfunc = tf.layers.Dense(units=n,
                               activations=activation,
                               kernel_initializer=kernel)
    y_pred = predfunc(znorm)
    return y_pred
