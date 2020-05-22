#!/usr/bin/env python3
""" This module has the learning_rate_decay method"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer
    for a neural network a neural network
    """

    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y_pred = tf.layers.dense(prev, units=n,
                             kernel_initializer=kernel)
    mean, var = tf.nn.moments(y_pred, [0], keep_dims=True)
    gamma = tf.Variable(tf.ones([y_pred.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([y_pred.get_shape()[-1]]))
    znorm = tf.nn.batch_normalization(y_pred, mean, var, beta, gamma, 1e-8)
    y_pred = activation(znorm)
    return y_pred
