#!/usr/bin/env python3
""" This module has the create_Adam_op method"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network
    """
    operation = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train = operation.minimize(loss)
    return train
