#!/usr/bin/env python3
""" This module has the create_momentum_op method"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    """
    operation = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    train = operation.minimize(loss)
    return train
