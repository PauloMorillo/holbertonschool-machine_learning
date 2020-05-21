#!/usr/bin/env python3
""" This module has the create_RMSProp_op method"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    """
    operation = tf.train.RMSPropOptimizer(alpha, momentum=beta2, epsilon=epsilon)
    train = operation.minimize(loss)
    return train
