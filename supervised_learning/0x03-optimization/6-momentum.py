#!/usr/bin/env python3
""" This module has the create_momentum_op method"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates the training operation for a neural network
    """
    operation = tf.train.MomentumOptimizer(alpha, beta1)
    train = operation.minimize(loss)
    return train
