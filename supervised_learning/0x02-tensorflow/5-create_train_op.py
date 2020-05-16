#!/usr/bin/env python3
""" This module has a method create_train_op """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ This method train the NN """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
