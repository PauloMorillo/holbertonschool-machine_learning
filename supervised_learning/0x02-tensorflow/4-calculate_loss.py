#!/usr/bin/env python3
""" This module has a method calculate_loss """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ This method calculates loss for the NN """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
