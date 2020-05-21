#!/usr/bin/env python3
""" This module has the learning_rate_decay method"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates the training operation for a neural network
    """
    lam = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate)
    return lam
