#!/usr/bin/env python3
""" This module has a method accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ This method calculates accuracy for the NN """
    accu = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32), 0)
    return accu
