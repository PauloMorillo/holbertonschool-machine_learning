#!/usr/bin/env python3
""" This module has a method accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ This method calculates accuracy for the NN """
    correc = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accu = tf.reduce_mean(tf.cast(correc, tf.float32))
    return accu
