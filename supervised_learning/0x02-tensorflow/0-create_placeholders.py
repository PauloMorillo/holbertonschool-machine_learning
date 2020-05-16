#!/usr/bin/env python3
""" This module has a method create_placeholder """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ This method create a placeholders for the NN """
    x = tf.placeholder(tf.float32, [None, nx], name="x")
    y = tf.placeholder(tf.float32, [None, classes], name="y")
    return x, y
