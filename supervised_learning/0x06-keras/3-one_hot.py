#!/usr/bin/env python3
""" This module has the method one_hot
    the prototype is
    one_hot(labels, classes=None)
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ This method converts a label vector into a one-hot matrix"""
    return K.utils.to_categorical(labels, classes)
