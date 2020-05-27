#!/usr/bin/env python3
"""This module has the method l2_reg_cost(cost):"""
import tensorflow as tf


def l2_reg_cost(cost):
    """This method calculates the cost of a neural
    network with L2 regularization"""
    return tf.losses.get_regularization_losses(cost)
