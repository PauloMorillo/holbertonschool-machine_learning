#!/usr/bin/env python3
""" This module has a method train """
import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ This method train the NN and save the model """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
