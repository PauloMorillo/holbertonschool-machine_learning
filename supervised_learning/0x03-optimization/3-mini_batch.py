#!/usr/bin/env python3
""" This module has the train_mini_batch method"""
import numpy as np


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    This method trains a loaded neural network model using mini-batch gradient descent
    """

    np.random.seed(0)
    Y = np.random.permutation(Y)
    return X, Y
