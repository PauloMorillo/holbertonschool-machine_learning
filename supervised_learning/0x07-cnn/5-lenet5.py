#!/usr/bin/env python3
"""
This script has the method lenet5(x, y):
using keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    X is the tensor input
    """
    kernel = "he_normal"
    con1 = K.layers.Conv2D(6, (5, 5), padding="same", activation='relu',
                           kernel_initializer=kernel,
                           input_shape=X.shape)(X)
    pool1 = K.layers.MaxPool2D((2, 2), (2, 2))(con1)
    con2 = K.layers.Conv2D(16, (5, 5), padding="valid", activation='relu',
                           kernel_initializer=kernel)(pool1)
    pool2 = K.layers.MaxPool2D((2, 2), (2, 2))(con2)
    flat = K.layers.Flatten()(pool2)
    lay1 = K.layers.Dense(120, activation='relu',
                          kernel_initializer=kernel)(flat)
    lay2 = K.layers.Dense(84, activation='relu',
                          kernel_initializer=kernel)(lay1)
    lay3 = K.layers.Dense(10, activation='softmax',
                          kernel_initializer=kernel)(lay2)
    model = K.Model(inputs=X, outputs=lay3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
