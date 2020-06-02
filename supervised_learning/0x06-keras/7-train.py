#!/usr/bin/env python3
""" This module has the method train_model
    the prototype is
    train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """ This method train a model using mini-batch gradient descent"""
    callback = None
    if validation_data is not None:
        if early_stopping is True:
            callback = K.callbacks.EarlyStopping(patience=patience)
        if learning_rate_decay is True:
            learning_rate_fn = K.optimizers.schedules.InverseTimeDecay(
                alpha, epochs, decay_rate)
    network.fit(data, labels, epochs=epochs,
                batch_size=batch_size,
                verbose=verbose, shuffle=shuffle,
                validation_data=validation_data,
                callbacks=[callback]
                )
