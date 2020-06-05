#!/usr/bin/env python3
""" This module has the method train_model
    the prototype is
    train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """ This method train a model using mini-batch gradient descent"""
    def decay(epoch):
        """ This method create the alpha"""
        return alpha / (1 + decay_rate * epoch)
    callback = []
    if type(validation_data) is not tuple:
        validation_data = None
    if early_stopping is True and validation_data is not None:
        callback += [K.callbacks.EarlyStopping(patience=patience)]
    if learning_rate_decay is True:
        callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
    if filepath is not None:
        callback += [K.callbacks.ModelCheckpoint(filepath,
                                                 save_best_only=True,
                                                 mode='min')]
    if len(callback) < 1:
        callback = None
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callback
                       )
