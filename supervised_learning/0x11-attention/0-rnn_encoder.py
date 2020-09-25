#!/usr/bin/env python3
"""
This module has the class RNNEncoder
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    This class encode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        all begins
        vocab is an integer representing the size of the input vocabulary
        embedding is an integer representing the dimensionality of the embedding vector
        units is an integer representing the number of hidden units in the RNN cell
        batch is an integer representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(self.units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        This method initialize the hidden state
        """
        initializer_z = tf.keras.initializers.Zeros()
        return initializer_z(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        This method call the GRU
        """
        outputs = self.embedding(x)
        outputs, hidden = self.gru(outputs, initial_state=initial)
        return outputs, hidden
