#!/usr/bin/env python3
"""
This module has the class SelfAttention and RNNDecoder
"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    This class decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        all begins here
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True
                                       )
        self.F = tf.keras.layers.Dense(vocab)
        self.units = units

    def call(self, x, s_prev, hidden_states):
        """
        This method has the model to call
        """
        # enc_output shape == (batch_size, max_length, hidden_size)
        attention = SelfAttention(self.units)
        context_vector, attention_weights = attention(s_prev, hidden_states)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.F(output)

        return x, state
