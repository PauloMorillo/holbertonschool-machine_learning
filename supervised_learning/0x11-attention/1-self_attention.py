#!/usr/bin/env python3
"""
This module has the class SelfAttention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    This class calculates the attention for machine translation
    based on https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, units):
        """
        All begins here
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        This method call SelfAttention
        """
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time
        # axis to calculate the score
        query = s_prev
        values = hidden_states
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is
        # (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W(query_with_time_axis) + self.U(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
