#!/usr/bin/env python3
"""
This module has the class MultiHeadAttention
"""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    This class performs multi head attention
    """

    def __init__(self, dm, h):
        """
        All begins here
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm

        assert dm % self.h == 0

        self.depth = dm // self.h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size,
        num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        This method call the model
        """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)  # (batch_size, seq_len, d_model)
        k = self.Wk(K)  # (batch_size, seq_len, d_model)
        v = self.Wv(V)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape ==
        # (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape ==
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size,
                                       -1,
                                       self.dm))
        # (batch_size, seq_len_q, d_model)

        output = self.linear(concat_attention)
        # (batch_size, seq_len_q, d_model)

        return output, attention_weights
