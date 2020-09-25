#!/usr/bin/env python3
"""
This module has the class EncoderBlock
"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    This class create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        All begins here
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.ffn = tf.keras.Sequential([
            self.dense_hidden,  # (batch_size, seq_len, dff)
            self.dense_output  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        # (batch_size, input_seq_len, d_model)

        return out2
