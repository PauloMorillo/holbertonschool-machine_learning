#!/usr/bin/env python3
"""
This module has the class DecoderBlock
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        # self.positional_encoding =
        # np.expand_dims(positional_encoding(max_seq_len, self.dm), axis=0)
        # self.positional_encoding =
        # tf.cast(self.positional_encoding, dtype=tf.float32)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = x.shape[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # print(self.positional_encoding.shape)
        # print(seq_len, "uqueruur", self.positional_encoding.shape)
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
