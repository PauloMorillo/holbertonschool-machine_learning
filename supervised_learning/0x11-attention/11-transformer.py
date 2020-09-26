#!/usr/bin/env python3
"""
This module has the Transformer class
"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """This class create a transformer network"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """
        All starts here
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        This method call the model
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)
        # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
