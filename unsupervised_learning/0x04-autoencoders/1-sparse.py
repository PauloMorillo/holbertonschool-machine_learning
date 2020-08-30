#!/usr/bin/env python3
"""
This module has the method
autoencoder(input_dims, hidden_layers, latent_dims, lambtha)
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    This method creates a sparse autoencoder
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    lambtha is the regularization parameter used for L1 regularization on
    the encoded output
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the sparse autoencoder model
    """
    input_encoder = keras.layers.Input(shape=(input_dims,))
    input_encoded = input_encoder

    l_1 = keras.regularizers.l1(lambtha)
    for n in hidden_layers:
        encoded = keras.layers.Dense(n, activation='relu',
                                     activity_regularizer=l_1)(input_encoded)
        input_encoded = encoded
    latent = keras.layers.Dense(latent_dims,
                                activation='relu',
                                activity_regularizer=l_1)(encoded)
    encoder = keras.models.Model(input_encoder, latent)

    input_decoder = keras.layers.Input(shape=(latent_dims,))
    input_decoded = input_decoder
    for i, n in enumerate(hidden_layers[::-1]):
        activation = 'relu'
        decoded = keras.layers.Dense(n, activation=activation)(input_decoded)
        input_decoded = decoded
    decoded = keras.layers.Dense(input_dims,
                                 activation='sigmoid')(input_decoded)
    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=(input_dims,))
    encoderOut = encoder(input_auto)
    decoderOut = decoder(encoderOut)
    auto = keras.models.Model(input_auto, decoderOut)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
