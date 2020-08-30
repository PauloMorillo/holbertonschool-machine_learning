#!/usr/bin/env python3
"""
This module has the method
autoencoder(input_dims, filters, latent_dims)
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    This method creates a convolutional autoencoder
    input_dims is a tuple of integers containing the dimensions of
    the model input
    filters is a list containing the number of filters for each
    convolutional layer in the encoder, respectively
    the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of
    the latent space representation
    Each convolution in the encoder should use a kernel size of (3, 3)
    with same padding and relu activation, followed by max pooling
    of size (2, 2)
    Each convolution in the decoder, except for the last two, should
    use a filter
    size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2)
    The second to last convolution should instead use valid padding
    The last convolution should have the same number of filters as
    the number of channels in input_dims with sigmoid activation and
    no upsampling
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    """
    input_encoder = keras.layers.Input(shape=input_dims)
    input_encoded = input_encoder

    for i, n in enumerate(filters):
        encoded = keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same')(input_encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
        input_encoded = encoded

    encoder = keras.models.Model(input_encoder, encoded)

    input_decoder = keras.layers.Input(shape=latent_dims)
    input_decoded = input_decoder

    for i, n in enumerate(filters[::-1]):
        if i == len(filters) - 1:
            decoded = keras.layers.Conv2D(n, (3, 3), activation='sigmoid', padding='valid')(input_decoded)
        else:
            decoded = keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same')(input_decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
        input_decoded = decoded

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(decoded)

    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=input_dims)
    encoderOut = encoder(input_auto)
    decoderOut = decoder(encoderOut)
    auto = keras.models.Model(input_auto, decoderOut)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
