#!/usr/bin/env python3
"""
This script has the method
densenet121(growth_rate=32, compression=1.0)
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture
    growth_rate is the growth rate
    compression is the compression factor
    """
    kernel = "he_normal"
    X = K.Input(shape=(224, 224, 3))
    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation("relu")(norm1)
    con1 = K.layers.Conv2D(64, kernel_size=7, strides=2,
                           padding="same", kernel_initializer=kernel)(act1)
    pool1 = K.layers.MaxPool2D((3, 3), 2, padding="same")(con1)

    Y1, nb = dense_block(pool1, 64, growth_rate, 6)
    nb = int(nb)
    Y2, nb = transition_layer(Y1, nb, compression)

    Y3, nb = dense_block(Y2, nb, growth_rate, 12)
    nb = int(nb)
    Y4, nb = transition_layer(Y3, nb, compression)

    Y5, nb = dense_block(Y4, nb, growth_rate, 24)
    nb = int(nb)
    Y6, nb = transition_layer(Y5, nb, compression)

    Y7, nb = dense_block(Y6, nb, growth_rate, 16)
    Y8 = K.layers.AveragePooling2D(7)(Y7)

    last_lay = K.layers.Dense(1000, activation='softmax')(Y8)
    model = K.models.Model(inputs=X, outputs=last_lay)
    return model
