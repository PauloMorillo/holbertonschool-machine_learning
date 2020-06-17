#!/usr/bin/env python3
"""
This script has the method
inception_network()
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    This method builds the inception network googleNET
    """
    X = K.Input(shape=(224, 224, 3))

    con1 = K.layers.Conv2D(64, kernel_size=7, strides=2, padding="same",
                           activation='relu')(X)
    pool1 = K.layers.MaxPool2D((3, 3), 2, padding="same")(con1)

    con2 = K.layers.Conv2D(64, kernel_size=1, strides=1, activation='relu',
                           padding="same")(pool1)
    con3 = K.layers.Conv2D(192, 3, 1, activation='relu',
                           padding="same")(con2)
    pool2 = K.layers.MaxPool2D((3, 3), 2, padding="same")(con3)

    Y1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    Y2 = inception_block(Y1, [128, 128, 192, 32, 96, 64])
    pool2 = K.layers.MaxPool2D((3, 3), 2, padding="same")(Y2)

    Y3 = inception_block(pool2, [192, 96, 208, 16, 48, 64])
    Y4 = inception_block(Y3, [160, 112, 224, 24, 64, 64])
    Y5 = inception_block(Y4, [128, 128, 256, 24, 64, 64])
    Y6 = inception_block(Y5, [112, 144, 288, 32, 64, 64])
    Y7 = inception_block(Y6, [256, 160, 320, 32, 128, 128])
    pool3 = K.layers.MaxPool2D((3, 3), 2, padding="same")(Y7)
    Y8 = inception_block(pool3, [256, 160, 320, 32, 128, 128])
    Y9 = inception_block(Y8, [384, 192, 384, 48, 128, 128])
    pool3 = K.layers.AveragePooling2D((7, 7), 1)(Y9)
    d1 = K.layers.Dropout(0.6)(pool3)
    # last_b = K.layers.Dense(10, activation='relu')(d1)
    last_lay = K.layers.Dense(1000, activation='softmax')(d1)

    model = K.models.Model(inputs=X, outputs=last_lay)
    return model
