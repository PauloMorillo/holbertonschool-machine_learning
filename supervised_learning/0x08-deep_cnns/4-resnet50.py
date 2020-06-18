#!/usr/bin/env python3
"""
This script has the method
resnet50()
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture
    """
    kernel = "he_normal"
    X = K.Input(shape=(224, 224, 3))
    con1 = K.layers.Conv2D(64, kernel_size=7, strides=2,
                           padding="same", kernel_initializer=kernel)(X)
    norm1 = K.layers.BatchNormalization()(con1)
    act1 = K.layers.Activation("relu")(norm1)
    pool1 = K.layers.MaxPool2D((3, 3), 2, padding="same")(act1)

    Y1 = projection_block(pool1, [64, 64, 256], 1)
    Y2 = identity_block(Y1, [64, 64, 256])
    Y3 = identity_block(Y2, [64, 64, 256])

    Y4 = projection_block(Y3, [128, 128, 512])
    Y5 = identity_block(Y4, [128, 128, 512])
    Y6 = identity_block(Y5, [128, 128, 512])
    Y7 = identity_block(Y6, [128, 128, 512])

    Y8 = projection_block(Y7, [256, 256, 1024])
    Y9 = identity_block(Y8, [256, 256, 1024])
    Y10 = identity_block(Y9, [256, 256, 1024])
    Y11 = identity_block(Y10, [256, 256, 1024])
    Y12 = identity_block(Y11, [256, 256, 1024])
    Y13 = identity_block(Y12, [256, 256, 1024])

    Y14 = projection_block(Y13, [512, 512, 2048])
    Y15 = identity_block(Y14, [512, 512, 2048])
    Y16 = identity_block(Y15, [512, 512, 2048])
    Y17 = K.layers.AveragePooling2D(7)(Y16)
    last_lay = K.layers.Dense(1000, activation='softmax')(Y17)
    model = K.models.Model(inputs=X, outputs=last_lay)
    return model
