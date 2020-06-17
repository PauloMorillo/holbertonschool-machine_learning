#!/usr/bin/env python3
"""
This script has the method
inception_block(A_prev, filters)
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    This method create an inception block
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
     respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution
         before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution
         before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution
         after the max pooling
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    kernel = "he_normal"
    con1 = K.layers.Conv2D(F1, (1, 1), activation='relu',
                           kernel_initializer=kernel)(A_prev)

    con2_1 = K.layers.Conv2D(F3R, (1, 1), activation='relu',
                             kernel_initializer=kernel)(A_prev)
    con2_2 = K.layers.Conv2D(F3, (3, 3), padding="same", activation='relu',
                             kernel_initializer=kernel)(con2_1)

    con3_1 = K.layers.Conv2D(F5R, (1, 1), activation='relu',
                             kernel_initializer=kernel)(A_prev)
    con3_2 = K.layers.Conv2D(F5, (5, 5), padding="same", activation='relu',
                             kernel_initializer=kernel)(con3_1)

    pool4 = K.layers.MaxPool2D((3, 3), (1, 1), padding="same")(A_prev)
    con4_2 = K.layers.Conv2D(FPP, (1, 1), (1, 1), activation='relu',
                             kernel_initializer=kernel)(pool4)

    return K.layers.concatenate([con1, con2_2, con3_2, con4_2])
