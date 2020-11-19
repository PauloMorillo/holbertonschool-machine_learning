#!/usr/bin/env python3
"""
This script has the shear_image(image, intensity) method
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    randomly shears an image:
    image is a 3D tf.Tensor containing the image to shear
    intensity is the intensity with which the image should be sheared
    Returns the sheared image
    """
    array_image = tf.keras.preprocessing.image.img_to_array(image)
    s_i = tf.keras.preprocessing.image.random_shear(
        array_image, intensity, row_axis=0, col_axis=1, channel_axis=2
    )
    return tf.keras.preprocessing.image.array_to_img(array_image)
