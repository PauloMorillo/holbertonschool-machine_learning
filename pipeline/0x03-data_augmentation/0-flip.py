#!/usr/bin/env python3
"""
This script has the flip_image(image) method
"""
import tensorflow as tf


def flip_image(image):
    """
    flips an image horizontally:
    image is a 3D tf.Tensor containing the image to flip
    Returns the flipped image
    """
    return tf.image.flip_left_right(image)
