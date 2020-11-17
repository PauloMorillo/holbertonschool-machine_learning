#!/usr/bin/env python3
"""
This script has the rotate_image(image) method
"""
import tensorflow as tf


def rotate_image(image):
    """
    This method rotates an image by 90 degrees counter-clockwise:
    image is a 3D tf.Tensor containing the image to rotate
    Returns the rotated image    """
    return tf.image.rot90(image=image, k=1, name=None)
