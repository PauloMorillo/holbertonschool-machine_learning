#!/usr/bin/env python3
"""
This script has the change_brightness(image, max_delta) method
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    This script randomly changes the brightness of an image:
    image is a 3D tf.Tensor containing the image to change
    max_delta is the maximum amount the image should be brightened (or darkened)
    Returns the altered image
    """
    return tf.image.adjust_brightness(image, max_delta)
