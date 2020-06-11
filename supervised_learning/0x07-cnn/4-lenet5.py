#!/usr/bin/env python3
"""
This script has the method lenet5(x, y):
"""
import tensorflow as tf


def lenet5(x, y):
    """
    x is the images and y are the labels
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    conv1 = tf.layers.Conv2D(6, (5, 5), padding="same", activation='relu',
                             kernel_initializer=kernel)
    c = conv1(x)
    pool1 = tf.layers.MaxPooling2D((2, 2), (2, 2))
    p = pool1(c)
    conv2 = tf.layers.Conv2D(16, (5, 5), padding="valid", activation='relu',
                             kernel_initializer=kernel)
    c2 = conv2(p)
    pool2 = tf.layers.MaxPooling2D((2, 2), (2, 2))
    p2 = pool2(c2)
    out = tf.layers.Flatten()(p2)
    lay1 = tf.layers.Dense(120, activation='relu', kernel_initializer=kernel)
    y1 = lay1(out)
    lay2 = tf.layers.Dense(84, activation='relu', kernel_initializer=kernel)
    y2 = lay2(y1)
    lay3 = tf.layers.Dense(10, activation='softmax', kernel_initializer=kernel)
    y_pred = lay3(y2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    correc = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accu = tf.reduce_mean(tf.cast(correc, tf.float32))
    operation = tf.train.AdamOptimizer()
    train = operation.minimize(loss)
    return y_pred, train, loss, accu
