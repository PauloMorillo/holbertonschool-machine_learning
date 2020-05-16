#!/usr/bin/env python3
""" This module has a method train """
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ This method train the NN and save the model """
    x, y = create_placeholders(784, 10)
    y_pred = forward_prop(x, layer_sizes, activations)
    accua = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(iterations):
        trainr, lossr = sess.run((train_op, loss), feed_dict={x: X_train, y: Y_train})
        validr, lossvar = sess.run((train_op, loss), feed_dict={x: X_valid, y: Y_valid})
        if i % 100:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(lossr))      
    return "/dsfdsf"
