#!/usr/bin/env python3
""" This module has a evaluate """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ This method restore a model and evaluate """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)
    x = tf.get_collection('x')
    y = tf.get_collection('y')
    y_pred = tf.get_collection('y_pred')
    loss = tf.get_collection('loss')
    accuracy = tf.get_collection('accuracy')
    train_op = tf.get_collection('train_op')
    sess.run((train_op), feed_dict={x: X, y: Y})
    y_predr, accuar, lossr = sess.run((y_pred, accua, loss),
                                      feed_dict={x: X, y: Y})
    sess.close()
    return y_predr, accuar, lossr
