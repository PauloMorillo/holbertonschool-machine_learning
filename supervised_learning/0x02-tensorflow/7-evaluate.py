#!/usr/bin/env python3
""" This module has a evaluate """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ This method restore a model and evaluate """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    y_pred = tf.get_collection('y_pred')[0]
    print(y_pred)
    accuracy = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')
    y_predr = sess.run((y_pred), feed_dict={x: X, y: Y})
    accuar = sess.run((accuracy), feed_dict={x: X, y: Y})
    lossr = sess.run((loss), feed_dict={x: X, y: Y})
    sess.close()
    return y_predr, accuar, lossr
