#!/usr/bin/env python3
""" This module has the train_mini_batch method"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    This method trains a loaded neural
    network model using mini-batch gradient descent
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(load_path + '.meta')
    saver.restore(sess, load_path)
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
    train_op = tf.get_collection('train_op')[0]
    m = X_train.shape[0]
    endpos = m // batch_size + (m % batch_size > 0)
    # print(X_train[posi:batch_size].shape)
    # print(xnew)
    for ep in range(epochs + 1):
        X_s, Y_s = shuffle_data(X_train, Y_train)
        nb = (X_train.shape[0] / batch_size) + 1
        t_cost, t_accuracy = sess.run([loss, accuracy],
                                      feed_dict={x: X_s, y: Y_s})
        v_cost, v_accuracy = sess.run([loss, accuracy],
                                      feed_dict={x: X_valid, y: Y_valid})
        # xnew = np.array_split(X_train, nb)
        # ynew = np.array_split(Y_train, nb)
        # print(len(xnew))
        print("After {} epochs:".format(ep))
        print("\tTraining Cost: {}".format(t_cost))
        print("\tTraining Accuracy: {}".format(t_accuracy))
        print("\tValidation Cost: {}".format(v_cost))
        print("\tValidation Accuracy: {}".format(v_accuracy))
        if ep < epochs:
            posi = 0
            posf = batch_size
            for i in range(1, (endpos + 1)):
                # print(X_train[posi:posf].shape, i)
                # print(Y_train[posi:posf].shape, i)
                # print(batch, labels)
                # print(batch.shape, labels.shape)
                # return
                # print(batch.shape, labels.shape)
                sess.run(train_op, feed_dict={x: X_s[posi:posf],
                                              y: Y_s[posi:posf]})
                b_cost, b_accuracy = sess.run([loss, accuracy],
                                              feed_dict={x: X_s[posi:posf],
                                                         y: Y_s[posi:posf]})
                posi = posf
                if i < endpos - 1:
                    posf = posf + batch_size
                else:
                    posf = len(X_train - 1)
                if (i) % 100 == 0 and i is not 0:
                    print("\tStep {}:".format(i))
                    print("\t\tCost: {}".format(b_cost))
                    print("\t\tAccuracy: {}".format(b_accuracy))
    path = saver.save(sess, save_path)
    return path
