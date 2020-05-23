#!/usr/bin/env python3
""" This module has the train_mini_batch method"""
import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    This method shuffle_data
    """
    vec = np.arange(X.shape[0])
    i = np.random.permutation(vec)
    return X[i], Y[i]


def create_layer(prev, n, activation):
    """ This method create a layer for the NN """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(units=n, activation=activation,
                          kernel_initializer=kernel, name="layer")
    return lay(prev)


def forward_prop(x, layer_sizes=[], activations=[], epsilon=1e-8):
    """ This method create a forwdward prop for the NN """
    for node, activation in zip(layer_sizes, activations):
        if activation is None:
            y = create_layer(x, node, activation)
        else:
            y = create_batch_norm_layer(x, node, activation, epsilon)
        x = y
    return x


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network
    """
    operation = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train = operation.minimize(loss)
    return train


def create_batch_norm_layer(prev, n, activation, epsilon):
    """
    creates a batch normalization layer
    for a neural network a neural network
    """

    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y_pred = tf.layers.dense(prev, units=n,
                             kernel_initializer=kernel)
    mean, var = tf.nn.moments(y_pred, [0], keep_dims=True)
    gamma = tf.Variable(tf.ones([y_pred.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([y_pred.get_shape()[-1]]))
    znorm = tf.nn.batch_normalization(y_pred, mean, var, beta, gamma, epsilon)
    y_pred = activation(znorm)
    return y_pred


def create_placeholders(nx, classes):
    """ This method create a placeholders for the NN """
    x = tf.placeholder(tf.float32, [None, nx], name="x")
    y = tf.placeholder(tf.float32, [None, classes], name="y")
    return x, y


def calculate_accuracy(y, y_pred):
    """ This method calculates accuracy for the NN """
    correc = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accu = tf.reduce_mean(tf.cast(correc, tf.float32))
    return accu


def calculate_loss(y, y_pred):
    """ This method calculates loss for the NN """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates the training operation for a neural network
    """
    alpha = tf.train.inverse_time_decay(alpha, global_step,
                                        decay_step, decay_rate, True)
    return alpha


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    This method trains a loaded neural
    network model using mini-batch gradient descent
    """
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]
    m = Y_train.shape[0]
    endpos = (m // batch_size) + (m % batch_size > 0)
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, endpos)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    # print(X_train[posi:batch_size].shape)
    # print(xnew)
    for ep in range(epochs + 1):
        X_s, Y_s = shuffle_data(X_train, Y_train)
        t_cost, t_accuracy = sess.run([loss, accuracy],
                                      feed_dict={x: X_train, y: Y_train})
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
                # print(X_train[posi:posf].shape, i, posi, posf)
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
                    posf = len(X_train)
                if i % 100 == 0 and i:
                    print("\tStep {}:".format(i))
                    print("\t\tCost: {}".format(b_cost))
                    print("\t\tAccuracy: {}".format(b_accuracy))
    path = saver.save(sess, save_path)
    return path
