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
    x, y = create_placeholders(X_train.shape[1], layer_sizes[-1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accua = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accua)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    lossr, accuar = sess.run((loss, accua), feed_dict={x: X_train, y: Y_train})
    lossv, accuav = sess.run((loss, accua), feed_dict={x: X_valid, y: Y_valid})

    for i in range(iterations + 1):
        if (i % 100) == 0:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(lossr))
            print("\tTraining Accuracy: {}".format(accuar))
            print("\tValidation Cost: {}".format(lossv))
            print("\tValidation Accuracy: {}".format(accuav))
        if (i < iterations):
            sess.run((train_op), feed_dict={x: X_train, y: Y_train})
            lossr, accuar = sess.run((loss, accua), feed_dict={x: X_train,
                                                               y: Y_train})
            lossv, accuav = sess.run((loss, accua), feed_dict={x: X_valid,
                                                               y: Y_valid})
    path = saver.save(sess, save_path)
    sess.close()
    return path
