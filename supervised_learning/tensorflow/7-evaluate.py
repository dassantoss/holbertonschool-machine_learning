#!/usr/bin/env python3
"""
This module defines a function to evaluate the output of a neural network
using TensorFlow.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.
    Args:
        X (np.ndarray): Input data to evaluate.
        Y (np.ndarray): One-hot labels for X.
        save_path (str): Location to load the model from.
    Returns:
        tuple: The network’s prediction, accuracy, and loss, respectively.
    """
    # Load the graph
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(save_path + '.meta')

    with tf.Session() as sess:
        # Restore the session
        saver.restore(sess, save_path)

        # Access the saved tensors and operations
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Run evaluation
        pred, acc, cost = sess.run([y_pred, accuracy, loss],
                                   feed_dict={x: X, y: Y})

    return pred, acc, cost
