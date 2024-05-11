#!/usr/bin/env python3
"""
This module defines a function to build, train, and save a neural network
classifier using TensorFlow.
"""
import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.
    Args:
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training labels.
        X_valid (np.ndarray): Validation input data.
        Y_valid (np.ndarray): Validation labels.
        layer_sizes (list): Number of nodes in each layer of the network.
        activations (list): Activation functions for each layer.
        alpha (float): Learning rate.
        iterations (int): Number of iterations to train over.
        save_path (str): Path to save the model.
    Returns:
        str: The path where the model was saved.
    """
    tf.reset_default_graph()
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add to collection later retrieval
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            train_cost, train_acc, _ = \
                sess.run([loss, accuracy, train_op],
                         feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0 or i == iterations:
                valid_cost, valid_acc = \
                    sess.run([loss, accuracy],
                             feed_dict={x: X_valid, y: Y_valid})
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")

        saved_path = saver.save(sess, save_path)
        return saved_path
