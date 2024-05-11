#!/usr/bin/env python3
"""
This Module defines a function to create the training operation for a
neural network using gradient descent optimization in TensorFlow.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.
    Args:
        loss (tf.Tensor): The loss of the networks prediction.
        alpha (float): The learning rate.
    Returns:
        tf.Operation: An operation that trains the network using gradient
        descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
