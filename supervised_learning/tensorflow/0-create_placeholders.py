#!/usr/bin/env python3
"""
This Module defines a function that creates placeholders for a
neural network.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates placeholders for the input data to the neural network and
    the one-hot labels.
    Args:
        nx (int): The number the feature columns in our data.
        classes (int): The number of classes in our classifies.
    Returns:
        x (tf.placeholder): Placeholder for the input data to the neural
            network.
        y (tf.placeholder): Placeholder for the one-hot  labels for the
            input data.
    """
    x = tf.placeholder(shape=(None, nx), dtype=tf.float32, name='x')
    y = tf.placeholder(shape=(None, classes), dtype=tf.float32, name='y')

    return x, y
