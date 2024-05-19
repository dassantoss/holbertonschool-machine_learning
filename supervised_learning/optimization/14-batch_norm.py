#!/usr/bin/env python3
"""Module for creating a batch normalization layer in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    prev (tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function to be used on the output
        of the layer.

    Returns:
    tensor: The activated output for the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create a dense layer
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)

    # Create a batch normalization layer
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    mean, variance = tf.nn.moments(Z, axes=0)
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-7)

    # Apply the activation function
    return activation(Z_norm)
