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
    dense_layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)

    # Create a batch normalization layer
    batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99,
                                                    epsilon=1e-7, center=True,
                                                    scale=True)(dense_layer)

    # Apply the activation function
    return activation(batch_norm)
