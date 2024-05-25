#!/usr/bin/env python3
"""
This module contains a function that creates a layer of a neural network
using dropout.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    Parameters:
        prev (tensor): Tensor containing the output of the previous layer.
        n (int): Number of nodes the new layer should contain.
        activation (function): Activation function for the new layer.
        keep_prob (float): Probability that a node will be kept.
        training (bool): Boolean indicating whether the model is in training
            mode.
    Returns:
        tensor: The output of the new layer.
    """
    # Weight initialization: He et. al
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Create Dense layer
    dense = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init)

    # Apply dropout
    dropout = tf.nn.dropout(dense(prev), rate=1-keep_prob)

    return dropout
