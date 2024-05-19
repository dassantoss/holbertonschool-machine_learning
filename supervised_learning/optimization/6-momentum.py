#!/usr/bin/env python3
"""
Module for creating a gradient descent with momentum optimizer in TensorFlow.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Set up the gradient descent with momentum optimization algorithm in
    TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.

    Returns:
    tf.keras.optimizers.Optimizer: The optimizer configured with the given
    alpha and beta1.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
