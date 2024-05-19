#!/usr/bin/env python3
"""Module for creating a RMSProp optimizer in TensorFlow."""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Set up the RMSProp optimization algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta2 (float): The RMSProp weight (discounting factor).
    epsilon (float): A small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.Optimizer: The optimizer configured with
        the given alpha, beta2, and epsilon.
    """
    return tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2,
                                       epsilon=epsilon)
