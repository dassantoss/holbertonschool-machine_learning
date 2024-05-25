#!/usr/bin/env python3
"""
This module conyains a fuction that calculates the cost of a neural
network with L2 regularization using TensorFlow/Keras.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    Parameters:
        cost (tensor): Tensor containing the cost of the network
            without L2 regularization.
        model (tf.keras.Model): Keras model that includes layers
            with L2 regularization.
    Returns:
        tensor: A tensor containing the total cost for each layer of the
            network, accounting for L2 regularization.
    """
    l2_cost = cost
    for layer in model.layers:
        if (hasattr(layer, 'kernel_regularizer') and
                layer.kernel_regularizer is not None):
            l2_cost += tf.reduce_sum(layer.kernel_regularizer(layer.kernel))
        if (hasattr(layer, 'bias_regularizer') and
                layer.bias_regularizer is not None):
            l2_cost += tf.reduce_sum(layer.bias_regularizer(layer.bias))
    return l2_cost
