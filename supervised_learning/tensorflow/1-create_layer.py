#!/usr/bin/env python3
"""
This Module defines a function to create a neural network layer with
specified parameters using TensorFlow.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Creates a neural network layer with He et al. initialization and
    specified activation function.
    Args:
        prev (tf.Tensor): Tensor output from the previous layer.
        n (int): The number of nodes in layer to create.
    Returns:
        tf.Tensor: The tensor output of the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=initializer,
                                  name='layer')(prev)
    return layer
