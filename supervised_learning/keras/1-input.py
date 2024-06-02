#!/usr/bin/env python3
"""
Builds a neural network with the Keras library using the functional API.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using the functional API.

    Args:
        nx (int): Number of input features.
        layers (list): List with the number of nodes in each layer.
        activations (list): List with the activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        keras.Model: The Keras model.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
