#!/usr/bin/env python3
"""
Sets up Adam optimization for a Keras model with categorical crossentropy loss.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical crossentropy
    loss.

    Args:
        network (keras.Model): The model to optimize.
        alpha (float): Learning rate.
        beta1 (float): First Adam optimization parameter.
        beta2 (float): Second Adam optimization parameter.

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
