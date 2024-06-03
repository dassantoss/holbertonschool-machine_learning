#!/usr/bin/env python3
"""
Functions to save and load model weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model’s weights.

    Args:
        network (tf.keras.Model): The model whose weights should be saved.
        filename (str): The path of the file that the weights should be
            saved to.
        save_format (str): The format in which the weights should be saved.

    Returns:
        None
    """
    network.save_weights(filename)


def load_weights(network, filename):
    """
    Loads a model’s weights.

    Args:
        network (tf.keras.Model): The model to which the weights should be
            loaded.
        filename (str): The path of the file that the weights should be loaded
            from.

    Returns:
        None
    """
    network.load_weights(filename)
