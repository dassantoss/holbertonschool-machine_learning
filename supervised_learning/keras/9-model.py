#!/usr/bin/env python3
"""
Saves and loads a Keras model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model.

    Args:
        network (keras.Model): The model to save.
        filename (str): The path of the file that the model should be saved to.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model.

    Args:
        filename (str): The path of the file that the model should be loaded
        from.

    Returns:
        keras.Model: The loaded model.
    """
    return K.models.load_model(filename)
