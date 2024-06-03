#!/usr/bin/env python3
"""
Functions to save and load model configuration in JSON format.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network (tf.keras.Model): The model whose configuration should be
            saved.
        filename (str): The path of the file that the configuration should
            be saved to.

    Returns:
        None
    """
    config = network.to_json()

    with open(filename, 'w') as json_file:
        json_file.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration.

    Args:
        filename (str): The path of the file containing the model’s
            configuration in JSON format.

    Returns:
        tf.keras.Model: The loaded model.
    """
    with open(filename, 'r') as json_file:
        network_config = json_file.read()

    return K.models.model_from_json(network_config)
