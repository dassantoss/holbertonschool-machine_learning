#!/usr/bin/env python3
"""
Function to make a prediction using a neural network.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network (tf.keras.Model): The network model to make the
            prediction with.
        data (np.ndarray): The input data to make the prediction with.
        verbose (bool): Determines if output should be printed during the
            prediction process.

    Returns:
    np.ndarray: The prediction for the data.
    """
    return network.predict(data, verbose=verbose)
