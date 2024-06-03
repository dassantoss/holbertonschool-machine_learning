#!/usr/bin/env python3
"""
Function to test a neural network.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network (tf.keras.Model): The network model to test.
        data (np.ndarray): The input data to test the model with.
        labels (np.ndarray): The correct one-hot labels of data.
        verbose (bool): Determines if output should be printed during the
            testing process.

    Returns:
    list: The loss and accuracy of the model with the testing data,
    respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)
