#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot labels of shape (m, classes).
        batch_size (int): Size of the batch used for mini-batch gradient
            descent.
        epochs (int): Number of passes through data for mini-batch gradient
            descent.
        verbose (bool): Determines if output should be printed during training.
        shuffle (bool): Determines whether to shuffle the batches every epoch.

    Returns:
        keras.callbacks.History: The History object generated after training
        the model.
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
