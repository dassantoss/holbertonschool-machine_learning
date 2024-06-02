#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels (numpy.ndarray): The labels to convert.
        classes (int): The number of classes.

    Returns:
        numpy.ndarray: The one-hot matrix.
    """
    one_hot_matrix = K.utils.to_categorical(labels, classes)
    return one_hot_matrix
