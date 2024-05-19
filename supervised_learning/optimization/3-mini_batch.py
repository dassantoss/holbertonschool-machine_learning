#!/usr/bin/env python3
"""Module for creating mini-batches for training neural networks."""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Create mini-batches from the input data X and labels Y.

    Parameters:
    X (numpy.ndarray): Input data of shape (m, nx) where m is the number of
        data points and nx is the number of features.
    Y (numpy.ndarray): Labels of shape (m, ny) where m is the number of data
        points and ny is the number of classes.
    batch_size (int): Number of data points in each batch.

    Returns:
    list: List of tuples (X_batch, Y_batch) representing mini-batches.
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
