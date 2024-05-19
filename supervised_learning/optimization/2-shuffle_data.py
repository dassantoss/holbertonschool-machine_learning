#!/usr/bin/env python3
"""Module for shuffling data matrices."""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle the data points in two matrices the same way.

    Parameters:
    X (numpy.ndarray): First matrix of shape (m, nx)
        where m is the number of data points and nx is the number of features.
    Y (numpy.ndarray): Second matrix of shape (m, ny)
        where m is the same number of data points as in X and ny is the number
        of features.

    Returns:
    tuple: The shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
