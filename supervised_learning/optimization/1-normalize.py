#!/usr/bin/env python3
"""Module for normalizing matrices."""
import numpy as np


def normalize(X, m, s):
    """
    Normalize a matrix X based on mean m and standard deviation s.

    Parameters:
    X (numpy.ndarray): Matrix of shape (d, nx) where d is the number
        of data points and nx is the number of features.
    m (numpy.ndarray): Vector of shape (nx,) containing the mean of
        all features of X.
    s (numpy.ndarray): Vector of shape (nx,) containing the standard
        deviation of all features of X.

    Returns:
    numpy.ndarray: The normalized X matrix.
    """
    X_normalized = (X - m) / s
    return X_normalized
