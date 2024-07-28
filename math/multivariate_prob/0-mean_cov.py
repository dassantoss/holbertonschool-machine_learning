#!/usr/bin/env python3
"""Module to calculate the mean and covariance of a data set."""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    - X (numpy.ndarray): Data set of shape (n, d) where n is the number of data
    points and d is the number of dimensions in each data point.

    Returns:
    - mean (numpy.ndarray): Array of shape (1, d) containing the mean of the
    data set.
    - cov (numpy.ndarray): Array of shape (d, d) containing the covariance
    matrix of the data set.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean of the data set
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate the covariance matrix
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
