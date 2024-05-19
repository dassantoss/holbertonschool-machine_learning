#!/usr/bin/env python3
"""
Module to calculate normalization constants.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculate the mean and standard deviation for each feature in matrix X.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) where m is the number of data
    points and nx is the number of features.

    Returns:
    tuple of numpy.ndarray: The mean and standard deviation of each feature,
    respectively.
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
