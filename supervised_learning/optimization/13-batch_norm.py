#!/usr/bin/env python3
"""Module for batch normalization."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Parameters:
    Z (numpy.ndarray): Matrix of shape (m, n) to be normalized,
        where m is the number of data points and n is the number of features.
    gamma (numpy.ndarray): Vector of shape (1, n)
        containing the scales used for batch normalization.
    beta (numpy.ndarray): Vector of shape (1, n)
        containing the offsets used for batch normalization.
    epsilon (float): A small number used to avoid division by zero.

    Returns:
    numpy.ndarray: The normalized Z matrix.
    """
    # Calculate the mean and variance of Z
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Normalize Z
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Z_batch_norm = gamma * Z_normalized + beta

    return Z_batch_norm
