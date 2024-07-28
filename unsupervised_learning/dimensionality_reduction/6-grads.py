#!/usr/bin/env python3
"""grads module"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): Dataset of shape (n, ndim) containing the low
        dimensional
                           transformation of X.
        P (numpy.ndarray): Array of shape (n, n) containing the P affinities
        of X.

    Returns:
        dY (numpy.ndarray): Array of shape (n, ndim) containing the gradients
        of Y.
        Q (numpy.ndarray): Array of shape (n, n) containing the Q affinities
        of Y.
    """
    # Compute the Q affinities
    Q, num = Q_affinities(Y)

    # Compute the gradients
    PQ_diff = P - Q
    dY = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        dY[i] = np.sum(
            (PQ_diff[:, i, np.newaxis] * num[:, i, np.newaxis]) *
            (Y[i] - Y), axis=0
        )

    return dY, Q
