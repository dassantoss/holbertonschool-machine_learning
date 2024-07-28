#!/usr/bin/env python3
"""Q_affinities module"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities for t-SNE.

    Args:
        Y (numpy.ndarray): Dataset of shape (n, ndim) containing the low
        dimensional
                           transformation of X.

    Returns:
        Q (numpy.ndarray): Array of shape (n, n) containing the Q affinities.
        num (numpy.ndarray): Array of shape (n, n) containing the numerator
        of the Q affinities.
    """
    # Compute the squared pairwise distances in the low-dimensional space
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    # Compute the numerator of the Q affinities
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    # Compute the denominator
    Q = num / np.sum(num)

    return Q, num
