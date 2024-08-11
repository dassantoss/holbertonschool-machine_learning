#!/usr/bin/env python3
"""
Regular Markov Chain module.
This module contains a function to calculate the steady state probabilities
of a regular Markov chain.
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): Square 2D array representing the transition
                           matrix, shape (n, n).

    Returns:
        numpy.ndarray: Steady state probabilities, shape (1, n), or None
                       on failure.
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    if P.ndim != 2:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]

    # Checking if P is regular
    square_P = np.linalg.matrix_power(P, n ** 2)
    if not np.all(square_P > 0):
        return None

    # Calculate eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the eigenvector corresponding to the eigenvalue 1
    index = np.argmin(np.abs(eigenvalues - 1))
    steady_state = eigenvectors[:, index]

    # Return the normalized value (a probability distribution)
    steady_state = steady_state / np.sum(steady_state)
    return steady_state.reshape(1, n)
