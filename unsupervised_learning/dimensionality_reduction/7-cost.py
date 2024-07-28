#!/usr/bin/env python3
"""cost module"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation.

    Args:
        P (numpy.ndarray): Array of shape (n, n) containing the P affinities.
        Q (numpy.ndarray): Array of shape (n, n) containing the Q affinities.

    Returns:
        C (float): The cost of the transformation.
    """
    # To avoid division by zero, use the maximum of Q and a small value
    Q = np.maximum(Q, 1e-12)
    P = np.maximum(P, 1e-12)

    # Compute the cost
    C = np.sum(P * np.log(P / Q))

    return C
