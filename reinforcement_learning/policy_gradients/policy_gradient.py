#!/usr/bin/env python3
"""Module to compute policy probabilities using softmax."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (probabilities of actions) using softmax.

    Args:
        matrix (np.ndarray): The state matrix of shape (1, n).
        weight (np.ndarray): The weight matrix of shape (n, m).

    Returns:
        np.ndarray: Probabilities of actions of shape (1, m).
    """
    # Compute the dot product of state and weight
    z = np.dot(matrix, weight)

    # Apply softmax to calculate probabilities
    exp_z = np.exp(z - np.max(z))
    probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return probabilities
