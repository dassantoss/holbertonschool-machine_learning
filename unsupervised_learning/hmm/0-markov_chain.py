#!/usr/bin/env python3
"""
Markov Chain module.
This module contains a function that calculates the probability
distribution over states of a Markov chain after a specified number of
iterations.
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular
    state after a specified number of iterations.

    Args:
        P (numpy.ndarray): Square 2D array representing the transition
                           matrix, shape (n, n).
        s (numpy.ndarray): Array representing the probability of starting
                           in each state, shape (1, n).
        t (int): Number of iterations that the Markov chain has been
                 through.

    Returns:
        numpy.ndarray: Probability distribution over states after t
                       iterations, shape (1, n), or None on failure.
    """
    # Validate input shapes
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if s.shape != (1, P.shape[0]):
        return None
    if not isinstance(t, int) or t < 1:
        return None

    # Perform the matrix multiplication for t iterations
    for _ in range(t):
        s = np.dot(s, P)

    return s
