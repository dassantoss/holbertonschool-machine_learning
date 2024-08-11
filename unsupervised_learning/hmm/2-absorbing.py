#!/usr/bin/env python3
"""
Absorbing Markov Chain module.
This module contains a function to determine if a Markov chain is absorbing.
"""
import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P (numpy.ndarray): Square 2D array representing the transition
                           matrix, shape (n, n).

    Returns:
        bool: True if the chain is absorbing, False otherwise or on failure.
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return False
    if P.ndim != 2:
        return False

    n = P.shape[0]

    # Identify absorbent states: Diagonal entries equal to 1
    absorbent_states = np.where(np.diag(P) == 1)[0]

    # If there are no absorbent states, the chain is not absorbing
    if absorbent_states.size == 0:
        return False

    # Create an identity matrix for initial reachable states
    reachable = np.eye(n)

    # Iterate to find all reachable states
    for _ in range(n):
        reachable = np.dot(reachable, P) + reachable

    # Check if every state can reach an absorbent state
    for state in range(n):
        if not np.any(reachable[state, absorbent_states] > 0):
            return False

    return True
