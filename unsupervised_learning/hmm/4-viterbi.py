#!/usr/bin/env python3
"""
Viterbi Algorithm for Hidden Markov Model module.
This module contains a function that implements the Viterbi algorithm to
calculate the most likely sequence of hidden states given a sequence of
observations.
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    Markov model.

    Args:
        Observation (numpy.ndarray): Indexes of the observations (shape T,).
        Emission (numpy.ndarray): Emission probability matrix (shape N, M).
        Transition (numpy.ndarray): Transition probability matrix (shape N, N).
        Initial (numpy.ndarray): Initial state probability vector (shape N, 1).

    Returns:
        path (list): Most likely sequence of hidden states.
        P (float): Probability of obtaining the path sequence.
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    # Initialize the probability and path matrices
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    # Initial step: t = 0
    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Iterate through time steps
    for t in range(1, T):
        for j in range(N):
            probabilities = \
                V[:, t-1] * Transition[:, j] * Emission[j, Observation[t]]
            V[j, t] = np.max(probabilities)
            B[j, t] = np.argmax(probabilities)

    # Backtrack to find the most likely path
    path = [np.argmax(V[:, T-1])]
    for t in range(T-1, 0, -1):
        path.insert(0, B[path[0], t])

    # The probability of the most likely path
    P = np.max(V[:, T-1])

    return path, P
