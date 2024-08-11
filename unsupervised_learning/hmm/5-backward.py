#!/usr/bin/env python3
"""
Backward Algorithm for Hidden Markov Model module.
This module contains a function that implements the backward algorithm to
calculate the likelihood of a sequence of observations given a hidden
Markov model and the backward path probabilities.
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): Indexes of the observations (shape T,).
        Emission (numpy.ndarray): Emission probability matrix (shape N, M).
        Transition (numpy.ndarray): Transition probability matrix (shape N, N).
        Initial (numpy.ndarray): Initial state probability vector (shape N, 1).

    Returns:
        P (float): Likelihood of the observations given the model.
        B (numpy.ndarray): Backward path probability matrix (shape N, T).
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    # Initialize the backward path probability matrix B
    B = np.zeros((N, T))

    # Initialize the last column of B to 1 (since B[i, T-1] = 1 for all i)
    B[:, T-1] = 1

    # Compute backward probabilities for each time step t backwards
    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t+1] * Transition[i, :] * Emission[:, Observation[t+1]]
            )

    # Calculate the probability of the observations given the model
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
