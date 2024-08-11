#!/usr/bin/env python3
"""
Forward Algorithm for Hidden Markov Model module.
This module contains a function that implements the forward algorithm to
calculate the likelihood of a sequence of observations given a hidden
Markov model.
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): Indexes of the observations (shape T,).
        Emission (numpy.ndarray): Emission probability matrix (shape N, M).
        Transition (numpy.ndarray): Transition probability matrix (shape N, N).
        Initial (numpy.ndarray): Initial state probability vector (shape N, 1).

    Returns:
        P (float): Likelihood of the observations given the model.
        F (numpy.ndarray): Forward path probability matrix (shape N, T).
    """
    # Extracting dimensions
    T = Observation.shape[0]
    N = Emission.shape[0]

    # Initialize the forward path probability matrix F
    F = np.zeros((N, T))

    # Initialize the first column of F using the initial state distribution
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Compute forward probabilities for the rest of the observations
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t-1] * Transition[:, j]
            ) * Emission[j, Observation[t]]

    # The probability of the observations given the model is the sum
    # of the last column of F
    P = np.sum(F[:, T-1])

    return P, F
