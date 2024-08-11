#!/usr/bin/env python3
"""
Baum-Welch Algorithm for Hidden Markov Models.
This module implements the Baum-Welch algorithm for training a
Hidden Markov Model. It includes the forward and backward algorithms,
and iteratively updates the transition and emission probabilities to
maximize the likelihood of the observed data.
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): Array of shape (T,) that contains the
                                     index of the observations.
        Emission (numpy.ndarray): Array of shape (N, M) containing the
                                  emission probabilities.
        Transition (numpy.ndarray): Array of shape (N, N) containing the
                                    transition probabilities.
        Initial (numpy.ndarray): Array of shape (N, 1) containing the
                                 initial state probabilities.

    Returns:
        P (float): Likelihood of the observations given the model.
        F (numpy.ndarray): Array of shape (N, T) containing the forward
                           path probabilities.
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = F[:, t-1] @ Transition * Emission[:, Observation[t]]

    P = np.sum(F[:, -1])

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): Array of shape (T,) containing the
                                     index of the observations.
        Emission (numpy.ndarray): Array of shape (N, M) containing the
                                  emission probabilities.
        Transition (numpy.ndarray): Array of shape (N, N) containing the
                                    transition probabilities.
        Initial (numpy.ndarray): Array of shape (N, 1) containing the
                                 initial state probabilities.

    Returns:
        P (float): Likelihood of the observations given the model.
        B (numpy.ndarray): Array of shape (N, T) containing the backward
                           path probabilities.
    """
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[t + 1]] *
            B[:, t + 1], axis=1)

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations (numpy.ndarray): Array of shape (T,) containing the
                                      indices of the observations.
        Transition (numpy.ndarray): Array of shape (N, N) containing the
                                    initialized transition probabilities.
        Emission (numpy.ndarray): Array of shape (N, M) containing the
                                  initialized emission probabilities.
        Initial (numpy.ndarray): Array of shape (N, 1) containing the
                                 initialized starting probabilities.
        iterations (int): Number of times expectation-maximization should
                          be performed.

    Returns:
        Transition (numpy.ndarray): The converged transition probability
                                    matrix.
        Emission (numpy.ndarray): The converged emission probability matrix.
    """
    if (not isinstance(Observations, np.ndarray) or Observations.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    N = Transition.shape[0]
    M = Emission.shape[1]
    T = Observations.shape[0]

    for _ in range(iterations):
        P_f, F = forward(Observations, Emission, Transition, Initial)
        P_b, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T-1))
        gamma = np.zeros((N, T))

        for t in range(T-1):
            xi[:, :, t] = (F[:, t, np.newaxis] * Transition *
                           Emission[:, Observations[t+1]] * B[:, t+1]) / P_f

        gamma = np.sum(xi, axis=1)
        gamma = np.hstack((gamma, (F[:, T-1] * B[:, T-1]).reshape((-1, 1)) /
                           np.sum(F[:, T-1] * B[:, T-1])))

        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))

        for k in range(M):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        Emission /= np.sum(gamma, axis=1).reshape(-1, 1)

    return Transition, Emission
