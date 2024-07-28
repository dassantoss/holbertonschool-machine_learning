#!/usr/bin/env python3
"""Module to calculate the likelihood of severe side effects given various
probabilities."""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various hypothetical
    probabilities.

    Parameters:
    - x (int): Number of patients that develop severe side effects.
    - n (int): Total number of patients observed.
    - P (numpy.ndarray): 1D array containing the various hypothetical
    probabilities.

    Returns:
    - numpy.ndarray: 1D array containing the likelihoods for each probability
    in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    binom_coeff = (np.math.factorial(n) /
                   (np.math.factorial(x) * np.math.factorial(n - x)))

    # Calculate the likelihood for each probability in P
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
