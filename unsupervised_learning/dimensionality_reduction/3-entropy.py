#!/usr/bin/env python3
"""HP module"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point.

    Args:
        Di (numpy.ndarray): Array of shape (n - 1,) containing the pairwise
        distances between a data point and all other points except itself.
        beta (numpy.ndarray): Array of shape (1,) containing the beta value
        for the Gaussian distribution.

    Returns:
        Hi (float): The Shannon entropy of the points.
        Pi (numpy.ndarray): Array of shape (n - 1,) containing the P affinities
        of the points.
    """
    # Compute the P affinities using the Gaussian kernel
    P = np.exp(-Di * beta)
    sum_P = np.sum(P)
    Pi = P / sum_P

    # Compute the Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
