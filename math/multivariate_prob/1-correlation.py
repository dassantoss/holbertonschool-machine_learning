#!/usr/bin/env python3
"""Module to calculate a correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Parameters:
    - C (numpy.ndarray): Covariance matrix of shape (d, d) where d is the
    number of dimensions.

    Returns:
    - numpy.ndarray: Correlation matrix of shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate the standard deviations
    std_devs = np.sqrt(np.diag(C))

    # Create the outer product of the standard deviations
    outer_product = np.outer(std_devs, std_devs)

    # Calculate the correlation matrix
    corr = C / outer_product

    # Correct any numerical errors by setting the diagonal to 1
    np.fill_diagonal(corr, 1)

    return corr
