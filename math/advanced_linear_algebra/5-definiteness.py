#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix.
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): The matrix whose definiteness is
        to be calculated.

    Returns:
        str: The definiteness of the matrix: "Positive definite",
        "Positive semi-definite", "Negative semi-definite",
        "Negative definite", or "Indefinite".

    Raises:
        TypeError: If the input is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 \
            or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if all(eigenvalues > 0):
        return "Positive definite"
    elif all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif all(eigenvalues < 0):
        return "Negative definite"
    elif all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
