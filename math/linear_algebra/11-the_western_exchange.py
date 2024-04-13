#!/usr/bin/env python3
"""
Function that transposes matrix using numpy.
"""


def np_transpose(matrix):
    """
    Calculate the shape of a numpy.ndaray.

    Parameters:
    matrix (numpy ndarray): The arrar for which to determine the shape.

    Returns:
    tuple: A tuple of integers representing the shape of the matrix.
    """
    return matrix.T
