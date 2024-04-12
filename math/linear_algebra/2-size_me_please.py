#!/usr/bin/env python3
"""
Determine the shape of a matrix as a list of its dimensions.
"""


def matrix_shape(matrix):
    """
    Determine the shape of a matrix as a list of its dimensions.

    Parameters:
    - matrix: The input matrix for which the shape needs to be calculated.

    Returns:
    A list of integers representing the dimensions of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
