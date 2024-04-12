#!/usr/bin/env python3
"""
Function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Parameters:
    - matrix: A list of lists representing the 2D matrix.

    Returns:
    A new matrix where the rows are the columns of the original matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
