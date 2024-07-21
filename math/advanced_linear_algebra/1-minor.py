#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix whose determinant is
        to be calculated.

    Returns:
        int or float: The determinant of the matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not square.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        cofactor = ((-1) ** c) * matrix[0][c]
        det += cofactor * determinant(sub_matrix)

    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.

    Args:
        matrix (list of lists): The matrix whose minor matrix is
        to be calculated.

    Returns:
        list of lists: The minor matrix of the input matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if n == 1:
        return [[1]]

    minor_matrix = []

    for i in range(n):
        minor_row = []
        for j in range(n):
            sub_matrix = \
                [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
