#!/usr/bin/env python3
"""
Function that performs a matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices (mat1 and mat2).

    Parameters:
    mat1 (list of list of ints/floats): First matrix.
    mat2 (list of list of ints/floats): Second matrix.

    Returns:
    list of list of ints/floats: Resulting matrix of size m x p from
    the multiplication of mat1 by mat2.
    None: If the number of columns in mat1 does not match the number
    of rows in mat2, hence multiplication not possible.
    """
    if len(mat1[0]) != len(mat2):
        return None

    num_rows = len(mat1)
    num_cols = len(mat2[0])
    result = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_cols):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
