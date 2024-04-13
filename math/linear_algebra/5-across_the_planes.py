#!/usr/bin/env python3
"""
Function that adds two matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds twi 2D matrices element-wise and returns a new matrix.

    Parameters:
    mat1 (list of list of ints/floats): First matrix.
    mat2 (list of list of ints/floats): Second matrix.

    Returns:
    list of list of ints/floats: New matrix with the sums of the elements
    of mat1 and mat2.
    None: If the matrices do not have the same shape.
    """
    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        row_result = [a + b for a, b in zip(row1, row2)]
        result.append(row_result)
    return result
