#!/usr/bin/env python3
"""
Function that concatenates two matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Parameters:
    mat1 (list of list of ints/floats): First matrix.
    mat2 (list of list of ints/floats): Second matrix.
    axis (int): Axis along which to concatenate the matrices (0 or 1).

    Returns:
    list of list of ints/floats: New matrix resulting from concatenatenation.
    None: If the concatenation is not posible due to dimension mismatch.
    """
    if axis == 0:
        # Check if both matrices have the same number of columns
        if not all(len(row) == len(mat1[0]) for row in mat1) or not all(len(row) == len(mat2[0]) for row in mat2):
            return None
        return mat1 + mat2
    elif axis == 1:
        # Check if both matrices have the same number of rows
        if len(mat1) != len(mat2) or not all(len(row1) == len(row2) for row1, row2 in zip(mat1, mat2)):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return None
