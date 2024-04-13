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
    # Handle empty matrices
    if not mat1 and not mat2:
        return []
    if not mat1:
        return mat2 if axis == 0 else None
    if not mat2:
        return mat1 if axis == 0 else None
    if axis == 0:
        # Check if both matrices have the same number of columns
        if all(len(row) == len(mat1[0]) for row in mat1) and \
           all(len(row) == len(mat2[0]) for row in mat2):
            return mat1 + mat2
        else:
            return None
    elif axis == 1:
        # Check if both matrices have the same number of rows
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None
    return None
