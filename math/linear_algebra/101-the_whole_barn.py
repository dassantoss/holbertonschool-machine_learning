#!/usr/bin/env python3
"""
Function that Add two matrices of the same shape.

"""
import numpy as np


def add_matrices(mat1, mat2):
    """
    Add two matrices of the same shape.

    Parameters:
    mat1 (list or numpy.ndarray): First matrix.
    mat2 (list or numpy.ndarray): Second matrix.

    Returns:
    list or None: The sum of the two matrices if they are the same shape,
    or None if they are not.
    """
    if np.shape(mat1) != np.shape(mat2):
        return None

    def recursive_add(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return [recursive_add(sub_a, sub_b) for sub_a, sub_b in zip(a, b)]
        else:
            return a + b

    return recursive_add(mat1, mat2)
