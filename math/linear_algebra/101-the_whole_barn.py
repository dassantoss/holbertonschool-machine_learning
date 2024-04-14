#!/usr/bin/env python3
"""
Function that Add two matrices of the same shape.

"""


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
    def same_shape(a, b):
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(same_shape(x, y) for x, y in zip(a, b))
        return not (isinstance(a, list) or isinstance(b, list))

    def recursive_add(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return [recursive_add(sub_a, sub_b) for sub_a, sub_b in zip(a, b)]
        else:
            return a + b

    # Only add matrices if they have the same shape
    if same_shape(mat1, mat2):
        return recursive_add(mat1, mat2)
    else:
        return None
