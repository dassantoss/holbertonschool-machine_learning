#!/usr/bin/env python3
"""
Function that concatenates two matrices along a specific axis.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specified axis.

    Args:
    - mat1 (list of lists): The first matrix as a nested list.
    - mat2 (list of lists): The second matrix as a nested list.
    - axis (int): The axis along which to concatenate the matrices.

    Returns:
    - list of lists: The concatenated matrix if the dimensions are compatible.
    - None: If the matrices cannot be concatenated along the specified axis.
    """

    def recursive_concat(a, b, depth):
        # Concatenate two sub-lists when reaching the specified axis
        if depth == axis:
            return a + b
        else:
            return [recursive_concat(a[i], b[i], depth + 1)
                    for i in range(len(a))]

    def can_concatenate(a, b, depth):
        # Check compatibility for concatenation at the current recursion depth
        if depth == axis:
            return True
        elif len(a) != len(b) or not (isinstance(a, list)
                                      and isinstance(b, list)):
            return False
        else:
            return all(can_concatenate(a[i], b[i], depth + 1)
                       for i in range(len(a)))

    # Start the concatenation if possible
    if can_concatenate(mat1, mat2, 0):
        return recursive_concat(mat1, mat2, 0)
    else:
        return None
