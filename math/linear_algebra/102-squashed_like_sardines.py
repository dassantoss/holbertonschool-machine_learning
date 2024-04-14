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
        """
        Recursively concatenates sub-lists when reaching the specified axis.
        """
        if depth == axis:
            return a + b
        elif depth < axis and isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return None
            return [recursive_concat(a_sub, b_sub, depth + 1)
                    for a_sub, b_sub in zip(a, b)]
        elif depth > axis:
            return [recursive_concat(a, b, depth)]
        else:
            return None

    def can_concatenate(a, b, depth):
        """
        Checks if two sub-lists can be concatenated at the given depth.
        Ensures all higher dimensions match unless at the concatenation axis.
        """
        if depth < axis:
            if not (isinstance(a, list) and isinstance(b, list)
                    and len(a) == len(b)):
                return False
            return all(can_concatenate(a_sub, b_sub, depth + 1)
                       for a_sub, b_sub in zip(a, b))
        return True

    # Start the concatenation if matrices are compatible
    if can_concatenate(mat1, mat2, 0):
        result = recursive_concat(mat1, mat2, 0)
        if isinstance(result[0], list):
            return result
        else:
            return None
    else:
        return None
