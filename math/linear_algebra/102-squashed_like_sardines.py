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

    def get_shape(matrix):
        if not isinstance(matrix, list) or not matrix:
            return []
        shape = [len(matrix)]
        while isinstance(matrix[0], list):
            shape.append(len(matrix[0]))
            matrix = matrix[0]
        return shape

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i, (s1, s2) in enumerate(zip(shape1, shape2)):
        if i != axis and s1 != s2:
            return None

    if axis == 0:
        return mat1 + mat2
    else:
        return [cat_matrices(m1, m2, axis-1) for m1, m2 in zip(mat1, mat2)]
