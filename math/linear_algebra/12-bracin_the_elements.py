#!/usr/bin/env python3
"""
Function that Performs element-wise arithmetic operations.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise arithmetic operations.

    Parameters:
    mat1 (numpy.ndarray): The first matrix.
    mat2 (numpy.ndarray): The second matrix.

    Returns:
    tuple: A tuple containing the results of element-wise addition,
    substraction, multiplication, and division of mat1 and mat2.
    """
    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return (addition, subtraction, multiplication, division)
