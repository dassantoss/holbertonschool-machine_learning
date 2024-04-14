#!/usr/bin/env python3
"""
Function that Perform matrix multiplication using NumPy.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication using NumPy.

    Parameters:
    mat1 (numpy.ndarray): First matrix.
    mat2 (numpy.ndarray): Second matrix.

    Returns:
    numpy.ndarray: The result of the matrix multiplication of mat1 and mat2.
    """
    return np.dot(mat1, mat2)
