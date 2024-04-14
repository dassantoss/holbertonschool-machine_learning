#!/usr/bin/env python3
"""
Function that Concatenate two numpy.ndarrays along a specified axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy.ndarrays along a specified axis.

    Parameters:
    mat1 (numpy.ndarray): The first matrix.
    mat2 (numpy.ndarray): The second matrix.
    axis (int, optional): The axis along which to concatenate the matrices.

    Returns:
    numpy.ndarray: resulting from the concatenation of mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
