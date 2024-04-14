#!/usr/bin/env python3
"""
Function that Slice a numpy.ndarray along specified axes.
"""


def np_slice(matrix, axes={}):
    """
    Slice a numpy.ndarray along specified axes.

    Parameters:
    matrix (numpy.ndarray): Matrix to slice.
    axes (dict): Dictionary where keys are axes to slice along and values are
    tuples representing the slice on that axis.

    Returns:
    numpy.ndarray:  A new numpy.ndarray that is the result of slicing the
    input matrix according to the axes dict.
    """
    slices = [slice(*axes.get(i, (None,))) for i in range(matrix.ndim)]
    return matrix[tuple(slices)]
