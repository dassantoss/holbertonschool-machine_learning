#!/usr/bin/env python3
"""
Function that adds two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise and returns a new list with the results.
    If arrays are not the same length, returns None.

    Parameters:
    arr1 (list): First list of integers or floats.
    arr2 (list): Second list of integers or floats.

    Returns:
    list: A new list containing the element-wise sums of arr1 and arr2.
    None: If the input lists are not the same length.
    """
    if len(arr1) != len(arr2):
        return None
    else:
        result = []
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
