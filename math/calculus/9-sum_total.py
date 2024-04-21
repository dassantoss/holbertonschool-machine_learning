#!/usr/bin/env python3
"""
This module provides functionality to calculate the sum of squares of
integers up to a given number. It includes the following function:
"""


def summation_i_squared(n):
    """
    Calculate the sum of the squares of all integers from 1 to n.

    Args:
    n (int): The upper limit of the range of int to be squared and summed.

    Returns:
    int: The sum of the squares of all integers from 1 to n if n is valid.
    None: If n is not a valid integer or less than 1.

    Examples:
    >>> summation_i_squared(5)
    55

    >>> summation_i_squared(-1)
    None

    >>> summation_i_squared(10)
    385
    """
    if not isinstance(n, int) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
