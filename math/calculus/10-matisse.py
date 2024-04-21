#!/usr/bin/env python3
"""
This Module contains a Function that Calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Function that Calculate the derivative of a polynomial.

    Args:
    poly (list): Coefficients, index is power of x.

    Returns:
    list: Coefficients of the derivative.
    None: Input is invalid.
    [0]: Polynomial is constant.
    """
    # Check if poly is a list of integers and not empty
    if not isinstance(poly, list) or not poly or not all(isinstance(c, int)
                                                         for c in poly):
        return None
    # Handle the case for constant polynomial
    if len(poly) == 1:
        return [0]
    # Calculate derivative coefficients
    derivative = [poly[i] * i for i in range(1, len(poly))]
    return derivative
