#!/usr/bin/env python3
"""
This Module contains a Function that Calculate the integral of a polynomial
represented by a list of coefficients.
"""


def poly_integral(poly, C=0):
    """
    Function that Calculate the integral of a polynomial represented by a
    list of coefficients.

    Parameters:
    poly (list): List of coefficients where the index is the power of x.
    C (int): The integration constant.

    Returns:
    list: A new list of coefficients representing the integral of the
    polynomial.
    """
    # Check if input is valid
    if not isinstance(poly, list) \
        or not all(isinstance(c, int)
                   for c in poly) or not isinstance(C, int):
        return None

    # Integrate each term and create a new list of coefficients
    integral = [C] + [coeff / (i + 1) for i, coeff in enumerate(poly)]
    integral = [int(c) if isinstance(c, float) and c.is_integer()
                else c for c in integral]

    return integral
