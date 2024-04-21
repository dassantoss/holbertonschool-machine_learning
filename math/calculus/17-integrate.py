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
    if not isinstance(poly, list) or not all(isinstance(c, int)
                                             for c in poly):
        return None

    # If the polynomial is of 0 degree (constant), return C as the integral
    if len(poly) == 0:
        return [C]
    if len(poly) == 1:
        return [C, poly[0]]

    # Integrate each term and create a new list of coefficients
    integral = [C] + [coeff / (i + 1) for i, coeff in enumerate(poly)]

    # Remove trailing zeros to minimize the length of the list
    while integral[-1] == 0 and len(integral) > 1:
        integral.pop()

    # Convert floating point numbers to integers if they are whole numbers
    integral = [int(c) if isinstance(c, float)
                and c.is_integer() else c for c in integral]

    return integral
