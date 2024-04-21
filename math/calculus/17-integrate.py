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
        or not isinstance(C, int) or not all(isinstance(coef, int)
                                             for coef in poly):
        return None

    # The new list of coefficients after integration
    integral = [C] + [coef / (i + 1) for i, coef in enumerate(poly)]

    # Round down if the coefficient is a whole number
    integral = [int(coef) if coef == int(coef) else coef
                for coef in integral]

    return integral
