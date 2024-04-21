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
    # Check if the inputs are of valid types
    if isinstance(poly, list) \
        and all(isinstance(x, (int, float))
                for x in poly) and isinstance(C, (int, float)):
        if not poly:
            return [C]
        integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
        return [int(x) if isinstance(x, float) and x.is_integer()
                else x for x in integral]
    return None
