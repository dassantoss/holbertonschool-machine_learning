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
    if not isinstance(poly, list) or not all(isinstance(x, int)
                                             for x in poly):
        return None
    if not isinstance(C, int):
        return None

    # The integral calculation
    integral = [C]
    for i in range(len(poly)):
        # Integral of x^n is x^(n+1)/(n+1)
        if poly[i] != 0:
            integral.append(poly[i] / (i + 1))

    # Convert float to int where possible
    for i in range(len(integral)):
        if integral[i] == int(integral[i]):
            integral[i] = int(integral[i])
      
    return integral
