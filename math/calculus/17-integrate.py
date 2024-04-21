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
    if type(poly) is not list or len(poly) == 0:
        return None
    elif type(C) is int:
        if poly == [0]:
            return [C]
        exponent = 0
        integral = poly.copy()
        for i in range(len(integral)):
            if type(integral[i]) is int or type(integral[i]) is float:
                exponent += 1
                number = integral[i] / exponent
                integral[i] = int(number) if number % 1 == 0 else number
            else:
                return None
        integral.insert(0, C)
        return integral
    else:
        return None
