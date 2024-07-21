#!/usr/bin/env python3
"""
Module to represent a Poisson distribution.
"""


class Poisson:
    """
    Class to represent a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.

        Parameters:
        data (list): List of data points to estimate the distribution.
        lambtha (float): Expected number of occurrences in a given time frame.

        Sets the instance attribute lambtha.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
