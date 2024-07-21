#!/usr/bin/env python3
"""
Module to represent an Exponential distribution.
"""


class Exponential:
    """
    Represents an exponential distribution.
    """
    e = 2.7182818285  # Approximation of Euler's number

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution.

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
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.

        Parameters:
        x (float): The time period.

        Returns:
        float: The PDF value for x. If x is out of range (x < 0), returns 0.
        """
        if x < 0:
            return 0
        return self.lambtha * (Exponential.e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period.

        Parameters:
        x (float): The time period.

        Returns:
        float: The CDF value for x. If x is out of range (x < 0), returns 0.
        """
        if x < 0:
            return 0
        return 1 - (Exponential.e ** (-self.lambtha * x))
