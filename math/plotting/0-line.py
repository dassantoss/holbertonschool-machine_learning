#!/usr/bin/env python3
"""
This module contains the function line that plots a cubic graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot a cubic graph of the function y = x^3 where x ranges from 0 to 10.

    This function plots y = x^3 as a solid red line on the interval [0, 10].
    It sets up the figure, creates the line plot, and shows it on the screen.

    Parameters:
    None

    Returns:
    None

    Example:
    >>> line()
    # This will display a plot with a cubic curve.
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)
    plt.show()
