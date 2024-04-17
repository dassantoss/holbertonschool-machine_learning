#!/usr/bin/env python3
"""
This module creates a scatter plot representing men's height vs wight using
a multivariate normal distribution for the data points.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Creates and display a scatter plot of men's height vs weight.

    Parameters:
    None

    Returns:
    None

    Example:
    >>> scatter()
    # This will display the scatter plot in a matplotlib viewer.
    """

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.scatter(x, y, color='m')
    plt.show()
