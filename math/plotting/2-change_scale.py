#!/usr/bin/env python3
"""
This module Plot an exponential decay line graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plot an exponential decay line graph of C-14 with a logarithmic y-axis.

    This function creates a line graph that shows the exponential decay
    of Carbon-14. The y-axis is logarithmic for better visualization
    over time, representing the fraction of C-14 remaining.

    Parameters:
    None

    Returns:
    None. The function will create and display a matplotlib figure.

    Example:
    >>> change_scale()
    # Displays the line graph in a viewer.
    """
    # Define the range of x-values corresponding to time in years
    x = np.arange(0, 28651, 5730)
    # Calculate the decay rate using the half-life of C-14
    r = np.log(0.5)
    t = 5730
    # Calculate the y-values using the exponential decay formula
    y = np.exp((r / t) * x)

    # Set the size of the figure
    plt.figure(figsize=(6.4, 4.8))
    # Set the title and labels of the graph
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Set the y-axis to a logarithmic scale
    plt.yscale('log')
    # Plot the line graph
    plt.plot(x, y)
    # Set the range of the x-axis
    plt.xlim(0, 28650)
    # Display the graph
    plt.show()
