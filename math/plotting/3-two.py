#!/usr/bin/env python3
"""
This module Plot the exponential decay line graphs.
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    This function creates a line graph that shows the exponential decay
    of two radioactive elements, Carbon-14 and Radium-226.  The graph
    plots twon line representing the fraction remaining over time.

    Parameters:
    None

    Returns:
    None.  Display a matplotlib figure.

    Example:
    >>> two()
    # This will display the line graph with two lines in matplotlib viewer.
    """
    # Step 1: Generate the x-values for the time in years
    x = np.arange(0, 21000, 1000)

    # Step 2: Calculate the y-values using de decay formula for C-14 and Ra-226
    r = np.log(0.5)
    t1 = 5730  # half-life of C-14
    t2 = 1600  # half-life of Ra-226
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    # Step 3: Set the figure size.
    plt.figure(figsize=(6.4, 4.8))

    # Step 4: Plot the two lines
    plt.plot(x, y1, 'r--', label='C-14')  # Dashed red for C-14
    plt.plot(x, y2, 'g-', label='Ra-226')  # Solid green for Ra 226

    # Step 5: Label the axes
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    # Step 6: Set the range for the axes
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Step 7: Add a legend
    plt.legend(loc='upper right')

    # Step 8:
    plt.title('Exponential Decay of Radioactive Elements')

    # Step 9: Show the plot
    plt.show()
