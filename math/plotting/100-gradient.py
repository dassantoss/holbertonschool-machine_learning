#!/usr/bin/env python3
"""
This module Plot a scatter graph of mountain elevations using random data.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Function that Plot a scatter graph of mountain elevations using
    random data.

    Parameters:
    None

    Returns:
    None

    Example:
    >>> gradient()
    # Displays the scatter plot of mountain elevations.
    """
    # Set random seed for reproducibility
    np.random.seed(5)

    # Generate random data for x, y coordinates and elevation z
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    # Create figure with specific size
    plt.figure(figsize=(6.4, 4.8))

    # Create scatter plot with a color gradient based on z
    scatter = plt.scatter(x, y, c=z, cmap='viridis')

    # Create colorbar and label it
    cbar = plt.colorbar(scatter)
    cbar.set_label('elevation (m)')

    # Label x-axis and y-axis
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')

    # Set the title of the scatter plot
    plt.title('Mountain Elevation')

    plt.show()
